# Copyright (c) Meta Platforms, Inc. and affiliates.
# Tous droits réservés.

# Ce code source est sous licence selon les termes de la licence trouvée dans le
# fichier LICENSE dans le répertoire racine de cet arbre de code source.

from typing import Any, Optional, Tuple, Type

import numpy as np
import torch
from torch import nn


class LayerNorm3d(nn.Module):
    """
    Normalisation de couche pour données volumétriques 3D.

    Extension de LayerNorm aux volumes 3D (tenseurs 5D au format NCDHW).
    Normalise sur la dimension des canaux tout en préservant les dimensions
    spatiales (Profondeur, Hauteur, Largeur).

    Utilisé dans le réseau de downscaling des masques pour stabiliser
    l'entraînement des convolutions 3D.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        """
        Initialise LayerNorm3d.

        Arguments:
            num_channels (int): Nombre de canaux dans le volume (dimension C).
            eps (float): Epsilon pour la stabilité numérique (évite division par zéro).
        """
        super().__init__()
        # Paramètres apprenables pour la transformation affine
        self.weight = nn.Parameter(torch.ones(num_channels))   # Facteur d'échelle γ
        self.bias = nn.Parameter(torch.zeros(num_channels))    # Décalage β
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalise un volume 3D sur la dimension des canaux.

        Formule : output = γ * (x - μ) / σ + β

        Arguments:
            x: Tenseur de forme (N, C, D, H, W) - Batch × Canaux × Prof × Haut × Larg

        Returns:
            Tenseur normalisé de même forme
        """
        # Calculer moyenne et variance sur les canaux (dim=1)
        u = x.mean(1, keepdim=True)  # Forme : (N, 1, D, H, W)
        s = (x - u).pow(2).mean(1, keepdim=True)  # Variance

        # Normalisation z-score
        x = (x - u) / torch.sqrt(s + self.eps)

        # Transformation affine apprise : γ*x + β
        # Broadcasting : weight[:, None, None, None] étend de (C,) à (C, 1, 1, 1)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

        return x


class PromptEncoder3D(nn.Module):
    """
    Encodeur d'invites pour volumes médicaux 3D dans SAM-Med3D.

    Convertit différents types d'invites utilisateur en embeddings que le
    décodeur de masques peut utiliser. Gère trois types d'invites :

    1. **Points** (sparse) : Clics positifs/négatifs dans le volume 3D
       - Utilisés pour indiquer ce qui doit être inclus/exclu du masque
       - Encodés avec des embeddings de position + type de point

    2. **Boîtes** (sparse) : Boîtes englobantes 3D définissant une région d'intérêt
       - Format : [x_min, y_min, z_min, x_max, y_max, z_max]
       - Les coins sont encodés comme des points spéciaux

    3. **Masques** (dense) : Masques basse résolution d'itérations précédentes
       - Permettent l'affinage itératif de la segmentation
       - Downscalés via un réseau de convolutions 3D

    Architecture :
    - Encodage positionnel : PositionEmbeddingRandom3D (fréquences aléatoires)
    - Embeddings de points : Lookups apprenables pour chaque type
    - Downscaling de masques : ConvNet 3D (1 → 4 → 16 → embed_dim canaux)
    """

    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int, int],
        input_image_size: Tuple[int, int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Initialise l'encodeur d'invites 3D.

        Encode les invites pour l'entrée du décodeur de masques de SAM-Med3D.
        Sépare les invites en sparse (points, boîtes) et dense (masques).

        Arguments:
          embed_dim (int): Dimension d'embedding des invites. Doit correspondre
            à la dimension attendue par le décodeur de masques (typiquement 384).

          image_embedding_size (tuple(int, int, int)): Taille spatiale de
            l'embedding d'image, au format (D, H, W). Correspond à la résolution
            de sortie de l'encodeur d'image (ex: 16×16×16 pour image 256³).

          input_image_size (tuple(int, int, int)): Taille paddée de l'image en entrée
            de l'encodeur d'image, au format (D, H, W). Utilisé pour normaliser
            les coordonnées des invites (typiquement 256×256×256).

          mask_in_chans (int): Nombre de canaux cachés utilisés pour encoder
            les masques d'entrée. Dimension intermédiaire du réseau de downscaling
            (typiquement 16).

          activation (nn.Module): Fonction d'activation à utiliser dans le
            réseau de downscaling des masques. GELU par défaut (plus lisse que ReLU).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size

        # Encodeur de position : utilise des fréquences aléatoires pour la robustesse
        # Divise embed_dim par 3 car on encode 3 dimensions (x, y, z)
        self.pe_layer = PositionEmbeddingRandom3D(embed_dim // 3)

        # Embeddings pour les points positifs et négatifs
        self.num_point_embeddings: int = 2  # 0=négatif, 1=positif
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)

        # Embedding pour les positions de padding (non-point)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        # Réseau de downscaling pour les masques d'entrée
        # Architecture : 1 → mask_in_chans//4 → mask_in_chans → embed_dim
        # Réduit la résolution spatiale de 4× via deux convolutions stride=2
        self.mask_input_size = (image_embedding_size[0], image_embedding_size[1],
                                image_embedding_size[2])
        self.mask_downscaling = nn.Sequential(
            # Première réduction : 1 → mask_in_chans//4, résolution /2
            nn.Conv3d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm3d(mask_in_chans // 4),
            activation(),
            # Deuxième réduction : mask_in_chans//4 → mask_in_chans, résolution /2
            nn.Conv3d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm3d(mask_in_chans),
            activation(),
            # Projection finale : mask_in_chans → embed_dim, résolution inchangée
            nn.Conv3d(mask_in_chans, embed_dim, kernel_size=1),
        )

        # Embedding par défaut quand aucun masque n'est fourni
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Retourne l'encodage positionnel dense pour l'ensemble du volume.

        Génère un encodage positionnel pour chaque voxel du volume d'embedding.
        Utilisé comme information spatiale par le décodeur de masques pour
        comprendre où se trouvent les différentes régions.

        L'encodage est basé sur des fréquences sinusoïdales aléatoires qui
        permettent au modèle d'apprendre des représentations spatiales robustes.

        Returns:
          torch.Tensor: Encodage positionnel de forme
            1 x (embed_dim) x (embedding_d) x (embedding_h) x (embedding_w)
            Par exemple : 1 × 384 × 16 × 16 × 16 pour un volume 256³
        """
        # Générer l'encodage pour la grille complète et ajouter dimension batch
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)  # 1xCxDxHxW

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """
        Encode les invites de points 3D en embeddings.

        Chaque point est encodé par :
        1. Son encodage positionnel (où dans le volume)
        2. Son type via un embedding appris (positif/négatif/padding)

        Arguments:
            points: Coordonnées 3D des points, forme (B, N, 3) où :
                   B = taille du batch, N = nombre de points, 3 = (x, y, z)
            labels: Étiquettes des points, forme (B, N) :
                   1 = point positif (inclure dans le masque)
                   0 = point négatif (exclure du masque)
                  -1 = point de padding (non utilisé)
            pad: Si True, ajoute un point de padding à la fin

        Returns:
            Embeddings de points de forme (B, N', embed_dim) où N' = N+1 si pad=True
        """
        # Décalage de 0.5 pour centrer les coordonnées au centre du voxel
        # (0,0,0) représente le coin, (0.5,0.5,0.5) le centre du premier voxel
        points = points + 0.5

        # Ajouter un point de padding si nécessaire (utilisé quand pas de boîtes)
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 3), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)

        # Encoder la position de chaque point
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)

        # Ajouter l'embedding de type selon l'étiquette
        # Points de padding : utiliser l'embedding "not_a_point"
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        # Points négatifs : ajouter embedding de point négatif
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        # Points positifs : ajouter embedding de point positif
        point_embedding[labels == 1] += self.point_embeddings[1].weight

        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Encode les boîtes englobantes 3D en embeddings.

        Une boîte 3D est encodée via ses deux coins opposés (8 coins → 2 suffisent).
        Chaque coin est traité comme un point spécial avec son propre embedding de type.

        Arguments:
            boxes: Boîtes 3D de forme (B, 6) au format :
                  [x_min, y_min, z_min, x_max, y_max, z_max]

        Returns:
            Embeddings de boîtes de forme (B, 2, embed_dim)
            Les 2 correspondent aux deux coins de la boîte
        """
        # Centrer les coordonnées dans les voxels
        boxes = boxes + 0.5

        # Reshape en deux coins : (B, 6) → (B, 2, 3)
        # Premier coin : (x_min, y_min, z_min)
        # Deuxième coin : (x_max, y_max, z_max)
        coords = boxes.reshape(-1, 2, 3)

        # Encoder la position de chaque coin
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)

        # Ajouter des embeddings de type spécifiques pour chaque coin
        # Note : utilise point_embeddings[2] et [3] qui doivent être définis
        # (le code original semble manquer ces embeddings dans __init__)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight  # Coin min
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight  # Coin max

        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Encode les masques d'entrée via un réseau de convolutions 3D.

        Les masques basse résolution d'itérations précédentes sont downscalés
        et transformés en embeddings denses. Ce downscaling permet :
        - De réduire la dimension spatiale (alignement avec image embeddings)
        - D'extraire des caractéristiques multi-échelles
        - De compresser l'information en embed_dim canaux

        Arguments:
            masks: Masques d'entrée de forme (B, 1, D, H, W)
                  Typiquement des masques basse résolution (64³) de l'itération précédente

        Returns:
            Embeddings de masques de forme (B, embed_dim, D', H', W')
            où D', H', W' = D/4, H/4, W/4 (deux convolutions stride=2)
        """
        # Passer à travers le réseau de downscaling
        # 1 canal → mask_in_chans//4 → mask_in_chans → embed_dim canaux
        # Résolution réduite de 4× (deux étapes de stride=2)
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Détermine la taille du batch à partir des invites fournies.

        Vérifie quelle invite est présente et utilise sa taille de batch.
        Au moins un type d'invite doit être fourni (ou retourne 1 par défaut).

        Arguments:
            points: Tuple optionnel (coords, labels) ou None
            boxes: Tenseur de boîtes optionnel ou None
            masks: Tenseur de masques optionnel ou None

        Returns:
            Taille du batch (nombre d'exemples à traiter)
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        """
        Retourne le périphérique (CPU/GPU) où se trouvent les paramètres du modèle.

        Utilise l'embedding de point comme proxy car c'est un paramètre toujours présent.
        """
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode différents types d'invites en embeddings sparse et dense.

        Pipeline de traitement :
        1. Déterminer la taille du batch
        2. Encoder les invites sparse (points, boîtes) → concaténation
        3. Encoder les invites dense (masques) ou utiliser embedding par défaut
        4. Retourner les deux types d'embeddings séparément

        La séparation sparse/dense permet au décodeur de traiter efficacement
        chaque type d'information avec des mécanismes appropriés.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) ou None): Tuple de
            (coordonnées, étiquettes) des points à encoder.
            Coordonnées : (B, N, 3), Étiquettes : (B, N)

          boxes (torch.Tensor ou None): Boîtes à encoder, forme (B, 6)
            Format : [x_min, y_min, z_min, x_max, y_max, z_max]

          masks (torch.Tensor ou None): Masques à encoder, forme (B, 1, D, H, W)
            Typiquement masques basse résolution d'itérations précédentes

        Returns:
          Tuple de deux tenseurs :

          1. torch.Tensor: **Embeddings sparse** pour les points et boîtes,
             forme (B, N, embed_dim) où N est déterminé par le nombre de
             points et boîtes d'entrée. Les embeddings sont concaténés :
             [embeddings_points, embeddings_boîtes]

          2. torch.Tensor: **Embeddings dense** pour les masques,
             forme (B, embed_dim, embed_D, embed_H, embed_W).
             Si aucun masque n'est fourni, utilise un embedding "no_mask" broadcast.
        """
        # Étape 1 : Déterminer la taille du batch
        bs = self._get_batch_size(points, boxes, masks)

        # Étape 2 : Initialiser le conteneur pour les embeddings sparse
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())

        # Encoder et concaténer les points si présents
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)

        # Encoder et concaténer les boîtes si présentes
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        # Étape 3 : Encoder les masques ou utiliser l'embedding par défaut
        if masks is not None:
            # Encoder les masques fournis via le réseau de convolutions
            dense_embeddings = self._embed_masks(masks)
        else:
            # Aucun masque fourni : utiliser l'embedding "no_mask" et broadcaster
            # Reshape de (1, embed_dim) → (1, embed_dim, 1, 1, 1)
            # Puis expand pour correspondre à la taille spatiale de l'image embedding
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1],
                self.image_embedding_size[2])

        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom3D(nn.Module):
    """
    Encodage positionnel utilisant des fréquences spatiales aléatoires.

    Cette approche d'encodage positionnel diffère des encodages sinusoïdaux fixes
    traditionnels (comme dans Transformer original) en utilisant des fréquences
    aléatoires apprises. Avantages :

    - **Robustesse** : Les fréquences aléatoires capturent mieux les patterns
      spatiaux variés dans les images médicales
    - **Flexibilité** : Peut gérer des résolutions arbitraires sans réentraînement
    - **Expressivité** : Les combinaisons sin/cos de fréquences aléatoires
      créent un espace d'embedding riche

    Méthode inspirée de "Fourier Features Let Networks Learn High Frequency
    Functions in Low Dimensional Domains" (Tancik et al., 2020).

    Pour un point 3D (x, y, z), génère un vecteur d'embedding via :
    1. Projection par matrice gaussienne aléatoire
    2. Application de fonctions sin et cos
    3. Concaténation pour obtenir un vecteur de haute dimension
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        """
        Initialise l'encodeur positionnel 3D.

        Arguments:
            num_pos_feats (int): Nombre de features de position par dimension.
                Le vecteur final aura dimension 3 * num_pos_feats * 3 car :
                - 3 dimensions spatiales (x, y, z)
                - Chaque dimension projetée en num_pos_feats features
                - Chaque feature encodée par sin, cos, sin → facteur 3
                Exemple : num_pos_feats=64 → dimension finale = 3*64*3 = 576

            scale (float, optional): Échelle de la distribution gaussienne pour
                initialiser la matrice de projection. Si None ou ≤0, utilise 1.0.
                Une échelle plus grande → fréquences plus hautes → détails plus fins.
        """
        super().__init__()
        # Valider et définir l'échelle
        if scale is None or scale <= 0.0:
            scale = 1.0

        # Créer la matrice de projection gaussienne aléatoire
        # Forme : (3, num_pos_feats) - une ligne par dimension spatiale (x, y, z)
        # Valeurs tirées de N(0, scale²) - distribution normale
        # Enregistrée comme buffer (non entraînable, sauvegardée avec le modèle)
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((3, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode des points normalisés en [0,1]³ en vecteurs positionnels.

        Pipeline d'encodage :
        1. Normaliser coords de [0,1] → [-1,1] (centrer autour de 0)
        2. Projeter via matrice gaussienne : (x,y,z) → features
        3. Multiplier par 2π pour obtenir des phases de 0 à 2π
        4. Appliquer sin et cos pour obtenir des oscillations périodiques
        5. Concaténer [sin, cos, sin] pour enrichir la représentation

        Arguments:
            coords: Coordonnées normalisées de forme (..., 3)
                   Valeurs attendues dans [0, 1] pour chaque dimension

        Returns:
            Encodages positionnels de forme (..., 3*num_pos_feats*3)
            Vecteur d'embedding de haute dimension pour chaque point
        """
        # Étape 1 : Normaliser de [0,1] → [-1,1]
        # Centrage important pour une projection symétrique
        coords = 2 * coords - 1

        # Étape 2 : Projection matricielle - produit avec matrice gaussienne
        # coords @ matrix : (..., 3) × (3, num_pos_feats) → (..., num_pos_feats)
        coords = coords @ self.positional_encoding_gaussian_matrix

        # Étape 3 : Multiplier par 2π pour convertir en phases angulaires
        # Les valeurs projetées deviennent des angles entre 0 et 2π
        coords = 2 * np.pi * coords

        # Étape 4 & 5 : Appliquer fonctions trigonométriques et concaténer
        # [sin(φ), cos(φ), sin(φ)] → redondance volontaire pour expressivité
        # Dimension finale : num_pos_feats * 3 = 3 * num_pos_feats
        # Note : le dernier sin semble redondant, peut-être une erreur originale
        return torch.cat([torch.sin(coords), torch.cos(coords), torch.sin(coords)], dim=-1)

    def forward(self, size: Tuple[int, int, int]) -> torch.Tensor:
        """
        Génère un encodage positionnel pour une grille 3D de taille spécifiée.

        Crée un encodage dense couvrant tout le volume, où chaque voxel
        a son propre vecteur positionnel unique basé sur ses coordonnées (x,y,z).

        Arguments:
            size: Taille de la grille 3D au format (D, H, W)
                 D = profondeur, H = hauteur, W = largeur
                 Exemple : (16, 16, 16) pour un volume 16³

        Returns:
            Encodage positionnel de forme (C, D, H, W) où :
            - C = 3 * num_pos_feats * 3 (dimension d'embedding)
            - D, H, W = dimensions spatiales de la grille
            Chaque position spatiale (d, h, w) a un vecteur unique de dimension C
        """
        # Extraire les dimensions
        x, y, z = size
        device: Any = self.positional_encoding_gaussian_matrix.device

        # Créer une grille 3D de uns
        grid = torch.ones((x, y, z), device=device, dtype=torch.float32)

        # Générer des coordonnées normalisées pour chaque dimension
        # cumsum crée des indices croissants : [1, 2, 3, ...] puis normalise
        # -0.5 pour centrer les coordonnées au centre des voxels
        y_embed = grid.cumsum(dim=0) - 0.5  # Coordonnées Y (profondeur)
        x_embed = grid.cumsum(dim=1) - 0.5  # Coordonnées X (largeur)
        z_embed = grid.cumsum(dim=2) - 0.5  # Coordonnées Z (hauteur)

        # Normaliser dans [0, 1]
        y_embed = y_embed / y
        x_embed = x_embed / x
        z_embed = z_embed / z

        # Empiler les trois dimensions : (D, H, W, 3)
        # Chaque position (d,h,w) a maintenant [x_norm, y_norm, z_norm]
        stacked_coords = torch.stack([x_embed, y_embed, z_embed], dim=-1)

        # Encoder toutes les positions
        pe = self._pe_encoding(stacked_coords)

        # Permuter de (D, H, W, C) → (C, D, H, W) pour format standard CNN
        return pe.permute(3, 0, 1, 2)

    def forward_with_coords(self, coords_input: torch.Tensor,
                            image_size: Tuple[int, int, int]) -> torch.Tensor:
        """
        Encode des points 3D dont les coordonnées ne sont PAS normalisées.

        Utilisé pour encoder des points spécifiques (clics utilisateur, coins de boîtes)
        plutôt qu'une grille dense complète. Normalise d'abord les coordonnées
        puis applique l'encodage positionnel.

        Arguments:
            coords_input: Coordonnées non normalisées de forme (B, N, 3)
                         B = batch size, N = nombre de points
                         Valeurs en pixels absolus dans l'image

            image_size: Taille de l'image de référence (D, H, W)
                       Utilisée pour normaliser les coordonnées

        Returns:
            Encodages des points de forme (B, N, C) où :
            - B = batch size (nombre d'exemples)
            - N = nombre de points par exemple
            - C = 3 * num_pos_feats * 3 (dimension d'embedding)
        """
        # Cloner pour ne pas modifier l'entrée originale
        coords = coords_input.clone()

        # Normaliser chaque dimension par la taille correspondante de l'image
        # Coordonnées de pixels absolus → coordonnées normalisées [0, 1]
        coords[:, :, 0] = coords[:, :, 0] / image_size[0]  # Normaliser dimension X
        coords[:, :, 1] = coords[:, :, 1] / image_size[1]  # Normaliser dimension Y
        coords[:, :, 2] = coords[:, :, 2] / image_size[2]  # Normaliser dimension Z

        # Encoder les coordonnées normalisées
        return self._pe_encoding(coords.to(torch.float))  # Forme : (B, N, C)
