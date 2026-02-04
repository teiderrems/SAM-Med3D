# Copyright (c) Meta Platforms, Inc. and affiliates.
# Tous droits réservés.

# Ce code source est sous licence selon les termes de la licence trouvée dans le
# fichier LICENSE dans le répertoire racine de cet arbre de code source.

from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .image_encoder3D import ImageEncoderViT3D
from .mask_decoder3D import MaskDecoder3D
from .prompt_encoder3D import PromptEncoder3D


class Sam3D(nn.Module):
    """
    Modèle SAM-Med3D pour la segmentation interactive d'images médicales 3D.

    SAM3D adapte l'architecture Segment Anything Model aux volumes médicaux 3D.
    Il prédit des masques de segmentation à partir d'un volume 3D et d'invites
    utilisateur (points, boîtes, masques précédents).

    Architecture en trois composants principaux :
    1. Image Encoder (Vision Transformer 3D) : Extrait les caractéristiques du volume
    2. Prompt Encoder : Encode les invites utilisateur en embeddings
    3. Mask Decoder : Génère les masques finaux en fusionnant image et invites

    Caractéristiques spécifiques au médical 3D :
    - Traite des volumes entiers (vs tranches 2D)
    - Gère l'espacement anisotrope des voxels
    - Optimisé pour les structures anatomiques 3D
    """

    # Seuil de binarisation pour convertir les logits en masques binaires
    mask_threshold: float = 0.0

    # Format d'image : "L" pour niveaux de gris (images médicales monochromes)
    image_format: str = "L"

    def __init__(
        self,
        image_encoder: ImageEncoderViT3D,
        prompt_encoder: PromptEncoder3D,
        mask_decoder: MaskDecoder3D,
        pixel_mean: List[float] = [123.675],
        pixel_std: List[float] = [58.395],
    ) -> None:
        """
        Initialise le modèle SAM-Med3D.

        SAM3D prédit des masques d'objets à partir d'un volume médical et d'invites d'entrée.
        L'architecture est conçue pour une segmentation interactive efficace où l'utilisateur
        peut affiner progressivement les résultats avec des clics ou d'autres invites.

        Arguments:
          image_encoder (ImageEncoderViT3D): Le backbone utilisé pour encoder le
            volume 3D en embeddings d'image qui permettent une prédiction de masque efficace.
            Basé sur Vision Transformer adapté aux données 3D.

          prompt_encoder (PromptEncoder3D): Encode divers types d'invites d'entrée :
            - Points (clics positifs/négatifs dans le volume)
            - Boîtes englobantes 3D (régions d'intérêt)
            - Masques de faible résolution (résultats d'itérations précédentes)

          mask_decoder (MaskDecoder3D): Prédit les masques à partir des embeddings
            d'image et des invites encodées. Génère des masques haute résolution
            et des scores de confiance IoU.

          pixel_mean (list(float)): Valeurs moyennes pour la normalisation des pixels
            dans le volume d'entrée. Utilisé pour standardiser l'entrée du modèle.
            Valeur par défaut de ImageNet bien que ce soit des images médicales.

          pixel_std (list(float)): Valeurs d'écart-type pour la normalisation des pixels.
            Utilisé en conjonction avec pixel_mean pour la normalisation z-score.
        """
        super().__init__()
        # Stocker les trois composants principaux du modèle
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        # Enregistrer les paramètres de normalisation comme buffers (non entraînables)
        # view(-1, 1, 1) reshape pour le broadcasting sur les dimensions spatiales 3D
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        """
        Retourne le périphérique (CPU/GPU) sur lequel le modèle est chargé.

        Utilise pixel_mean comme proxy car c'est un buffer toujours présent.
        """
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Prédit des masques de bout en bout à partir des volumes et invites fournis.

        Cette méthode gère le pipeline complet de prédiction :
        1. Prétraitement des volumes (normalisation)
        2. Extraction des embeddings d'image via l'encodeur
        3. Encodage des invites (points, boîtes, masques)
        4. Décodage des masques finaux
        5. Post-traitement (upsampling, binarisation)

        Si les invites ne sont pas connues à l'avance, il est recommandé
        d'utiliser SamPredictor plutôt que d'appeler le modèle directement,
        car SamPredictor gère mieux les embeddings réutilisables.

        Le décorateur @torch.no_grad() désactive les gradients (mode inférence).

        Arguments:
          batched_input (list(dict)): Une liste de volumes d'entrée, chacun étant un
            dictionnaire avec les clés suivantes. Une clé d'invite peut être exclue
            si elle n'est pas présente :

              'image': Le volume en tant que tenseur torch au format 1xDxHxW,
                déjà transformé pour l'entrée du modèle (normalisé, redimensionné).

              'original_size': (tuple(int, int, int)) La taille originale du
                volume avant transformation, au format (D, H, W).

              'point_coords': (torch.Tensor) Points d'invite groupés pour ce volume,
                avec forme BxNx3 (Batch × Nombre_points × Coordonnées_3D).
                Déjà transformés au cadre d'entrée du modèle.

              'point_labels': (torch.Tensor) Étiquettes groupées pour les points,
                avec forme BxN. 1 = premier plan, 0 = arrière-plan.

              'boxes': (torch.Tensor) Boîtes englobantes 3D groupées, avec forme Bx6.
                Format [x_min, y_min, z_min, x_max, y_max, z_max].
                Déjà transformées au cadre d'entrée du modèle.

              'mask_inputs': (torch.Tensor) Masques d'entrée groupés au modèle,
                au format Bx1xDxHxW. Masques basse résolution d'itérations précédentes.

          multimask_output (bool): Si le modèle doit prédire plusieurs masques
            pour désambiguïser, ou retourner un seul masque. Avec multimask=True,
            génère 3 masques candidats avec scores de qualité.

        Returns:
          (list(dict)): Une liste sur les volumes d'entrée, où chaque élément est
            un dictionnaire avec les clés suivantes :

              'masks': (torch.Tensor) Prédictions de masques binaires groupés,
                avec forme BxCxDxHxW, où B est le nombre d'invites d'entrée,
                C est déterminé par multimask_output (1 ou 3), et (D, H, W) est la
                taille originale du volume.

              'iou_predictions': (torch.Tensor) Les prédictions du modèle de la
                qualité du masque (IoU prédits), en forme BxC. Valeurs entre 0 et 1.

              'low_res_logits': (torch.Tensor) Logits basse résolution avec
                forme BxCxDxHxW. Peuvent être passés comme mask_input à des
                itérations suivantes de prédiction pour affinage itératif.
        """
        # Étape 1 : Prétraiter tous les volumes d'entrée
        # Normalisation et empilement en un batch unique
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)

        # Étape 2 : Encoder tous les volumes en une seule passe (efficace)
        # Extrait les caractéristiques visuelles de chaque volume
        image_embeddings = self.image_encoder(input_images)

        # Étape 3 : Traiter chaque volume individuellement pour les invites
        # (car chaque volume peut avoir des invites différentes)
        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            # Extraire et préparer les invites de points si présentes
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None

            # Encoder toutes les invites en embeddings sparse et dense
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,  # Points cliqués (sparse)
                boxes=image_record.get("boxes", None),  # Boîtes englobantes (sparse)
                masks=image_record.get("mask_inputs", None),  # Masques précédents (dense)
            )

            # Décoder les masques finaux en fusionnant image et invites
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),  # Ajouter dimension batch
                image_pe=self.prompt_encoder.get_dense_pe(),  # Encodage positionnel
                sparse_prompt_embeddings=sparse_embeddings,  # Embeddings sparse (points, boîtes)
                dense_prompt_embeddings=dense_embeddings,  # Embeddings dense (masques)
                multimask_output=multimask_output,  # 1 ou 3 masques de sortie
            )

            # Post-traiter les masques : upsampling vers résolution originale
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-3:],  # Taille transformée
                original_size=image_record["original_size"],  # Taille originale
            )

            # Binariser les masques selon le seuil (logits → 0/1)
            masks = masks > self.mask_threshold

            # Stocker les résultats pour ce volume
            outputs.append({
                "masks": masks,  # Masques binaires haute résolution
                "iou_predictions": iou_predictions,  # Scores de qualité
                "low_res_logits": low_res_masks,  # Pour affinage itératif
            })

        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Supprime le padding et agrandit les masques à la taille d'image originale.

        Le modèle travaille sur des volumes de taille fixe (ex: 256³) et génère
        des masques basse résolution. Cette méthode effectue :
        1. Upsampling vers la taille du modèle (256³)
        2. Suppression du padding ajouté lors du prétraitement
        3. Upsampling final vers la taille originale du volume

        Utilise l'interpolation bilinéaire pour un redimensionnement lisse.

        Arguments:
          masks (torch.Tensor): Masques groupés du mask_decoder,
            au format BxCxDxHxW. Typiquement basse résolution (64³).

          input_size (tuple(int, int, int)): La taille du volume d'entrée au
            modèle, au format (D, H, W). Utilisé pour supprimer le padding.
            C'est la taille avant padding mais après redimensionnement.

          original_size (tuple(int, int, int)): La taille originale du volume
            avant redimensionnement pour l'entrée du modèle, au format (D, H, W).
            Taille finale désirée des masques.

        Returns:
          (torch.Tensor): Masques groupés au format BxCxDxHxW, où (D, H, W)
            est donné par original_size. Masques à pleine résolution.
        """
        # Étape 1 : Upsampling vers la taille fixe de l'encodeur (ex: 256³)
        # Interpolation bilinéaire (trilinéaire en 3D) pour un upsampling lisse
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size,
             self.image_encoder.img_size),  # Tuple (256, 256, 256) ou autre
            mode="trilinear",  # Interpolation 3D lisse
            align_corners=False,  # Ne pas aligner les coins (meilleur pour upsampling)
        )

        # Étape 2 : Découper pour supprimer le padding ajouté lors du prétraitement
        # Ne garder que la région correspondant à la taille d'entrée réelle
        masks = masks[..., :input_size[0], :input_size[1], :input_size[2]]

        # Étape 3 : Upsampling final vers la taille originale du volume
        # Restaure les dimensions exactes du volume d'origine
        masks = F.interpolate(masks, original_size, mode="trilinear", align_corners=False)

        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalise les valeurs de voxels et ajoute du padding pour une entrée cubique.

        Le modèle SAM3D exige des volumes de taille fixe (ex: 256³).
        Cette méthode :
        1. Normalise les intensités des voxels (z-score)
        2. Ajoute du padding pour atteindre la taille requise

        Le padding est ajouté à droite/bas/profondeur (pas de centrage).

        Arguments:
            x: Volume d'entrée en format torch.Tensor, forme 1xDxHxW

        Returns:
            Volume normalisé et paddé, forme 1x256x256x256 (ou autre taille fixe)
        """
        # Étape 1 : Normaliser les intensités des voxels
        # Formule : (x - moyenne) / écart-type → distribution ~ N(0, 1)
        # Utilise les statistiques ImageNet bien que ce soit des images médicales
        x = (x - self.pixel_mean) / self.pixel_std

        # Étape 2 : Calculer le padding nécessaire pour chaque dimension
        d, h, w = x.shape[-3:]  # Dimensions actuelles (Profondeur, Hauteur, Largeur)
        padd = self.image_encoder.img_size - d  # Padding en profondeur
        padh = self.image_encoder.img_size - h  # Padding en hauteur
        padw = self.image_encoder.img_size - w  # Padding en largeur

        # Ajouter le padding (zéros) pour atteindre la taille fixe du modèle
        # F.pad applique le padding selon (gauche, droite, haut, bas, avant, arrière)
        # (0, padw, 0, padh, 0, padd) → padding à droite, bas et arrière seulement
        x = F.pad(x, (0, padw, 0, padh, 0, padd))

        return x
