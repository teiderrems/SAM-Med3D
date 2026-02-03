# Copyright (c) Meta Platforms, Inc. and affiliates.
# Tous droits réservés.

# Ce code source est sous licence selon les termes de la licence trouvée dans le
# fichier LICENSE dans le répertoire racine de cet arbre de code source.

from typing import Type

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """
    Bloc Multi-Layer Perceptron (MLP) avec activation non-linéaire.

    Architecture standard : Linear → Activation → Linear

    Ce bloc est utilisé dans de nombreux composants du modèle pour :
    - Augmenter la capacité d'expression du réseau
    - Transformer les embeddings dans différents espaces de représentation
    - Introduire de la non-linéarité entre les couches

    Structure typique dans un Transformer :
    Attention → LayerNorm → MLPBlock → LayerNorm
    """

    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Initialise le bloc MLP.

        Arguments:
            embedding_dim (int): Dimension des embeddings d'entrée et de sortie.
                               Le MLP préserve cette dimension (skip connection possible).
            mlp_dim (int): Dimension de la couche cachée intermédiaire.
                          Typiquement 4× embedding_dim pour expansion puis compression.
            act (Type[nn.Module]): Fonction d'activation non-linéaire.
                                  GELU par défaut (Gaussian Error Linear Unit).
                                  Plus lisse que ReLU, utilisée dans les Transformers modernes.
        """
        super().__init__()
        # Première couche linéaire : expansion de embedding_dim → mlp_dim
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        # Deuxième couche linéaire : compression de mlp_dim → embedding_dim
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        # Fonction d'activation (appliquée entre les deux couches)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passe avant à travers le bloc MLP.

        Pipeline : x → Linear1 → Activation → Linear2 → sortie

        Arguments:
            x: Tenseur d'entrée de forme [..., embedding_dim]

        Returns:
            Tenseur de forme [..., embedding_dim] (même forme qu'entrée)
        """
        # Expansion → Activation → Compression
        return self.lin2(self.act(self.lin1(x)))


# Adapté de https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
# Lui-même adapté de https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
class LayerNorm2d(nn.Module):
    """
    Normalisation de couche pour données spatiales 2D (images).

    La LayerNorm standard fonctionne sur la dernière dimension, mais pour les images
    au format NCHW (Batch × Canaux × Hauteur × Largeur), nous voulons normaliser
    sur la dimension des canaux tout en préservant les dimensions spatiales.

    Contrairement à BatchNorm qui normalise sur le batch, LayerNorm normalise
    chaque échantillon indépendamment, ce qui est plus stable pour petits batchs.

    Utilisé dans les architectures modernes comme Vision Transformers et ConvNeXt.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        """
        Initialise LayerNorm2d.

        Arguments:
            num_channels (int): Nombre de canaux dans l'image (dimension C).
            eps (float): Petit nombre ajouté au dénominateur pour la stabilité numérique.
                        Évite la division par zéro. Défaut : 1e-6.
        """
        super().__init__()
        # Paramètres apprenables pour la transformation affine : y = γ*x + β
        self.weight = nn.Parameter(torch.ones(num_channels))   # γ (gamma) : facteur d'échelle
        self.bias = nn.Parameter(torch.zeros(num_channels))    # β (beta) : décalage
        self.eps = eps  # Epsilon pour la stabilité numérique

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalise l'entrée sur la dimension des canaux.

        Formule : output = γ * (x - μ) / σ + β
        où μ est la moyenne et σ l'écart-type calculés sur les canaux.

        Arguments:
            x: Tenseur d'entrée de forme (N, C, H, W)
               N = taille du batch, C = canaux, H = hauteur, W = largeur

        Returns:
            Tenseur normalisé de même forme (N, C, H, W)
        """
        # Calculer la moyenne sur la dimension des canaux (dim=1)
        # keepdim=True préserve la dimension pour le broadcasting
        u = x.mean(1, keepdim=True)  # Forme : (N, 1, H, W)

        # Calculer la variance sur la dimension des canaux
        s = (x - u).pow(2).mean(1, keepdim=True)  # Forme : (N, 1, H, W)

        # Normalisation : (x - moyenne) / écart-type
        x = (x - u) / torch.sqrt(s + self.eps)

        # Transformation affine apprise : γ * x_normalisé
        # Broadcasting : weight[:, None, None] étend de (C,) à (C, 1, 1)
        y = self.weight[:, None, None] * x

        # Ajouter le biais appris : + β
        x = y + self.bias[:, None, None]

        return x
