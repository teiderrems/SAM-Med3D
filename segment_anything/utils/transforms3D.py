# Copyright (c) Meta Platforms, Inc. and affiliates.
# Tous droits réservés.

# Ce code source est sous licence selon les termes de la licence trouvée dans le
# fichier LICENSE dans le répertoire racine de cet arbre de code source.

from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize  # type: ignore
from torchvision.transforms.functional import to_pil_image


class ResizeLongestSide3D:
    """
    Redimensionne les volumes 3D selon le côté le plus long à 'target_length'.

    Cette classe fournit des méthodes pour redimensionner de manière cohérente :
    - Les volumes d'images 3D (numpy arrays ou tensors PyTorch)
    - Les coordonnées de points 3D (pour les invites)
    - Les boîtes englobantes 3D

    Le redimensionnement préserve le ratio d'aspect en ajustant le côté le plus
    long à la longueur cible, puis en paddant pour obtenir un cube.

    Essentiel pour SAM3D car le modèle exige des entrées de taille fixe (256³).
    """

    def __init__(self, target_length: int) -> None:
        """
        Initialise le transformateur de redimensionnement.

        Arguments:
            target_length: Longueur cible pour le côté le plus long (ex: 256)
        """
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Redimensionne un volume d'image au format numpy array.

        Attend un tableau numpy avec forme DxHxWxC en format uint8.
        C = nombre de canaux (généralement 1 pour images médicales en niveaux de gris).

        Arguments:
            image: Volume numpy de forme (Profondeur, Hauteur, Largeur, Canaux)

        Returns:
            Volume redimensionné préservant le ratio d'aspect
        """
        # Calculer la nouvelle taille en préservant le ratio d'aspect
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        # Convertir en image PIL, redimensionner, reconvertir en numpy
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Transforme des coordonnées de points selon le redimensionnement d'image.

        Les coordonnées doivent être mises à l'échelle proportionnellement au
        redimensionnement de l'image pour rester alignées spatialement.

        Attend un tableau numpy avec longueur 3 dans la dimension finale.
        Requiert la taille d'image originale au format (D, H, W).

        Arguments:
            coords: Coordonnées 3D de forme [..., 3] où la dernière dim = (x, y, z)
            original_size: Taille originale du volume (D, H, W)

        Returns:
            Coordonnées transformées de même forme
        """
        old_d, old_h, old_w = original_size
        new_d, new_h, new_w = self.get_preprocess_shape(original_size[0], original_size[1],
                                                        self.target_length)
        # Copier pour ne pas modifier l'original
        coords = deepcopy(coords).astype(float)

        # Mettre à l'échelle chaque composante selon le ratio de redimensionnement
        coords[..., 0] = coords[..., 0] * (new_w / old_w)  # Coordonnée X
        coords[..., 1] = coords[..., 1] * (new_h / old_h)  # Coordonnée Y
        coords[..., 2] = coords[..., 2] * (new_d / old_d)  # Coordonnée Z

        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Transforme des boîtes englobantes 3D selon le redimensionnement d'image.

        Attend un tableau numpy de forme Bx6 au format:
        [x_min, y_min, z_min, x_max, y_max, z_max] pour chaque boîte.
        Requiert la taille d'image originale au format (D, H, W).

        Arguments:
            boxes: Boîtes 3D de forme (N_boxes, 6)
            original_size: Taille originale du volume (D, H, W)

        Returns:
            Boîtes transformées de forme (N_boxes, 6)
        """
        # Reshape en coins opposés : (N, 2, 3) pour appliquer apply_coords
        boxes = self.apply_coords(boxes.reshape(-1, 2, 3), original_size)
        # Remettre au format plat (N, 6)
        return boxes.reshape(-1, 6)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return F.interpolate(image,
                             target_size,
                             mode="bilinear",
                             align_corners=False,
                             antialias=True)

    def apply_coords_torch(self, coords: torch.Tensor, original_size: Tuple[int,
                                                                            ...]) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(original_size[0], original_size[1],
                                                 self.target_length)
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(self, boxes: torch.Tensor, original_size: Tuple[int,
                                                                          ...]) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
