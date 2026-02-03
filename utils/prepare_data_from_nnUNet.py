# -*- encoding: utf-8 -*-
'''
@File    :   prepare_data_from_nnUNet.py
@Time    :   2023/12/10 23:07:39
@Author  :   Haoyu Wang
@Contact :   small_dark@sina.com
@Brief   :   Pré-traiter les ensembles de données de style nnUNet en format SAM-Med3D

             Ce script convertit les données au format nnUNet (méthode de segmentation médicale)
             vers le format attendu par SAM-Med3D. Il effectue le ré-échantillonnage spatial,
             la séparation par classe et la réorganisation des fichiers.
'''

import json
import os
import os.path as osp
import shutil

import nibabel as nib
import torchio as tio
from tqdm import tqdm


def resample_nii(input_path: str,
                 output_path: str,
                 target_spacing: tuple = (1.5, 1.5, 1.5),
                 n=None,
                 reference_image=None,
                 mode="linear"):
    """
    Ré-échantillonne un fichier nii.gz à un espacement spatial spécifié.

    Cette fonction utilise torchio pour effectuer le ré-échantillonnage 3D d'images
    médicales. L'espacement (spacing) détermine la distance physique entre les voxels.
    Un espacement uniforme (comme 1.5x1.5x1.5 mm) est important pour que le modèle
    traite les images de manière cohérente.

    Paramètres:
        input_path (str): Chemin vers le fichier .nii.gz d'entrée.
        output_path (str): Chemin où sauvegarder le fichier .nii.gz ré-échantillonné.
        target_spacing (tuple): Espacement désiré pour le ré-échantillonnage en mm.
                               Par défaut (1.5, 1.5, 1.5) - un bon compromis entre
                               résolution et taille de fichier pour les images médicales.
        n (int ou list, optional): Indice(s) de classe(s) à extraire. Si fourni,
                                  ne conserve que ces classes et binarise le masque.
        reference_image (tio.ScalarImage, optional): Image de référence pour le recadrage/padding.
        mode (str): Mode d'interpolation - "linear" pour les images, "nearest" pour les masques.
                   Linear: interpolation trilinéaire (lisse)
                   Nearest: voisin le plus proche (préserve les labels)
    """
    # Charger le fichier nii.gz en tant que sujet torchio
    # torchio est une bibliothèque spécialisée pour le traitement d'images médicales 3D
    subject = tio.Subject(img=tio.ScalarImage(input_path))

    # Créer le ré-échantillonneur avec l'espacement cible
    # target_spacing définit la résolution physique finale (en mm entre voxels)
    resampler = tio.Resample(target=target_spacing, image_interpolation=mode)

    # Appliquer le ré-échantillonnage
    resampled_subject = resampler(subject)

    # Si un indice de classe spécifique est demandé, extraire et binariser
    if (n is not None):
        image = resampled_subject.img
        tensor_data = image.data

        # Convertir l'entier unique en liste pour un traitement uniforme
        if (isinstance(n, int)):
            n = [n]

        # Marquer temporairement les voxels de la classe désirée avec -1
        for ni in n:
            tensor_data[tensor_data == ni] = -1

        # Mettre tous les autres voxels à 0 (arrière-plan)
        tensor_data[tensor_data != -1] = 0

        # Convertir les voxels marqués en 1 (premier plan)
        tensor_data[tensor_data != 0] = 1

        # Créer une nouvelle image avec les données binarisées
        save_image = tio.ScalarImage(tensor=tensor_data, affine=image.affine)

        # Si une image de référence est fournie, recadrer/padder pour correspondre
        reference_size = reference_image.shape[1:]  # omettre la dimension du canal
        cropper_or_padder = tio.CropOrPad(reference_size)
        save_image = cropper_or_padder(save_image)
    else:
        # Sinon, utiliser l'image ré-échantillonnée directement
        save_image = resampled_subject.img

    # Sauvegarder l'image traitée
    save_image.save(output_path)


# Racine du répertoire contenant les ensembles de données brutes
dataset_root = "./data"

# Liste des ensembles de données à traiter
# Format nnUNet: chaque dataset contient un fichier dataset.json avec les métadonnées
dataset_list = [
    'AMOS_val',  # Dataset AMOS (Abdominal Multi-Organ Segmentation)
]

# Répertoire cible pour les données pré-traitées au format SAM-Med3D
target_dir = "./data/medical_preprocessed"

# Boucle sur chaque ensemble de données à traiter
for dataset in dataset_list:
    # Construire le chemin vers le répertoire du dataset
    dataset_dir = osp.join(dataset_root, dataset)

    # Charger les métadonnées du dataset depuis dataset.json
    # Ce fichier contient les informations sur les modalités, classes, et chemins de fichiers
    meta_info = json.load(open(osp.join(dataset_dir, "dataset.json")))

    # Afficher les informations du dataset
    print(meta_info['name'], meta_info['modality'])

    # Calculer le nombre de classes (en excluant l'arrière-plan qui est généralement la classe 0)
    num_classes = len(meta_info["labels"]) - 1
    print("num_classes:", num_classes, meta_info["labels"])

    # Créer un répertoire pour stocker les images ré-échantillonnées à 1.5mm
    # Cela évite de ré-échantillonner plusieurs fois la même image
    resample_dir = osp.join(dataset_dir, "imagesTr_1.5")
    os.makedirs(resample_dir, exist_ok=True)

    # Boucle sur chaque classe anatomique dans le dataset
    for idx, cls_name in meta_info["labels"].items():
        # Remplacer les espaces dans les noms de classe par des underscores
        # Ex: "left kidney" -> "left_kidney"
        cls_name = cls_name.replace(" ", "_")
        idx = int(idx)

        # Extraire le nom du dataset (ex: "AMOS_val" -> "val")
        dataset_name = dataset.split("_", maxsplit=1)[1]

        # Créer la structure de répertoires pour cette classe
        # Format: target_dir/nom_classe/dataset_name/imagesTr et labelsTr
        target_cls_dir = osp.join(target_dir, cls_name, dataset_name)
        target_img_dir = osp.join(target_cls_dir, "imagesTr")
        target_gt_dir = osp.join(target_cls_dir, "labelsTr")
        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_gt_dir, exist_ok=True)

        # Traiter chaque cas d'entraînement
        for item in tqdm(meta_info["training"], desc=f"{dataset_name}-{cls_name}"):
            # Obtenir les chemins de l'image et de l'annotation
            img, gt = item["image"], item["label"]

            # nnUNet ajoute "_0000" pour le premier canal/modalité
            img = osp.join(dataset_dir, img.replace(".nii.gz", "_0000.nii.gz"))
            gt = osp.join(dataset_dir, gt)

            # Ré-échantillonner l'image si ce n'est pas déjà fait
            resample_img = osp.join(resample_dir, osp.basename(img))
            if (not osp.exists(resample_img)):
                resample_nii(img, resample_img)
            img = resample_img

            # Construire les chemins de destination finaux
            target_img_path = osp.join(target_img_dir,
                                       osp.basename(img).replace("_0000.nii.gz", ".nii.gz"))
            target_gt_path = osp.join(target_gt_dir,
                                      osp.basename(gt).replace("_0000.nii.gz", ".nii.gz"))

            # Charger l'annotation pour vérifier le volume de la structure anatomique
            gt_img = nib.load(gt)

            # Obtenir l'espacement spatial (résolution en mm)
            # pixdim[1:4] contient l'espacement en X, Y, Z
            spacing = tuple(gt_img.header['pixdim'][1:4])

            # Calculer le volume d'un voxel en mm³
            spacing_voxel = spacing[0] * spacing[1] * spacing[2]

            # Obtenir les données du masque sous forme de tableau numpy
            gt_arr = gt_img.get_fdata()

            # Isoler la classe actuelle (mettre tout le reste à 0)
            gt_arr[gt_arr != idx] = 0
            gt_arr[gt_arr != 0] = 1

            # Calculer le volume réel de la structure en mm³
            # Nombre de voxels × volume d'un voxel
            volume = gt_arr.sum() * spacing_voxel

            # Ignorer les structures trop petites (< 10 mm³)
            # Les très petites régions peuvent être du bruit ou des artefacts
            if (volume < 10):
                print("skip", target_img_path)
                continue

            # Charger l'image comme référence pour le recadrage/padding
            reference_image = tio.ScalarImage(img)

            # Cas spécial pour le dataset KiTS23 (kidney tumor segmentation)
            # La classe 1 (rein) inclut les sous-classes 1, 2, 3
            if (meta_info['name'] == "kits23" and idx == 1):
                resample_nii(gt,
                             target_gt_path,
                             n=[1, 2, 3],  # Combiner plusieurs sous-classes
                             reference_image=reference_image,
                             mode="nearest")  # Utiliser "nearest" pour préserver les labels
            else:
                # Cas général: extraire une seule classe
                resample_nii(gt,
                             target_gt_path,
                             n=idx,
                             reference_image=reference_image,
                             mode="nearest")

            # Copier l'image ré-échantillonnée vers le répertoire cible
            shutil.copy(img, target_img_path)
