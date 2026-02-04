#!/usr/bin/env python3
"""
Script d'inférence SAM-Med3D amélioré
Segmente un volume 3D avec visualisation des résultats
"""

import torch
import numpy as np
import os
import sys
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Imports des utilitaires
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.infer_utils import (
    read_arr_from_nifti,
    data_preprocess,
    sam_model_infer,
    data_postprocess,
    save_numpy_to_nifti,
)

# Déterminer le device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✓ Using device: {device}")

# Import du modèle
try:
    from segment_anything.build_sam3D import sam_model_registry3D

    print("✓ Model imported successfully")
except Exception as e:
    print(f"✗ Failed to import model: {e}")
    sys.exit(1)

# Import TorchIO
try:
    import torchio as tio

    print("✓ TorchIO imported successfully")
except Exception as e:
    print(f"✗ Failed to import TorchIO: {e}")
    sys.exit(1)

# Paramètres
img_size = 128
checkpoint_path = "work_dir/ft_b2x2/sam_model_loss_best.pth"
img_path = "data/train/prostate/ct_PROSTATE/imagesTr/patient_001.nii.gz"
gt_path = "data/train/prostate/ct_PROSTATE/labelsTr/patient_001.nii.gz"
output_dir = "./results"
target_spacing = (1.5, 1.5, 1.5)
crop_size = 128
num_clicks = 1

# ===========================
# Charger modèle
# ===========================
print("Creating model SAM-Med3D...")
model = sam_model_registry3D["vit_b_ori"](checkpoint=None).to(device)
model.eval()

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("✅ Checkpoint loaded successfully")
else:
    print(f"❌ Checkpoint not found at {checkpoint_path}")
    exit(1)

# ===========================
# Vérifier fichiers
# ===========================
if not os.path.exists(img_path):
    print(f"❌ Image not found at {img_path}")
    exit(1)

if not os.path.exists(gt_path):
    print(f"⚠️ Ground truth not found at {gt_path}")
    gt_path = None

# ===========================
# Créer répertoires
# ===========================
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/visualizations", exist_ok=True)

# ===========================
# Charger et prétraiter données
# ===========================
print(f"Loading image from {img_path}...")
img_array, img_meta = read_arr_from_nifti(img_path, get_meta_info=True)
print(f"Original image shape: {img_array.shape}")

if gt_path:
    print(f"Loading ground truth from {gt_path}...")
    gt_array, gt_meta = read_arr_from_nifti(gt_path, get_meta_info=True)
    exist_categories = [int(l) for l in np.unique(gt_array) if l != 0]
    print(f"Categories found: {exist_categories}")
else:
    exist_categories = [1]
    print(f"No GT provided, using default category: {exist_categories}")

# ===========================
# Inférence par catégorie
# ===========================
final_pred_numpy = np.zeros(img_array.shape, dtype=np.uint8)

for category_idx in exist_categories:
    print(f"\n--- Processing category {category_idx} ---")

    # Créer subject
    if gt_path:
        subject = tio.Subject(
            image=tio.ScalarImage(img_path),
            label=tio.LabelMap(gt_path)
        )
    else:
        # Créer une étiquette vide si pas de GT
        empty_label_tensor = torch.zeros((1, *img_array.shape))
        subject = tio.Subject(
            image=tio.ScalarImage(img_path),
            label=tio.LabelMap(tensor=empty_label_tensor, affine=np.eye(4))
        )

    # Prétraitement
    roi_image, roi_label, meta_info = data_preprocess(
        subject,
        img_meta.copy(),
        category_index=category_idx,
        target_spacing=target_spacing,
        crop_size=crop_size
    )

    print(f"ROI image shape: {roi_image.shape}")
    print(f"ROI label shape: {roi_label.shape}")

    # Inférence
    print("Running inference...")
    with torch.no_grad():
        roi_pred_numpy, _ = sam_model_infer(
            model,
            roi_image,
            roi_gt=roi_label,
            num_clicks=num_clicks,
            prev_low_res_mask=None
        )

    print(f"ROI prediction shape: {roi_pred_numpy.shape}")
    print(f"Prediction stats - min: {roi_pred_numpy.min()}, max: {roi_pred_numpy.max()}, "
          f"sum: {roi_pred_numpy.sum()}")

    # Post-traitement
    cls_pred_original_grid = data_postprocess(roi_pred_numpy, meta_info)
    final_pred_numpy[cls_pred_original_grid == 1] = category_idx

    # ===========================
    # Visualisation toutes les slices axiales
    # ===========================
    if roi_label is not None:
        roi_label_numpy = roi_label[0, 0].cpu().numpy()
        roi_image_numpy = roi_image[0, 0].cpu().numpy()

        # Trouver les slices avec du contenu (label ou prédiction non vides)
        non_empty_slices = []
        for z in range(roi_label_numpy.shape[0]):
            if (roi_label_numpy[z] > 0).any() or (roi_pred_numpy[z] > 0).any():
                non_empty_slices.append(z)

        if len(non_empty_slices) > 0:
            print(f"✓ Found {len(non_empty_slices)} slices with content")

            # Afficher max 12 slices (3 lignes x 4 colonnes) pour les slices avec contenu
            num_display = min(12, len(non_empty_slices))
            step = max(1, len(non_empty_slices) // num_display)
            selected_slices = non_empty_slices[::step][:12]

            # Ajouter des slices intermédiaires si pas assez
            if len(selected_slices) < 6:
                step = max(1, len(non_empty_slices) // 6)
                selected_slices = non_empty_slices[::step][:12]

            # Créer une figure avec toutes les slices
            num_slices = len(selected_slices)
            cols = min(4, num_slices)
            rows = (num_slices + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
            if num_slices == 1:
                axes = [[axes]]
            elif rows == 1:
                axes = [axes]
            axes = axes.flatten()

            for idx, z_slice in enumerate(selected_slices):
                ax = axes[idx]
                img_slice = roi_image_numpy[z_slice]
                label_slice = roi_label_numpy[z_slice]
                pred_slice = roi_pred_numpy[z_slice]

                # Créer une image composite: image en grayscale + masques en couleur
                img_display = np.stack([img_slice, img_slice, img_slice], axis=-1)  # Convert to RGB
                img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-5)

                # Ajouter le label en vert
                img_display[label_slice > 0, 1] = 1.0  # Vert pour label
                img_display[label_slice > 0, 0] = 0.0
                img_display[label_slice > 0, 2] = 0.0

                # Ajouter la prédiction en rouge (par-dessus si différent du label)
                mask_only = (pred_slice > 0) & (label_slice == 0)
                img_display[mask_only, 0] = 1.0  # Rouge pour prédiction seule
                img_display[mask_only, 1] = 0.0
                img_display[mask_only, 2] = 0.0

                ax.imshow(img_display)

                # Compter les pixels
                label_count = (label_slice > 0).sum()
                pred_count = (pred_slice > 0).sum()
                overlap = ((label_slice > 0) & (pred_slice > 0)).sum()

                title = f'Slice {z_slice}\nGT:{label_count} Pred:{pred_count} Overlap:{overlap}'
                ax.set_title(title, fontsize=10)
                ax.axis('off')

            # Masquer les axes restants
            for idx in range(num_slices, len(axes)):
                axes[idx].axis('off')

            plt.tight_layout()
            viz_path = f"{output_dir}/visualizations/category_{category_idx}_all_slices.png"
            plt.savefig(viz_path, dpi=100, bbox_inches='tight')
            print(f"✅ All slices visualization saved to {viz_path}")
            plt.close()

            # Aussi créer une visualisation du slice du milieu en détail
            if len(non_empty_slices) > 0:
                middle_idx = len(non_empty_slices) // 2
                middle_slice = non_empty_slices[middle_idx]

                fig, axes = plt.subplots(1, 4, figsize=(20, 5))

                # Image brute
                axes[0].imshow(roi_image_numpy[middle_slice], cmap='gray')
                axes[0].set_title(f'Input Image (slice {middle_slice})')
                axes[0].axis('off')

                # Label seul
                axes[1].imshow(roi_label_numpy[middle_slice], cmap='Greens', alpha=0.8)
                axes[1].set_title(f'Ground Truth')
                axes[1].axis('off')

                # Prédiction seule
                axes[2].imshow(roi_pred_numpy[middle_slice], cmap='Reds', alpha=0.8)
                axes[2].set_title(f'Prediction')
                axes[2].axis('off')

                # Overlay: Vert=GT, Rouge=Pred, Jaune=Overlap
                overlay = np.zeros((*roi_image_numpy[middle_slice].shape, 3))
                overlay[roi_label_numpy[middle_slice] > 0] = [0, 1, 0]  # Vert
                overlay[roi_pred_numpy[middle_slice] > 0] = [1, 0, 0]  # Rouge
                overlap_mask = (roi_label_numpy[middle_slice] > 0) & (roi_pred_numpy[middle_slice] > 0)
                overlay[overlap_mask] = [1, 1, 0]  # Jaune

                axes[3].imshow(roi_image_numpy[middle_slice], cmap='gray', alpha=0.5)
                axes[3].imshow(overlay, alpha=0.7)
                axes[3].set_title(f'Overlay\n(Green=GT, Red=Pred, Yellow=Overlap)')
                axes[3].axis('off')

                plt.tight_layout()
                viz_path = f"{output_dir}/visualizations/category_{category_idx}_detailed.png"
                plt.savefig(viz_path, dpi=100, bbox_inches='tight')
                print(f"✅ Detailed visualization saved to {viz_path}")
                plt.close()

# ===========================
# Sauvegarder résultats
# ===========================
print("\n--- Saving results ---")
output_nii_path = f"{output_dir}/prediction_patient_017.nii.gz"
save_numpy_to_nifti(final_pred_numpy, output_nii_path, img_meta)
print(f"✅ Results saved to {output_nii_path}")

print("✅ Inference completed!")
