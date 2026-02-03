# -*- encoding: utf-8 -*-

import medim

from utils.infer_utils import validate_paired_img_gt
from utils.metric_utils import compute_metrics, print_computed_metrics

if __name__ == "__main__":
    ''' 1. préparer le modèle pré-entraîné avec un chemin local ou une URL huggingface '''
    ckpt_path = "https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth"
    # ou vous pouvez utiliser un chemin local comme :
    model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt_path)

    ''' 2. lire et pré-traiter vos données d'entrée '''
    img_path = "./test_data/amos_val_toy_data/imagesVa/amos_0013.nii.gz"
    gt_path = "./test_data/amos_val_toy_data/labelsVa/amos_0013.nii.gz"
    out_path = "./test_data/amos_val_toy_data/pred/amos_0013.nii.gz"
    
    ''' 3. inférer avec le modèle SAM-Med3D pré-entraîné '''
    print("Validation démarrée ! veuillez patienter quelques instants.")
    validate_paired_img_gt(model, img_path, gt_path, out_path, num_clicks=1)
    print("Validation terminée ! veuillez vérifier votre prédiction.")

    ''' 4. calculer les métriques de votre prédiction avec la vérité terrain '''
    metrics = compute_metrics(
        gt_path=gt_path,
        pred_path=out_path,
        metrics=['dice'],
        classes=None,
    )
    print_computed_metrics(metrics)
