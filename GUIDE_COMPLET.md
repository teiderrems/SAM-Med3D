# SAM-Med3D - Guide Complet

## 🚀 Pipeline Complet en 3 Étapes

### 1️⃣ Télécharger le Modèle

```bash
cd ckpt
wget https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth
```

### 2️⃣ Entraîner le Modèle

```bash
python train.py \
  --batch_size 2 \
  --num_workers 4 \
  --task_name "ft_b2x1" \
  --checkpoint "ckpt/sam_med3d_turbo.pth" \
  --lr 8e-5
```

### 3️⃣ Faire de l'Inférence et Visualiser

```bash
# Inférence
python inference.py
→ Génère: results/prediction_patient_017.nii.gz

# Visualisation
python visualisation.py
→ Génère: visualisation_results/comparison.png
```

## 📋 Fichiers Importants

### Code
- **`train.py`** - Entraînement (corrigé - 5 bugs résolus)
- **`inference.py`** - Inférence sur une image
- **`visualisation.py`** - Visualisation des résultats

### Documentation
- **`README.md`** - Guide de démarrage rapide
- **`README_BUGS_FIXES.md`** - Détail des corrections

### Résultats
- **`results/`** - Fichiers de segmentation
- **`visualisation_results/`** - Images PNG de visualisation
- **`work_dir/ft_b2x1/`** - Checkpoints d'entraînement

## 🔧 Options d'Entraînement

### Standard
```bash
python train.py --batch_size 2 --num_workers 4 \
  --task_name "ft_b2x1" --checkpoint "ckpt/sam_med3d_turbo.pth" --lr 8e-5
```

### Peu de VRAM
```bash
python train.py --batch_size 1 --num_workers 2 \
  --task_name "ft_b2x1_small" --checkpoint "ckpt/sam_med3d_turbo.pth"
```

### Reprendre depuis un checkpoint
```bash
python train.py --checkpoint "work_dir/ft_b2x1/sam_model_latest.pth" --resume
```

## 📊 Résultats Entraînement

- **Logs:** `work_dir/ft_b2x1/output_*.log`
- **Checkpoints:** `work_dir/ft_b2x1/sam_model_*.pth`
- **Graphiques:** `work_dir/ft_b2x1/Loss.png`, `Dice.png`

## 🐛 Bugs Corrigés

✅ Checkpoint corrompu → Gestion gracieuse
✅ Type de données incompatible → torch.float32
✅ Dice Score → Conversion float
✅ Matplotlib CUDA → Conversion CPU
✅ Avertissements TorchIO → Supprimés

Voir `README_BUGS_FIXES.md` pour les détails.

## 🔗 Ressources

- **GitHub:** https://github.com/uni-medical/SAM-Med3D
- **Paper:** https://arxiv.org/abs/2310.15161
- **Modèle:** https://huggingface.co/blueyo0/SAM-Med3D

---

**Status:** ✅ Prêt pour production
