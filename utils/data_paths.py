''' Exemple-1 : lister manuellement tous les chemins d'ensembles de données '''
# img_datas = [
# 'sam3d_train/medical_data_all/COVID_lesion/COVID1920_ct',
# 'sam3d_train/medical_data_all/COVID_lesion/Chest_CT_Scans_with_COVID-19_ct',
# 'sam3d_train/medical_data_all/adrenal/WORD_ct',
# ]
''' Exemple-2 : utiliser glob pour lister automatiquement tous les chemins d'ensembles de données '''
"""import os.path as osp
from glob import glob

PROJ_DIR = osp.dirname(osp.dirname(__file__))
img_datas = glob(osp.join(PROJ_DIR, "data", "brain_pre_sam", "*", "*"))"""

img_datas = [
    "data/train/prostate/ct_PROSTATE",
]

