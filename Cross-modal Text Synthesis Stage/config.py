import os
from datetime import datetime
import torch
import random
import numpy as np

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.5
LAMBDA_CYCLE = 10
NUM_WORKERS = 1
NUM_EPOCHS = 100
LOCAL_EPOCHS = 5
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_M = "genm.pth_fed.tar"
CHECKPOINT_GEN_C = "genc.pth_fed.tar"
CHECKPOINT_CRITIC_M = "criticm_fed.pth.tar"
CHECKPOINT_CRITIC_C = "criticc_fed.pth.tar"
NUM_Xray_CLIENT = 2
NUM_CT_CLIENT = 2
NUM_MRI_CLIENT = 2


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
set_seed(0)


transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
    is_check_shapes=False
)

transforms_val = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
    is_check_shapes=False
)