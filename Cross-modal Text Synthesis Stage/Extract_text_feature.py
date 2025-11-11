import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import os
import clip
import config as config
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


class MriCTDataset_extract_text_feature(Dataset):
    def __init__(self, root_CT, root_Mri, root_cap_Mri, root_cap_CT, transform=None):
        self.root_CT = root_CT
        self.root_Mri = root_Mri
        self.transform = transform
        self.root_cap_Mri = root_cap_Mri
        self.root_cap_CT = root_cap_CT
        self.feature_dir_ct = root_cap_CT
        self.feature_dir_mri = root_cap_Mri

        self.CT_images = os.listdir(root_CT)
        self.Mri_images = os.listdir(root_Mri)
        self.length_dataset = max(len(self.CT_images), len(self.Mri_images))
        self.CT_len = len(self.CT_images)
        self.Mri_len = len(self.Mri_images)
        
        self.caption_df_mri = pd.read_csv(root_cap_Mri, encoding="ISO-8859-1")
        self.caption_df_mri["key"] = self.caption_df_mri["pair_id"].apply(lambda x: os.path.splitext(x)[0])
        self.caption_dict_mri = dict(zip(self.caption_df_mri["key"], self.caption_df_mri["fig_caption"]))

        self.caption_df_ct = pd.read_csv(root_cap_CT, encoding="ISO-8859-1")
        self.caption_df_ct["key"] = self.caption_df_ct["pair_id"].apply(lambda x: os.path.splitext(x)[0])
        self.caption_dict_ct = dict(zip(self.caption_df_ct["key"], self.caption_df_ct["fig_caption"]))
        

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        CT_img_name = self.CT_images[index % self.CT_len]
        Mri_img_name = self.Mri_images[index % self.Mri_len]

        CT_path = os.path.join(self.root_CT, CT_img_name)
        Mri_path = os.path.join(self.root_Mri, Mri_img_name)

        CT_img = np.array(Image.open(CT_path).convert("RGB"))
        Mri_img = np.array(Image.open(Mri_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=CT_img, image0=Mri_img)
            CT_img = augmentations["image"]
            Mri_img = augmentations["image0"]

        ct_key = os.path.splitext(CT_img_name)[0]
        mri_key = os.path.splitext(Mri_img_name)[0]

        ct_cap = self.caption_dict_ct.get(ct_key, "")
        mri_cap = self.caption_dict_mri.get(mri_key, "")

        return CT_img, Mri_img, ct_cap, mri_cap


class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name="ViT-B/32"):
        super().__init__()
        self.model, _ = clip.load(model_name, device=config.DEVICE)
        self.model.eval()

    def forward(self, texts):
        with torch.no_grad():
            tokenized = clip.tokenize(texts, truncate=True).to(config.DEVICE)
            text_features = self.model.encode_text(tokenized)
            text_features = F.normalize(text_features, dim=-1)
        return text_features 
    

tr_dataset = MriCTDataset_extract_text_feature(
        root_CT=config.TRAIN_DIR+"/ct", root_Mri=config.TRAIN_DIR+"/mri", 
        root_cap_Mri = config.TRAIN_DIR + "/text/pmc-15m-mri-train.csv", 
        root_cap_CT = config.TRAIN_DIR + "/text/pmc-15m-ct-train.csv", 
        transform=config.transforms
    )

va_dataset = MriCTDataset_extract_text_feature(
        root_CT=config.VAL_DIR+"/ct", root_Mri=config.VAL_DIR+"/mri", 
        root_cap_Mri = config.VAL_DIR + "/text/pmc-15m-mri-val.csv", 
        root_cap_CT = config.VAL_DIR + "/text/pmc-15m-ct-val.csv", 
        transform=config.transforms
    )

text_encoder = CLIPTextEncoder().to(config.DEVICE)

for idx in tqdm(range(len(tr_dataset))):
    ct_img, mri_img, ct_cap, mri_cap = tr_dataset[idx]
    
    with torch.no_grad():
        ct_feat = text_encoder([ct_cap]).cpu()
        mri_feat = text_encoder([mri_cap]).cpu()

    torch.save(ct_feat, f"{config.TRAIN_DIR}/text/ct_feat_{idx}.pt")
    torch.save(mri_feat, f"{config.TRAIN_DIR}/text/mri_feat_{idx}.pt")
    

for idx in tqdm(range(len(va_dataset))):
    ct_img, mri_img, ct_cap, mri_cap = va_dataset[idx]
    
    with torch.no_grad():
        ct_feat = text_encoder([ct_cap]).cpu()
        mri_feat = text_encoder([mri_cap]).cpu()

    torch.save(ct_feat, f"{config.VAL_DIR}/text/ct_feat_{idx}.pt")
    torch.save(mri_feat, f"{config.VAL_DIR}/text/mri_feat_{idx}.pt")