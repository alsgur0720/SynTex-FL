from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch

class MriCTDataset(Dataset):
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


    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        CT_img = self.CT_images[index % self.CT_len]
        Mri_img = self.Mri_images[index % self.Mri_len]

        CT_path = os.path.join(self.root_CT, CT_img)
        Mri_path = os.path.join(self.root_Mri, Mri_img)

        CT_img = np.array(Image.open(CT_path).convert("RGB"))
        Mri_img = np.array(Image.open(Mri_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=CT_img, image0=Mri_img)
            CT_img = augmentations["image"]
            Mri_img = augmentations["image0"]

        ct_feat = torch.load(os.path.join(self.feature_dir_ct, f"ct_feat_{index}.pt"))
        mri_feat = torch.load(os.path.join(self.feature_dir_mri, f"mri_feat_{index}.pt"))        
        
        
        return CT_img, Mri_img, ct_feat, mri_feat 



class MriDataset(Dataset):
    def __init__(self, root_Mri, root_cap_Mri, transform=None):
        self.root_Mri = root_Mri
        self.transform = transform
        self.root_cap_Mri = root_cap_Mri
        self.feature_dir_ct = root_cap_Mri
        self.Mri_images = os.listdir(root_Mri)
        self.Mri_len = len(self.Mri_images)
        self.feature_dir_mri = root_cap_Mri
        self.length_dataset = len(self.Mri_images)
        


    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        Mri_img = self.Mri_images[index % self.Mri_len]
        Mri_path = os.path.join(self.root_Mri, Mri_img)
        Mri_img = np.array(Image.open(Mri_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=Mri_img)
            Mri_img = augmentations["image"]

        mri_feat = torch.load(os.path.join(self.feature_dir_mri, f"mri_feat_{index}.pt"))        
        
        
        return Mri_img, mri_feat 



class CTDataset(Dataset):
    def __init__(self, root_CT, root_cap_CT, transform=None):
        self.root_CT = root_CT

        self.transform = transform
        self.root_cap_CT = root_cap_CT
        self.feature_dir_ct = root_cap_CT

        self.CT_images = os.listdir(root_CT)

        self.length_dataset = len(self.CT_images)
        self.CT_len = len(self.CT_images)


    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        CT_img = self.CT_images[index % self.CT_len]


        CT_path = os.path.join(self.root_CT, CT_img)


        CT_img = np.array(Image.open(CT_path).convert("RGB"))


        if self.transform:
            augmentations = self.transform(image=CT_img)
            CT_img = augmentations["image"]


        ct_feat = torch.load(os.path.join(self.feature_dir_ct, f"ct_feat_{index}.pt"))
     
        
        
        return CT_img, ct_feat
    
    
    
class MriDataset_val(Dataset):
    def __init__(self, root_Mri, root_cap_Mri, transform=None):
        self.root_Mri        = root_Mri
        self.transform       = transform
        self.feature_dir_mri = root_cap_Mri
        self.Mri_images      = os.listdir(root_Mri)
        self.Mri_len         = len(self.Mri_images)

    def __len__(self):
        return self.Mri_len

    def __getitem__(self, index):
        filename = self.Mri_images[index % self.Mri_len]
        name, _  = os.path.splitext(filename)
        img_pil  = Image.open(os.path.join(self.root_Mri, filename)).convert("RGB")
        orig_w, orig_h = img_pil.size

        img_np = np.array(img_pil)
        if self.transform:
            aug       = self.transform(image=img_np)
            img_tensor= aug["image"]
        else:
            img_tensor= torch.from_numpy(img_np).permute(2,0,1).float() / 255.0

        mri_feat = torch.load(os.path.join(self.feature_dir_mri, f"mri_feat_{index}.pt"))

        return img_tensor, mri_feat, name, (orig_h, orig_w)
    
    

class CTDataset_val(Dataset):
    def __init__(self, root_CT, root_cap_CT, transform=None):
        self.root_CT        = root_CT
        self.transform      = transform
        self.feature_dir_ct = root_cap_CT
        self.CT_images      = os.listdir(root_CT)
        self.CT_len         = len(self.CT_images)

    def __len__(self):
        return self.CT_len

    def __getitem__(self, index):
        filename = self.CT_images[index % self.CT_len]
        name, _  = os.path.splitext(filename)
        img_pil  = Image.open(os.path.join(self.root_CT, filename)).convert("RGB")
        orig_w, orig_h = img_pil.size

        img_np = np.array(img_pil)
        if self.transform:
            aug    = self.transform(image=img_np)
            img    = aug["image"]
        else:
            img    = torch.from_numpy(img_np).permute(2,0,1).float() / 255.0

        ct_feat = torch.load(os.path.join(self.feature_dir_ct, f"ct_feat_{index}.pt"))

        return img, ct_feat, name, (orig_h, orig_w)



class XrayDataset(Dataset):
    def __init__(self, root_Xray, root_cap_Xray, transform=None):
        self.root_Xray = root_Xray
        self.transform = transform
        self.feature_dir_xray = root_cap_Xray
        self.Xray_images = os.listdir(root_Xray)
        self.length_dataset = len(self.Xray_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        Xray_img = self.Xray_images[index]
        Xray_path = os.path.join(self.root_Xray, Xray_img)
        Xray_img = np.array(Image.open(Xray_path).convert("RGB"))
        
        if self.transform:
            Xray_img = self.transform(image=Xray_img)["image"]
            
        xray_feat = torch.load(os.path.join(self.feature_dir_xray, f"xray_feat_{index}.pt"))
        return Xray_img, xray_feat

class XrayDataset_val(Dataset):
    def __init__(self, root_Xray, root_cap_Xray, transform=None):
        self.root_Xray = root_Xray
        self.transform = transform
        self.feature_dir_xray = root_cap_Xray
        self.Xray_images = os.listdir(root_Xray)

    def __len__(self):
        return len(self.Xray_images)

    def __getitem__(self, index):
        filename = self.Xray_images[index]
        name, _ = os.path.splitext(filename)
        img_pil = Image.open(os.path.join(self.root_Xray, filename)).convert("RGB")
        orig_w, orig_h = img_pil.size
        img_np = np.array(img_pil)
        if self.transform:
            img_tensor = self.transform(image=img_np)["image"]
        else:
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        xray_feat = torch.load(os.path.join(self.feature_dir_xray, f"xray_feat_{index}.pt"))
        return img_tensor, xray_feat, name, (orig_h, orig_w)



class MriCTXRDataset(Dataset):
    def __init__(self, root_Mri, root_CT, root_Xray, cap_Mri, cap_CT, cap_Xray, transform=None):
        self.transform = transform
        self.samples = []

        # Load all CSVs and tag modality
        for root, cap_path, modality in [
            (root_Mri, cap_Mri, 'mri'),
            (root_CT, cap_CT, 'ct'),
            (root_Xray, cap_Xray, 'xray')
        ]:
            df = pd.read_csv(cap_path)
            for _, row in df.iterrows():
                image_name = row['image_name']
                text = row['caption']
                image_path = os.path.join(root, image_name)
                if os.path.exists(image_path):
                    self.samples.append({
                        'image_path': image_path,
                        'text': text,
                        'modality': modality
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # For cycle-based structure: treat it as (src_img, trg_img, src_text, trg_text)
        return image, image, sample['text'], sample['text']
