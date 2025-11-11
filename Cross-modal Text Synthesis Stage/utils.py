import random, torch, os, numpy as np
import torch.nn as nn
import config
import copy

from torchmetrics.functional import peak_signal_noise_ratio as psnr_fn
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def save_side_by_side(input_tensor, output_tensor, recon_img, filename="output.png", title=None):

    input_img = input_tensor.detach().cpu().to(torch.float32) * 0.5 + 0.5
    output_img = output_tensor.detach().cpu().to(torch.float32) * 0.5 + 0.5
    recon_img = recon_img.detach().cpu().to(torch.float32) * 0.5 + 0.5

    input_img = input_img.permute(1, 2, 0).numpy()
    output_img = output_img.permute(1, 2, 0).numpy()
    recon_img = recon_img.permute(1, 2, 0).numpy()

    plt.figure(figsize=(6, 3))
    if title:
        plt.suptitle(title)

    plt.subplot(1, 3, 1)
    plt.imshow(input_img)
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(output_img)
    plt.title("Generated")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(recon_img)
    plt.title("Reconstruction")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    
def calc_psnr(x, y):
    x = (x * 0.5 + 0.5).clamp(0, 1) 
    y = (y * 0.5 + 0.5).clamp(0, 1)
    return psnr_fn(x, y, data_range=1.0).item()

def calc_ssim(x, y):
    x = (x * 0.5 + 0.5).clamp(0, 1)
    y = (y * 0.5 + 0.5).clamp(0, 1)
    return ssim_fn(x, y, data_range=1.0).item()