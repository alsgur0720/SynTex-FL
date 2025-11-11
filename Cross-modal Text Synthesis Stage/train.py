import os
import sys
import math
import random
import numpy as np
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import config as config
from model.discriminator import TransformerDiscriminator as Discriminator
from dataset import MriDataset, CTDataset
from clients import train_ct_client, train_mri_client, train_xray_client
from models.model.transformer import Transformer as Generator
from dataset import XrayDataset


def aggregate_generators_weighted_by_score(clients_list: list, modality: str, global_epoch: int, prev_G_r=0.0):
    theta_list     = [client[0] for client in (clients_list)]           
    g_list         = [client[1] for client in (clients_list)]           
    s_list         = [client[2] for client in (clients_list)]           

    N_M = len(theta_list)
    updated_G_r = prev_G_r + sum(g ** 2 for g in g_list)
    sqrt_G_r = math.sqrt(updated_G_r)

    
    denom = sum(s_list) + sqrt_G_r
    raw_weights = [(s + g) for s, g in zip(s_list, g_list)]
    norm_weights = [w / denom for w in raw_weights]
    norm_weights = torch.softmax(torch.tensor(norm_weights), dim=0).tolist() 
    
    log_report = f'''\n🧮 Epoch({global_epoch}/{config.NUM_EPOCHS}) : Aggregation Weight Report\n
    → g_list (gradient norms)  : {g_list}\n
    → s_list (scores)          : {s_list}\n
    → (G_r)                    : {updated_G_r:.4f}\n
    → sqrt(G_r)                : {sqrt_G_r:.4f}\n
    → denom (sum(s) + sqrt(G)) : {denom:.4f}\n'''
    
    os.makedirs(f"save_log/aggregation/", exist_ok=True)
    with open(f'save_log/aggregation/{modality}.txt', 'a', encoding="utf-8") as f:
        f.write(log_report)
        
    with open(f'save_log/aggregation/{modality}.txt', 'a', encoding="utf-8") as f:
        for i, w in enumerate(norm_weights):
            f.write(f"  - Client {i+1} weight: {w:.4f} ({raw_weights[i]:.4f} / {denom:.4f})\n")
            print(f"  - Client {i+1} weight: {w:.4f} ({raw_weights[i]:.4f} / {denom:.4f})")
        f.close()
    
    agg_state_dict = {}
    for key in theta_list[0].state_dict().keys():
        param = sum((w * theta_list[i].state_dict()[key].float()) for i, w in zip(range(N_M), norm_weights))
        agg_state_dict[key] = param
    return agg_state_dict, updated_G_r


def train():
    num_mri_client = config.NUM_MRI_CLIENT
    num_ct_client = config.NUM_CT_CLIENT
    
    # set the dataset.dataloader
    mri_dataset = MriDataset(
        root_Mri=config.TRAIN_DIR + "/mri",
        root_cap_Mri=config.TRAIN_DIR + "/text", 
        transform=config.transforms
    )
    ct_dataset = CTDataset(
        root_CT=config.TRAIN_DIR + "/ct",
        root_cap_CT=config.TRAIN_DIR + "/text", 
        transform=config.transforms
    )
    
    xray_dataset = XrayDataset(
    root_Xray=config.TRAIN_DIR + "/xray",
    root_cap_Xray=config.TRAIN_DIR + "/text", 
    transform=config.transforms
)
    total_len_xray = len(xray_dataset)
    unit_len_xray = total_len_xray // config.NUM_XRAY_CLIENT
    split_sizes_xray = [unit_len_xray] * (config.NUM_XRAY_CLIENT-1) + [total_len_xray - unit_len_xray * (config.NUM_XRAY_CLIENT-1)]
    xray_dataset = random_split(xray_dataset, split_sizes_xray)

    loader_xray = [
        DataLoader(xray_dataset[i], batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True) for i in range(config.NUM_XRAY_CLIENT)
    ]
    
    print(f"total set : mri({mri_dataset.__len__()}) - ct({ct_dataset.__len__()})")
    
    # set the client data
    total_len_mri = len(mri_dataset)
    unit_len_mri = total_len_mri // num_mri_client
    split_sizes_mri = [unit_len_mri] * (num_mri_client-1) + [total_len_mri - unit_len_mri * (num_mri_client-1)]

    total_len_ct = len(ct_dataset)
    unit_len_ct = total_len_ct // num_ct_client
    split_sizes_ct = [unit_len_ct] * (num_ct_client-1) + [total_len_ct - unit_len_ct * (num_ct_client-1)]
    
    mri_dataset = random_split(mri_dataset, split_sizes_mri)
    ct_dataset = random_split(ct_dataset,  split_sizes_ct)

    # set the loader
    loader_mri = [
        DataLoader(mri_dataset[i], batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True) for i in range(0, num_mri_client)
    ]
    
    loader_ct = [
        DataLoader(ct_dataset[i], batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True) for i in range(0, num_ct_client)
    ]
    for i in range(0, num_mri_client):
        print(f'mri_client({i}) set : {mri_dataset[i].__len__()}')
    
    for i in range(0, num_ct_client):
        print(f'ct_client({i}) set : {ct_dataset[i].__len__()}')
    
    
    # set the cumulative gradient
    prev_G_M_r = 0
    prev_G_C_r = 0
    prev_G_X_r = 0
    
    # set the mri clients
    G_M = [Generator(3, 9).to(config.DEVICE) for _ in range(num_mri_client)]
    common_G_M = Generator(3, 9).to(config.DEVICE)
    opt_G_M = [optim.Adam(G_M[i].parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999)) for i in range(num_mri_client)] 
    opt_common_G_M = optim.Adam(common_G_M.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    
    # set the ct clients
    G_C = [Generator(3, 9).to(config.DEVICE) for _ in range(num_ct_client)] 
    common_G_C = Generator(3, 9).to(config.DEVICE)
    opt_G_C = [optim.Adam(G_C[i].parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999)) for i in range(num_ct_client)] 
    opt_common_G_C = optim.Adam(common_G_M.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    
    G_X = [Generator(3, 9).to(config.DEVICE) for _ in range(config.NUM_XRAY_CLIENT)]
    common_G_X = Generator(3, 9).to(config.DEVICE)
    opt_G_X = [optim.Adam(G_X[i].parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999)) for i in range(config.NUM_XRAY_CLIENT)]
    opt_common_G_X = optim.Adam(common_G_X.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    
    # set the loss
    L1 = nn.L1Loss()
    mse = nn.MSELoss()
    
    # start training rounds
    for global_epoch in range(config.NUM_EPOCHS):
        print(f"\n🌍 Global Epoch {global_epoch+1}/{config.NUM_EPOCHS}")
        
        D_M     = [Discriminator(3).to(config.DEVICE) for _ in range(num_mri_client)]
        opt_D_M = [optim.Adam(D_M[i].parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999)) for i in range(num_mri_client)]
        
        D_C     = [Discriminator(3).to(config.DEVICE) for _ in range(num_ct_client)]
        opt_D_C = [optim.Adam(D_C[i].parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999)) for i in range(num_ct_client)]
        
        D_X = [Discriminator(3).to(config.DEVICE) for _ in range(config.NUM_XRAY_CLIENT)]
        opt_D_X = [optim.Adam(D_X[i].parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999)) for i in range(config.NUM_XRAY_CLIENT)]
        
        if global_epoch == 0:
            # for MRI client
            for local_epoch in range(config.LOCAL_EPOCHS):
                print(f"🧠 MRI Client - Local Epoch {local_epoch+1}/{config.LOCAL_EPOCHS}")
                M_outs = []
                for i in range(num_mri_client):
                    M_out = train_mri_client(G_M[i], common_G_C, D_M[i], loader_mri[i], opt_G_M[i], opt_D_M[i], L1, mse, global_epoch, local_epoch, i)
                    M_outs.append(M_out)

                agg_C2M_state, update_G_M = aggregate_generators_weighted_by_score(
                    clients_list=M_outs, global_epoch=global_epoch, modality='mri', prev_G_r=prev_G_M_r
                )
            
            print(f'G{global_epoch} - MRI Clients Aggregation : Prev_G={prev_G_M_r} \t Update_G : {update_G_M}')
            prev_G_M_r = update_G_M
            
            os.makedirs(f"saved_weights/{global_epoch}", exist_ok=True)
            torch.save(agg_C2M_state, f"saved_weights/{global_epoch}/server_c2m.pt")
            
            # for CT client
            for local_epoch in range(config.LOCAL_EPOCHS):
                print(f"🧠 CT Client - Local Epoch {local_epoch+1}/{config.LOCAL_EPOCHS}")
                C_outs = []
                for i in range(num_ct_client):
                    C_out = train_ct_client(G_C[i], common_G_M, D_C[i], loader_ct[i], opt_G_C[i], opt_D_C[i], L1, mse, global_epoch, local_epoch, i)
                    C_outs.append(C_out)

                agg_M2C_state, update_G_C = aggregate_generators_weighted_by_score(
                clients_list=C_outs, global_epoch=global_epoch, modality='ct', prev_G_r=prev_G_C_r)
            
            
            print(f'G{global_epoch} - CT Clients Aggregation : Prev_G={prev_G_C_r} \t Update_G : {update_G_C}')
            prev_G_C_r = update_G_C
            
            os.makedirs(f"saved_weights/{global_epoch}", exist_ok=True)
            torch.save(agg_M2C_state, f"saved_weights/{global_epoch}/server_m2c.pt")
            
            for local_epoch in range(config.LOCAL_EPOCHS):
                print(f"🧠 X-ray Client - Local Epoch {local_epoch+1}/{config.LOCAL_EPOCHS}")
                X_outs = []
                for client_id in range(config.NUM_XRAY_CLIENT):
                    X_out = train_xray_client(
                        G_X[client_id], common_G_M, D_X[client_id], loader_xray[client_id],
                        opt_G_X[client_id], opt_D_X[client_id], L1, mse,
                        global_epoch, local_epoch, client_id
                    )
                    X_outs.append(X_out)

                agg_X_state, update_G_X = aggregate_generators_weighted_by_score(
                    clients_list=X_outs, global_epoch=global_epoch, modality='xray', prev_G_r=prev_G_X_r
                )
            
            prev_G_X_r = update_G_X
            torch.save(agg_X_state, f"saved_weights/{global_epoch}/server_x2m.pt")
            
        else:
            # intialize models
            agg_C2M_state = torch.load(f"saved_weights/{global_epoch-1}/server_c2m.pt")
            agg_C2M = [Generator(3, 9).to(config.DEVICE) for _ in  range(config.NUM_MRI_CLIENT)]
            agg_common_C2M = Generator(3, 9).to(config.DEVICE)
            
            agg_M2C_state = torch.load(f"saved_weights/{global_epoch-1}/server_m2c.pt")
            agg_M2C = [Generator(3, 9).to(config.DEVICE) for _ in  range(config.NUM_CT_CLIENT)]
            agg_common_M2C = Generator(3, 9).to(config.DEVICE)
            
            for i in range(config.NUM_CT_CLIENT):
                agg_C2M[i].load_state_dict(agg_C2M_state)
            
            for j in range(config.NUM_MRI_CLIENT):
                agg_M2C[j].load_state_dict(agg_M2C_state)
                
                
            agg_common_M2C.load_state_dict(agg_M2C_state)
            agg_common_C2M.load_state_dict(agg_C2M_state)
            
            # set the optimizer
            agg_opt_C2M = [optim.Adam(agg_C2M[i].parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999)) for i in range(config.NUM_MRI_CLIENT)]
            agg_opt_M2C = [optim.Adam(agg_M2C[i].parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999)) for i in range(config.NUM_CT_CLIENT)]
            
            # for mri clients
            for local_epoch in range(config.LOCAL_EPOCHS):
                print(f"🧠 MRI Client - Local Epoch {local_epoch+1}/{config.LOCAL_EPOCHS}")
                M_out1 = train_mri_client(agg_C2M[0], agg_common_M2C, D_M[0], loader_mri[0], agg_opt_C2M[0], opt_D_M[0], L1, mse, global_epoch, local_epoch, 0)
                M_out2 = train_mri_client(agg_C2M[1], agg_common_M2C, D_M[1], loader_mri[1], agg_opt_C2M[1], opt_D_M[1], L1, mse, global_epoch, local_epoch, 1)
                M_out3 = train_mri_client(agg_C2M[2], agg_common_M2C, D_M[2], loader_mri[2], agg_opt_C2M[2], opt_D_M[2], L1, mse, global_epoch, local_epoch, 2)
                M_out4 = train_mri_client(agg_C2M[3], agg_common_M2C, D_M[3], loader_mri[3], agg_opt_C2M[3], opt_D_M[3], L1, mse, global_epoch, local_epoch, 3)

            agg_C2M_state, update_G_M = aggregate_generators_weighted_by_score(
                clients_list=[M_out1, M_out2, M_out3, M_out4], global_epoch=global_epoch, modality='mri', prev_G_r=prev_G_M_r
            )
            
            os.makedirs(f"saved_weights/{global_epoch}", exist_ok=True)
            torch.save(agg_C2M_state, f"saved_weights/{global_epoch}/server_c2m.pt")
            
            # for ct clients
            for local_epoch in range(config.LOCAL_EPOCHS):
                print(f"🧠 CT Client - Local Epoch {local_epoch+1}/{config.LOCAL_EPOCHS}")
                C_out1 = train_ct_client(agg_M2C[0], agg_common_C2M, D_C[0], loader_ct[0], agg_opt_M2C[0], opt_D_C[0], L1, mse, global_epoch, local_epoch, 0)
                C_out2 = train_ct_client(agg_M2C[1], agg_common_C2M, D_C[1], loader_ct[1], agg_opt_M2C[1], opt_D_C[1], L1, mse, global_epoch, local_epoch, 1)
                C_out3 = train_ct_client(agg_M2C[2], agg_common_C2M, D_C[2], loader_ct[2], agg_opt_M2C[2], opt_D_C[2], L1, mse, global_epoch, local_epoch, 2)
                C_out4 = train_ct_client(agg_M2C[3], agg_common_C2M, D_C[3], loader_ct[3], agg_opt_M2C[3], opt_D_C[3], L1, mse, global_epoch, local_epoch, 3)
                
            agg_M2C_state, update_G_C = aggregate_generators_weighted_by_score(
                clients_list=[C_out1, C_out2, C_out3, C_out4], global_epoch=global_epoch,  modality='ct', prev_G_r=prev_G_C_r
            )
            
            print(f'G{global_epoch} - CT Clients Aggregation : Prev_G={prev_G_C_r} \t Update_G : {update_G_C}')
            prev_G_C_r = update_G_C
            
            os.makedirs(f"saved_weights/{global_epoch}", exist_ok=True)
            torch.save(agg_M2C_state, f"saved_weights/{global_epoch}/server_m2c.pt")
            
            
if __name__ == '__main__':
    train()