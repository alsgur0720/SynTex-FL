import os
import math
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import T5Tokenizer
from torch.nn.utils.rnn import pad_sequence

import config
from dataset import MriCTXRDataset
from clients_text import train_mri_client_text, train_ct_client_text, train_xray_client_text
from models.model.transformer import Transformer
from model.discriminator import TransformerDiscriminator


def aggregate_generators_weighted_by_score(clients_list, modality, global_epoch, prev_G_r=0.0):
    theta_list = [client[0] for client in clients_list]
    g_list     = [client[1] for client in clients_list]
    s_list     = [client[2] for client in clients_list]

    N = len(theta_list)
    updated_G_r = prev_G_r + sum(g ** 2 for g in g_list)
    sqrt_G_r = math.sqrt(updated_G_r)

    denom = sum(s_list) + sqrt_G_r
    raw_weights = [(s + g) for s, g in zip(s_list, g_list)]
    norm_weights = [w / denom for w in raw_weights]
    norm_weights = torch.softmax(torch.tensor(norm_weights), dim=0).tolist()

    os.makedirs(f"save_log/aggregation/", exist_ok=True)
    with open(f'save_log/aggregation/{modality}.txt', 'a') as f:
        f.write(f"\nEpoch {global_epoch} Aggregation Weights ({modality}):\n")
        for i, w in enumerate(norm_weights):
            f.write(f"  - Client {i+1} weight: {w:.4f}\n")

    agg_state_dict = {}
    for key in theta_list[0].state_dict().keys():
        param = sum((w * theta_list[i].state_dict()[key].float()) for i, w in enumerate(norm_weights))
        agg_state_dict[key] = param

    return agg_state_dict, updated_G_r


def train():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    pad_id = tokenizer.pad_token_id
    sos_id = pad_id
    eos_id = tokenizer.eos_token_id
    vocab_size = tokenizer.vocab_size

    num_mri_client = config.NUM_MRI_CLIENT
    num_ct_client = config.NUM_CT_CLIENT
    num_xray_client = config.NUM_XRAY_CLIENT

    # === Dataset Load & Split === #
    dataset = MriCTXRDataset(
        root_Mri=config.TRAIN_DIR + "/mri",
        root_CT=config.TRAIN_DIR + "/ct",
        root_Xray=config.TRAIN_DIR + "/xray",
        cap_Mri=config.TRAIN_DIR + "/text/pmc-15m-mri-train.csv",
        cap_CT=config.TRAIN_DIR + "/text/pmc-15m-ct-train.csv",
        cap_Xray=config.TRAIN_DIR + "/text/pmc-15m-xray-train.csv",
        transform=config.transforms
    )

    total_clients = num_mri_client + num_ct_client + num_xray_client
    total_len = len(dataset)
    unit_len = total_len // total_clients
    splits = [unit_len] * (total_clients - 1)
    splits.append(total_len - unit_len * (total_clients - 1))
    subsets = random_split(dataset, splits)

    loader_clients = [
        DataLoader(subsets[i], batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS) 
        for i in range(total_clients)
    ]

    def build_generator():
        return Transformer(
            src_pad_idx=pad_id,
            trg_pad_idx=pad_id,
            trg_sos_idx=sos_id,
            enc_voc_size=vocab_size,
            dec_voc_size=vocab_size,
            d_model=512, n_head=8, max_len=1024, ffn_hidden=2048,
            n_layers=6, drop_prob=0.1, device=config.DEVICE
        ).to(config.DEVICE)

    def build_discriminator():
        return TransformerDiscriminator(
            vocab_size=vocab_size, d_model=512, n_head=8,
            num_layers=6, max_len=1024, pad_id=pad_id
        ).to(config.DEVICE)

    G_M = [build_generator() for _ in range(num_mri_client)]
    G_C = [build_generator() for _ in range(num_ct_client)]
    G_X = [build_generator() for _ in range(num_xray_client)]
    D_M = [build_discriminator() for _ in range(num_mri_client)]
    D_C = [build_discriminator() for _ in range(num_ct_client)]
    D_X = [build_discriminator() for _ in range(num_xray_client)]

    opt_G_M = [optim.Adam(G_M[i].parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999)) for i in range(num_mri_client)]
    opt_G_C = [optim.Adam(G_C[i].parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999)) for i in range(num_ct_client)]
    opt_G_X = [optim.Adam(G_X[i].parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999)) for i in range(num_xray_client)]
    opt_D_M = [optim.Adam(D_M[i].parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999)) for i in range(num_mri_client)]
    opt_D_C = [optim.Adam(D_C[i].parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999)) for i in range(num_ct_client)]
    opt_D_X = [optim.Adam(D_X[i].parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999)) for i in range(num_xray_client)]

    L1 = nn.L1Loss()
    mse = nn.MSELoss()
    prev_G_M_r, prev_G_C_r, prev_G_X_r = 0, 0, 0

    for global_epoch in range(config.NUM_EPOCHS):
        print(f"\n🌍 Global Epoch {global_epoch+1}/{config.NUM_EPOCHS}")

        M_outs, C_outs, X_outs = [], [], []

        for i in range(num_mri_client):
            M_outs.append(train_mri_client_text(G_M[i], G_C[0], D_M[i], loader_clients[i], opt_G_M[i], opt_D_M[i], L1, mse, global_epoch, 0, i, tokenizer))

        for j in range(num_ct_client):
            C_outs.append(train_ct_client_text(G_C[j], G_M[0], D_C[j], loader_clients[num_mri_client + j], opt_G_C[j], opt_D_C[j], L1, mse, global_epoch, 0, j, tokenizer))

        for k in range(num_xray_client):
            X_outs.append(train_xray_client_text(G_X[k], G_M[0], D_X[k], loader_clients[num_mri_client + num_ct_client + k], opt_G_X[k], opt_D_X[k], L1, mse, global_epoch, 0, k, tokenizer))

        agg_M_state, prev_G_M_r = aggregate_generators_weighted_by_score(M_outs, modality='mri', global_epoch=global_epoch, prev_G_r=prev_G_M_r)
        agg_C_state, prev_G_C_r = aggregate_generators_weighted_by_score(C_outs, modality='ct', global_epoch=global_epoch, prev_G_r=prev_G_C_r)
        agg_X_state, prev_G_X_r = aggregate_generators_weighted_by_score(X_outs, modality='xray', global_epoch=global_epoch, prev_G_r=prev_G_X_r)

        os.makedirs(f"saved_weights/{global_epoch}", exist_ok=True)
        torch.save(agg_M_state, f"saved_weights/{global_epoch}/agg_mri.pt")
        torch.save(agg_C_state, f"saved_weights/{global_epoch}/agg_ct.pt")
        torch.save(agg_X_state, f"saved_weights/{global_epoch}/agg_xray.pt")


if __name__ == '__main__':
    train()
