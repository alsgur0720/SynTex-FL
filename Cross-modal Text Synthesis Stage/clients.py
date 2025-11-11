import config as config
# clients_text.py
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import save_side_by_side

MAX_LEN = 1024

def pad_and_tokenize(batch_texts, tokenizer):
    pad_id = tokenizer.pad_token_id
    sos_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    token_ids = []
    for text in batch_texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        encoded = encoded[:MAX_LEN - 2]
        token_ids.append(torch.tensor([sos_id] + encoded + [eos_id], dtype=torch.long))
    return torch.nn.utils.rnn.pad_sequence(token_ids, batch_first=False, padding_value=pad_id).transpose(0, 1)

def train_generic_client(G, G_cond, D, loader, opt_G, opt_D, L1, mse, global_epoch, epoch, client_id, tokenizer, modality):
    G.train()
    D.train()
    total_adv_loss, total_l1_loss, grad_norm_total, score_total = 0, 0, 0, 0
    num_batches = 0

    for batch in tqdm(loader, desc=f"[{modality.upper()} Client {client_id}] Epoch {epoch}"):
        if batch is None:
            continue

        src_img, trg_img, src_texts, trg_texts = batch
        src_img = src_img.to(G.device)
        trg_img = trg_img.to(G.device)

        src_input_ids = pad_and_tokenize(src_texts, tokenizer).to(G.device)
        trg_input_ids = pad_and_tokenize(trg_texts, tokenizer).to(G.device)

        # === Discriminator ===
        with torch.no_grad():
            fake_trg_logits = G_cond(src_input_ids, trg_input_ids, src_img)
        recon_logits = G(fake_trg_logits.argmax(dim=-1), trg_input_ids, trg_img)

        pad_mask = trg_input_ids == tokenizer.pad_token_id
        D_real = D(trg_input_ids, pad_mask, is_soft=False)
        D_fake = D(recon_logits.detach().argmax(dim=-1), pad_mask, is_soft=True)

        loss_D = (mse(D_real, torch.ones_like(D_real)) + mse(D_fake, torch.zeros_like(D_fake))) / 2
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # === Generator ===
        adv_loss = mse(D(recon_logits.argmax(dim=-1), pad_mask, is_soft=True), torch.ones_like(D_real))
        ce_loss = F.cross_entropy(recon_logits.view(-1, recon_logits.size(-1)), trg_input_ids.view(-1), ignore_index=tokenizer.pad_token_id)
        G_loss = adv_loss + config.LAMBDA_IDENTITY * ce_loss

        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

        total_adv_loss += adv_loss.item()
        total_l1_loss += ce_loss.item()
        grad_norm = sum((p.grad.norm(2).item())**2 for p in G.parameters() if p.grad is not None) ** 0.5
        grad_norm_total += grad_norm
        score_total += D_real.mean().item()
        num_batches += 1

    avg_grad_norm = grad_norm_total / num_batches
    avg_score = score_total / num_batches
    return [G.cpu().eval(), avg_grad_norm, avg_score]


def train_mri_client_text(*args, **kwargs):
    return train_generic_client(*args, modality="mri", **kwargs)

def train_ct_client_text(*args, **kwargs):
    return train_generic_client(*args, modality="ct", **kwargs)

def train_xray_client_text(*args, **kwargs):
    return train_generic_client(*args, modality="xray", **kwargs)
