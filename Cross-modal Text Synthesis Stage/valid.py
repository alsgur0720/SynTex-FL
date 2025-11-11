# valid.py (text-to-text style for all modalities)
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer
from dataset import MriCTXRDataset
import config


def validate_clients(model, loader, tokenizer, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for batch_idx, (src_img, trg_img, src_texts, trg_texts) in enumerate(tqdm(loader)):
            src_img = src_img.to(config.DEVICE)
            src_input_ids = pad_and_tokenize(src_texts, tokenizer).to(config.DEVICE)
            outputs = model.generate(src_input_ids, src_img)
            outputs_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for i, (pred, truth) in enumerate(zip(outputs_text, trg_texts)):
                with open(os.path.join(output_dir, f"val_{batch_idx}_{i}.txt"), 'w') as f:
                    f.write(f"GT: {truth}\nPred: {pred}\n")


def pad_and_tokenize(batch_texts, tokenizer):
    pad_id = tokenizer.pad_token_id
    sos_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    token_ids = []
    for text in batch_texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        encoded = encoded[:1022]
        token_ids.append(torch.tensor([sos_id] + encoded + [eos_id], dtype=torch.long))
    return torch.nn.utils.rnn.pad_sequence(token_ids, batch_first=False, padding_value=pad_id).transpose(0, 1)


if __name__ == '__main__':
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    dataset = MriCTXRDataset(
        root_Mri=config.VALID_DIR + "/mri",
        root_CT=config.VALID_DIR + "/ct",
        root_Xray=config.VALID_DIR + "/xray",
        cap_Mri=config.VALID_DIR + "/text/pmc-15m-mri-val.csv",
        cap_CT=config.VALID_DIR + "/text/pmc-15m-ct-val.csv",
        cap_Xray=config.VALID_DIR + "/text/pmc-15m-xray-val.csv",
        transform=config.transforms
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)

    # 예시: 학습된 모델 로딩
    from models.model.transformer import Transformer
    model = Transformer(
        src_pad_idx=tokenizer.pad_token_id,
        trg_pad_idx=tokenizer.pad_token_id,
        trg_sos_idx=tokenizer.pad_token_id,
        enc_voc_size=tokenizer.vocab_size,
        dec_voc_size=tokenizer.vocab_size,
        d_model=512, n_head=8, max_len=1024, ffn_hidden=2048,
        n_layers=6, drop_prob=0.1, device=config.DEVICE
    ).to(config.DEVICE)
    model.load_state_dict(torch.load("saved_weights/last_model.pt"))
    validate_clients(model, loader, tokenizer, output_dir="val_results")
