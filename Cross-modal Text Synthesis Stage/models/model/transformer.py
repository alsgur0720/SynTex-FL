import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder


class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)



        
        self.image_proj = nn.Conv2d(
            in_channels=3, out_channels=d_model, kernel_size=3, stride=4, padding=1
        )

        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_head, batch_first=True
        )

    def forward(self, src, trg, image):
        
        

        src_mask = self.make_src_mask(src)    # [B, 1, 1, src_len]
        trg_mask = self.make_trg_mask(trg)    # [B, 1, trg_len, trg_len]

        img_feat = self.image_proj(image)  # [B, D, H', W']
        B, D, H1, W1 = img_feat.shape
        img_query = img_feat.flatten(2).transpose(1, 2)  # [B, N_q, D]


        enc_src = self.encoder(src, src_mask)




        img_query_attn, _ = self.cross_attn(img_query, enc_src, enc_src)  # [B, N_q, D]


        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def make_src_mask(self, src):
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, src_len]

    def make_trg_mask(self, trg):
        batch_size, trg_len = trg.size()
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, trg_len]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()  # [trg_len, trg_len]
        trg_mask = trg_pad_mask & trg_sub_mask  # [B, 1, trg_len, trg_len]
        return trg_mask