import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration



class ImageTextCrossAttention(nn.Module):
    """
    
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, img_emb, encoder_outputs, encoder_attention_mask=None):
        """
        img_emb: [B, 1, D]
        encoder_outputs: BaseModelOutput (must use .last_hidden_state)
        encoder_attention_mask: [B, T] → True=attend, False=mask
        """
        # Extract hidden states from BaseModelOutput
        encoder_hidden = encoder_outputs.last_hidden_state  # [B, T, D]

        attn_output, attn_weights = self.cross_attn(
            query=img_emb,               # [B, 1, D]
            key=encoder_hidden,          # [B, T, D]
            value=encoder_hidden,        # [B, T, D]
            key_padding_mask=~encoder_attention_mask.bool() if encoder_attention_mask is not None else None
        )

        out = self.norm(img_emb + self.dropout(attn_output))
        return out, attn_weights


class CrossModalBART(nn.Module):
    def __init__(self, bart_name="facebook/bart-base"):
        super().__init__()
        self.tokenizer = BartTokenizer.from_pretrained(bart_name)
        self.bart = BartForConditionalGeneration.from_pretrained(bart_name)

        self.image_proj = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, self.bart.config.d_model)
        )
        self.cross_attn = ImageTextCrossAttention(
            embed_dim=self.bart.config.d_model,
            num_heads=8,  # BART base는 8 heads
            dropout=0.1
        )

    def forward(self, image, text_input_ids, text_attention_mask, labels):
        B = image.size(0)

        img_emb = self.image_proj(image).unsqueeze(1)  # [B, 1, D]
        text_emb = self.bart.model.encoder.embed_tokens(text_input_ids)

        encoder_inputs_embeds = torch.cat([img_emb, text_emb], dim=1)
        prefix_mask = torch.ones(B, 1).to(text_attention_mask.device)
        encoder_attention_mask = torch.cat([prefix_mask, text_attention_mask], dim=1)

        encoder_outputs = self.bart.model.encoder(
            inputs_embeds=encoder_inputs_embeds,
            attention_mask=encoder_attention_mask
        )

        
        cross_attended_img, attn_weights = self.cross_attn(
            img_emb,
            encoder_outputs,
            encoder_attention_mask
        )

        
        fused_encoder_hidden = torch.cat([cross_attended_img, encoder_outputs.last_hidden_state[:, 1:, :]], dim=1)

        decoder_input_ids = labels[:, :-1]
        decoder_target = labels[:, 1:]
        decoder_padding_mask = (decoder_input_ids != self.tokenizer.pad_token_id)

        decoder_outputs = self.bart.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=fused_encoder_hidden,
            # encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=encoder_attention_mask,
            attention_mask=decoder_padding_mask
        )

        lm_logits = self.bart.lm_head(decoder_outputs.last_hidden_state) + self.bart.final_logits_bias
        return lm_logits, decoder_target


    def encode_image(self, images):
        """
        
        """
        embedding = self.image_proj(images)  # [B, D]
        embedding = embedding.unsqueeze(1)  # [B, 1, D] => Encoder expects sequence
        return embedding



    def generate(self, image, text_input_ids, text_attention_mask, max_len=50):
        B = image.size(0)
        img_emb = self.image_proj(image).unsqueeze(1)
        text_emb = self.bart.model.encoder.embed_tokens(text_input_ids)

        encoder_inputs_embeds = torch.cat([img_emb, text_emb], dim=1)
        prefix_mask = torch.ones(B, 1).to(text_attention_mask.device)
        encoder_attention_mask = torch.cat([prefix_mask, text_attention_mask], dim=1)

        encoder_outputs = self.bart.model.encoder(
            inputs_embeds=encoder_inputs_embeds,
            attention_mask=encoder_attention_mask
        )

        generated_ids = self.bart.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            max_length=max_len,
            num_beams=4
        )
        return generated_ids
