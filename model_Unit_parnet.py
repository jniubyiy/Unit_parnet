# model_Unit_parnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import config_training_tag_unit as cfg

def make_conv2d_block(in_ch, hidden, out_ch, num_layers, dropout):
    layers = []
    current_in = in_ch
    for i in range(num_layers):
        current_out = hidden if i < num_layers - 1 else out_ch
        layers.append(nn.Conv2d(current_in, current_out, kernel_size=3, padding=1))
        if i < num_layers - 1:
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
        current_in = current_out
    return nn.Sequential(*layers)

def make_conv1d_block(in_ch, hidden, out_ch, num_layers, dropout):
    layers = []
    current_in = in_ch
    for i in range(num_layers):
        current_out = hidden if i < num_layers - 1 else out_ch
        layers.append(nn.Conv1d(current_in, current_out, kernel_size=1))
        if i < num_layers - 1:
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        current_in = current_out
    return nn.Sequential(*layers)

class UnifiedUnitParnet(nn.Module):
    """
    Единая модель: на основе текста находит «что нельзя удалять»,
    на основе шума — «что нужно удалять», и восстанавливает чистый парнет.
    """
    def __init__(self,
                 in_channels=None, out_channels=None,
                 txt_feat_hidden=None, txt_feat_layers=None, txt_feat_dropout=None,
                 txt_encoder_hidden=None, txt_encoder_dropout=None,
                 attn_dropout=None,
                 noise_enc_hidden=None, noise_enc_layers=None, noise_enc_dropout=None,
                 fusion_hidden=None, fusion_layers=None, fusion_dropout=None,
                 residual_hidden=None, residual_layers=None, residual_dropout=None):
        if in_channels is None:        in_channels = cfg.UNIT_IN_CHANNELS
        if out_channels is None:       out_channels = cfg.UNIT_OUT_CHANNELS
        if txt_feat_hidden is None:    txt_feat_hidden = cfg.UNIT_TXT_FEAT_HIDDEN
        if txt_feat_layers is None:    txt_feat_layers = cfg.UNIT_TXT_FEAT_LAYERS
        if txt_feat_dropout is None:   txt_feat_dropout = cfg.UNIT_TXT_FEAT_DROPOUT
        if txt_encoder_hidden is None: txt_encoder_hidden = cfg.UNIT_TXT_ENCODER_HIDDEN
        if txt_encoder_dropout is None:txt_encoder_dropout = cfg.UNIT_TXT_ENCODER_DROPOUT
        if attn_dropout is None:       attn_dropout = cfg.UNIT_ATTN_DROPOUT
        if noise_enc_hidden is None:   noise_enc_hidden = cfg.UNIT_NOISE_ENC_HIDDEN
        if noise_enc_layers is None:   noise_enc_layers = cfg.UNIT_NOISE_ENC_LAYERS
        if noise_enc_dropout is None:  noise_enc_dropout = cfg.UNIT_NOISE_ENC_DROPOUT
        if fusion_hidden is None:      fusion_hidden = cfg.UNIT_FUSION_HIDDEN
        if fusion_layers is None:      fusion_layers = cfg.UNIT_FUSION_LAYERS
        if fusion_dropout is None:     fusion_dropout = cfg.UNIT_FUSION_DROPOUT
        if residual_hidden is None:    residual_hidden = cfg.UNIT_RESIDUAL_HIDDEN
        if residual_layers is None:    residual_layers = cfg.UNIT_RESIDUAL_LAYERS
        if residual_dropout is None:   residual_dropout = cfg.UNIT_RESIDUAL_DROPOUT

        super().__init__()

        # Текстовая ветвь
        self.txt_feat_conv = make_conv2d_block(
            in_channels, txt_feat_hidden, txt_feat_hidden,
            txt_feat_layers, txt_feat_dropout
        )
        self.txt_encoder = nn.Sequential(
            make_conv1d_block(32, txt_encoder_hidden, txt_encoder_hidden, 2, txt_encoder_dropout),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.txt_query = nn.Linear(txt_encoder_hidden, txt_feat_hidden)
        self.txt_query_drop = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()
        self.txt_key = nn.Conv2d(txt_feat_hidden, txt_feat_hidden, kernel_size=1)

        # Шумовая ветвь
        self.noise_encoder = make_conv2d_block(
            2 * in_channels, noise_enc_hidden, noise_enc_hidden,
            noise_enc_layers, noise_enc_dropout
        )

        # Объединение и предсказание остаточного шума
        fusion_in = txt_feat_hidden + noise_enc_hidden
        self.fusion = make_conv2d_block(
            fusion_in, fusion_hidden, fusion_hidden,
            fusion_layers, fusion_dropout
        )
        self.residual = make_conv2d_block(
            fusion_hidden, residual_hidden, in_channels,
            residual_layers, residual_dropout
        )

    def forward(self, parnet_noisy, tag_parnets, noise):
        B, C, H, W = parnet_noisy.shape

        # Текстовая ветвь
        img_feat = self.txt_feat_conv(parnet_noisy)
        seq = tag_parnets.permute(0, 2, 1)
        txt_vec = self.txt_encoder(seq)
        query = self.txt_query(txt_vec)
        query = self.txt_query_drop(query)
        query = query.unsqueeze(-1).unsqueeze(-1)
        keys = self.txt_key(img_feat)
        attn = torch.sum(query * keys, dim=1, keepdim=True)
        attn = torch.sigmoid(attn)
        save_feat = img_feat * attn + img_feat

        # Шумовая ветвь
        noise_input = torch.cat([parnet_noisy, noise], dim=1)
        noise_feat = self.noise_encoder(noise_input)

        # Объединение и предсказание остаточного шума
        combined = torch.cat([save_feat, noise_feat], dim=1)
        fused = self.fusion(combined)
        noise_residual = self.residual(fused)

        clean_parnet = parnet_noisy - noise_residual
        return clean_parnet