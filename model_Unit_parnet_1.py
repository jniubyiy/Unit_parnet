# model_Unit_parnet_1.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import config_training_tag_unit as cfg

def make_conv2d_block(in_ch, hidden, out_ch, num_layers, dropout):
    """Создаёт последовательность свёрток: num_layers свёрток с GELU и dropout."""
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
    """Создаёт последовательность 1D свёрток."""
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

class UnitParnet1(nn.Module):
    """
    Ищет в парнете области, соответствующие текстовому описанию, и выдаёт парнет
    с усиленными признаками найденных областей.
    """
    def __init__(self, in_channels=None, out_channels=None,
                 img_conv_hidden=None, img_conv_layers=None, img_conv_dropout=None,
                 text_hidden=None, text_dropout=None,
                 attn_dropout=None,
                 out_conv_hidden=None, out_conv_layers=None, out_conv_dropout=None):
        # Чтение параметров из конфига, если не переданы явно
        if in_channels is None:        in_channels = cfg.UNIT1_IN_CHANNELS
        if out_channels is None:       out_channels = cfg.UNIT1_OUT_CHANNELS
        if img_conv_hidden is None:    img_conv_hidden = cfg.UNIT1_IMG_CONV_HIDDEN
        if img_conv_layers is None:    img_conv_layers = cfg.UNIT1_IMG_CONV_LAYERS
        if img_conv_dropout is None:   img_conv_dropout = cfg.UNIT1_IMG_CONV_DROPOUT
        if text_hidden is None:        text_hidden = cfg.UNIT1_TEXT_HIDDEN
        if text_dropout is None:       text_dropout = cfg.UNIT1_TEXT_DROPOUT
        if attn_dropout is None:       attn_dropout = cfg.UNIT1_ATTENTION_DROPOUT
        if out_conv_hidden is None:    out_conv_hidden = cfg.UNIT1_OUT_CONV_HIDDEN
        if out_conv_layers is None:    out_conv_layers = cfg.UNIT1_OUT_CONV_LAYERS
        if out_conv_dropout is None:   out_conv_dropout = cfg.UNIT1_OUT_CONV_DROPOUT

        super().__init__()

        # Извлечение признаков изображения
        self.img_conv = make_conv2d_block(in_channels, img_conv_hidden, img_conv_hidden,
                                          img_conv_layers, img_conv_dropout)

        # Кодировщик текста (1D свёртки + avg pool)
        self.text_encoder = nn.Sequential(
            make_conv1d_block(32, text_hidden, text_hidden, 2, text_dropout),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # Проекции для внимания
        self.query_proj = nn.Linear(text_hidden, img_conv_hidden)
        if attn_dropout > 0:
            self.query_dropout = nn.Dropout(attn_dropout)
        else:
            self.query_dropout = nn.Identity()
        self.key_conv = nn.Conv2d(img_conv_hidden, img_conv_hidden, kernel_size=1)

        # Выходной блок
        self.out_conv = make_conv2d_block(img_conv_hidden, out_conv_hidden, out_channels,
                                          out_conv_layers, out_conv_dropout)

    def forward(self, parnet_main, parnet_seq):
        B, _, H, W = parnet_main.shape

        # Признаки изображения
        img_feat = self.img_conv(parnet_main)                     # [B, HID, H, W]

        # Текстовые признаки
        seq = parnet_seq.permute(0, 2, 1)                        # [B, 128, 32]
        text_feat = self.text_encoder(seq)                       # [B, text_hidden]

        # Пространственное внимание
        query = self.query_proj(text_feat)                       # [B, img_conv_hidden]
        query = self.query_dropout(query)
        query = query.unsqueeze(-1).unsqueeze(-1)                # [B, img_conv_hidden, 1, 1]
        keys = self.key_conv(img_feat)                           # [B, img_conv_hidden, H, W]

        attn_map = torch.sum(query * keys, dim=1, keepdim=True)  # [B, 1, H, W]
        attn_map = torch.sigmoid(attn_map)

        # Модуляция признаков вниманием + остаточная связь
        modulated = img_feat * attn_map + img_feat

        # Выходной парнет
        out = self.out_conv(modulated)
        return out