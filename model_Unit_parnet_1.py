# model_Unit_parnet_1.py
import torch
import torch.nn as nn


class UnitParnet1(nn.Module):
    """
    Модель для совместной обработки основного парнета и временной последовательности.
    Добавлен временной шаг t.

    Входы:
        parnet_main:  [B, 3, W, H]  – основной парнет (3 канала)
        parnet_seq:   [B, H, 32]    – последовательность из 32 одноканальных
                                      парнетов длины H
        t:            [B] или [B,1]  – номер временного шага (целое)
    Выход:
        [B, 3, W, H] – модифицированный парнет
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Входной слой: принимает 3 (парнет) + 1 (t) каналов
        self.img_conv = nn.Sequential(
            nn.Conv2d(4, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        # Кодировщик последовательности
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(32, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()          # [B, hidden_dim]
        )
        # FiLM
        self.film_scale = nn.Linear(hidden_dim, hidden_dim)
        self.film_shift = nn.Linear(hidden_dim, hidden_dim)
        # Выходной блок
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 3, kernel_size=3, padding=1)
        )

    def forward(self, parnet_main, parnet_seq, t):
        B, _, W, H = parnet_main.shape
        # Преобразуем t в канал [B,1,W,H]
        if t.dim() == 1:
            t = t.unsqueeze(1)          # [B,1]
        t_map = t.float().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, W, H)  # [B,1,W,H]
        # Конкатенация с основным парнетом
        x = torch.cat([parnet_main, t_map], dim=1)   # [B,4,W,H]
        img_feat = self.img_conv(x)                  # [B,hidden,W,H]

        # Обработка последовательности
        seq = parnet_seq.permute(0, 2, 1)            # [B,32,H]
        temporal_feat = self.temporal_encoder(seq)   # [B,hidden]

        # FiLM
        scale = self.film_scale(temporal_feat).view(B, -1, 1, 1)
        shift = self.film_shift(temporal_feat).view(B, -1, 1, 1)
        modulated = img_feat * scale + shift

        return self.out_conv(modulated)              # [B,3,W,H]