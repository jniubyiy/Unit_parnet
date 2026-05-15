# model_Unit_parnet_1.py
import torch
import torch.nn as nn


class UnitParnet1(nn.Module):
    """
    Модель для совместной обработки основного парнета и временной последовательности.

    Входы:
        parnet_main:  [B, 3, W, H]  – основной парнет (3 канала)
        parnet_seq:   [B, H, 32]    – последовательность из 32 одноканальных
                                      парнетов длины H
    Выход:
        [B, 3, W, H] – модифицированный парнет
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Кодировщик последовательности (сжатие в глобальный вектор)
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(32, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()          # [B, hidden_dim]
        )
        # Извлечение признаков из основного парнета
        self.img_conv = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        # Параметры FiLM для модуляции признаков парнета
        self.film_scale = nn.Linear(hidden_dim, hidden_dim)
        self.film_shift = nn.Linear(hidden_dim, hidden_dim)
        # Выходной блок, возвращающий 3 канала
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 3, kernel_size=3, padding=1)
        )

    def forward(self, parnet_main, parnet_seq):
        B, _, W, H = parnet_main.shape
        # 1. Обработка последовательности: [B, 32, H] -> глобальный контекст
        seq = parnet_seq.permute(0, 2, 1)                     # [B, 32, H]
        temporal_feat = self.temporal_encoder(seq)            # [B, hidden_dim]

        # 2. Признаки основного парнета
        img_feat = self.img_conv(parnet_main)                 # [B, hidden_dim, W, H]

        # 3. FiLM-модуляция
        scale = self.film_scale(temporal_feat).view(B, -1, 1, 1)
        shift = self.film_shift(temporal_feat).view(B, -1, 1, 1)
        modulated = img_feat * scale + shift

        # 4. Получение итогового парнета
        return self.out_conv(modulated)                       # [B, 3, W, H]


# Небольшой тест при прямом запуске
if __name__ == "__main__":
    model = UnitParnet1()
    B, W, H = 2, 64, 64
    main = torch.randn(B, 3, W, H)
    seq = torch.randn(B, H, 32)
    out = model(main, seq)
    print("UnitParnet1")
    print(" main input:", main.shape)
    print(" seq input:", seq.shape)
    print(" output:", out.shape)