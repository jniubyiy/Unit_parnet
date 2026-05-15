# model_Unit_parnet_2.py
import torch
import torch.nn as nn


class UnitParnet2(nn.Module):
    """
    Модель, объединяющая два независимых парнета одинаковой формы.

    Входы:
        parnet_a: [B, 3, W, H]  – первый парнет
        parnet_b: [B, 3, W, H]  – второй парнет
    Выход:
        [B, 3, W, H] – результирующий парнет
    """
    def __init__(self, hidden_dim=32):
        super().__init__()
        # Обработка конкатенированных входов
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 3, kernel_size=3, padding=1)
        )

    def forward(self, parnet_a, parnet_b):
        x = torch.cat([parnet_a, parnet_b], dim=1)   # [B, 6, W, H]
        x = self.conv1(x)
        x = self.conv2(x)
        return x                                     # [B, 3, W, H]


# Небольшой тест при прямом запуске
if __name__ == "__main__":
    model = UnitParnet2()
    a = torch.randn(2, 3, 64, 64)
    b = torch.randn(2, 3, 64, 64)
    out = model(a, b)
    print("UnitParnet2")
    print(" input a:", a.shape)
    print(" input b:", b.shape)
    print(" output:", out.shape)