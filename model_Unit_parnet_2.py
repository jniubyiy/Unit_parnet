# model_Unit_parnet_2.py
import torch
import torch.nn as nn


class UnitParnet2(nn.Module):
    """
    Модель, объединяющая два независимых парнета одинаковой формы.
    Добавлен временной шаг t.

    Входы:
        parnet_a: [B, 3, W, H]  – первый парнет
        parnet_b: [B, 3, W, H]  – второй парнет
        t:        [B] или [B,1]  – номер временного шага
    Выход:
        [B, 3, W, H] – результирующий парнет
    """
    def __init__(self, hidden_dim=32):
        super().__init__()
        # Вход: 3+3+1 = 7 каналов
        self.conv1 = nn.Sequential(
            nn.Conv2d(7, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 3, kernel_size=3, padding=1)
        )

    def forward(self, parnet_a, parnet_b, t):
        B, _, W, H = parnet_a.shape
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t_map = t.float().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, W, H)  # [B,1,W,H]
        x = torch.cat([parnet_a, parnet_b, t_map], dim=1)  # [B,7,W,H]
        x = self.conv1(x)
        x = self.conv2(x)
        return x