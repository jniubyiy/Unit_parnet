# model_parnet_tag.py
import torch
import torch.nn as nn


class ParNetTag(nn.Module):
    """
    Модель, преобразующая последовательность ровно из 128 байтов
    в одноканальный парнет размерности 128.

    Вход:  тензор [B, 128], где значения – байты (0–255), желательно в float.
    Выход: тензор [B, 128] – одноканальный парнет, непрерывные значения (без гарантии диапазона).
    """
    def __init__(self, hidden_dim=256, num_layers=3, activation='gelu'):
        super().__init__()
        layers = []
        in_dim = 128
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 128
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(out_dim))
                if activation == 'gelu':
                    layers.append(nn.GELU())
                elif activation == 'relu':
                    layers.append(nn.ReLU())
                else:
                    raise ValueError(f"Unsupported activation: {activation}")
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, byte_sequence: torch.Tensor) -> torch.Tensor:
        """
        byte_sequence: [B, 128], float (рекомендуется нормализация 0..1 или оставить 0..255)
        возвращает [B, 128]
        """
        # Приводим к float, если на вход подали целые
        x = byte_sequence.float()
        return self.mlp(x)


# Опционально: небольшой тест
if __name__ == "__main__":
    model = ParNetTag()
    dummy = torch.randint(0, 256, (2, 128)).float()  # [2, 128]
    out = model(dummy)
    print("Input shape:", dummy.shape)
    print("Output shape:", out.shape)