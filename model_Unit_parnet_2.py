# model_Unit_parnet_2.py
import torch
import torch.nn as nn
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

class UnitParnet2(nn.Module):
    """
    Удаляет шум из парнета, используя информацию об исходном шуме.
    Предсказывает остаточную шумовую компоненту и вычитает её.
    """
    def __init__(self, in_channels=None, out_channels=None,
                 encoder_hidden=None, encoder_layers=None, encoder_dropout=None,
                 residual_hidden=None, residual_layers=None, residual_dropout=None):
        if in_channels is None:        in_channels = cfg.UNIT2_IN_CHANNELS
        if out_channels is None:       out_channels = cfg.UNIT2_OUT_CHANNELS
        if encoder_hidden is None:     encoder_hidden = cfg.UNIT2_ENCODER_HIDDEN
        if encoder_layers is None:     encoder_layers = cfg.UNIT2_ENCODER_LAYERS
        if encoder_dropout is None:    encoder_dropout = cfg.UNIT2_ENCODER_DROPOUT
        if residual_hidden is None:    residual_hidden = cfg.UNIT2_RESIDUAL_HIDDEN
        if residual_layers is None:    residual_layers = cfg.UNIT2_RESIDUAL_LAYERS
        if residual_dropout is None:   residual_dropout = cfg.UNIT2_RESIDUAL_DROPOUT

        super().__init__()
        in_features = 2 * in_channels

        self.encoder = make_conv2d_block(in_features, encoder_hidden, encoder_hidden,
                                         encoder_layers, encoder_dropout)

        self.residual = make_conv2d_block(encoder_hidden, residual_hidden, in_channels,
                                          residual_layers, residual_dropout)

    def forward(self, parnet_a, parnet_b):
        x = torch.cat([parnet_a, parnet_b], dim=1)
        feat = self.encoder(x)
        noise_residual = self.residual(feat)
        out = parnet_a - noise_residual
        return out