import torch
import torch.nn as nn


SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class FANLayer(nn.Module):
    def __init__(
        self, input_dim, output_dim, p_ratio=0.25, activation="gelu", use_p_bias=True
    ):
        super(FANLayer, self).__init__()
        assert 0 < p_ratio < 0.5, "p_ratio must be between 0 and 0.5"
        self.p_ratio = p_ratio
        p_output_dim = int(output_dim * self.p_ratio)
        g_output_dim = output_dim - p_output_dim * 2
        self.input_linear_p = nn.Linear(input_dim, p_output_dim, bias=use_p_bias)
        self.input_linear_g = nn.Linear(input_dim, g_output_dim)
        self.activation = (
            getattr(nn.functional, activation)
            if isinstance(activation, str)
            else activation
        )

    def forward(self, src):
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)
        output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        return output


class FANForecastingModel(nn.Module):
    def __init__(self, input_dim, output_dim, p_ratio=0.25, fan_units=64):
        super(FANForecastingModel, self).__init__()
        self.fan_layer = FANLayer(input_dim, fan_units, p_ratio)
        self.output_layer = nn.Linear(fan_units, output_dim)

    def forward(self, x):
        x = self.fan_layer(x)
        x = self.output_layer(x)
        return x
