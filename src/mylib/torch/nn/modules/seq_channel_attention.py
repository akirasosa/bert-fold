import torch

import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max


# %%
# noinspection PyAbstractClass
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16, activation=F.relu):
        super().__init__()
        self.fc_0 = nn.Linear(channel, channel // reduction, bias=False)
        self.fc_1 = nn.Linear(channel // reduction, channel, bias=False)
        self.activation = activation

    def forward(self, x, attention_mask):
        mask = attention_mask.bool()
        batch = torch.arange(len(mask), device=x.device) \
            .repeat_interleave(mask.sum(dim=1))

        x_mean = scatter_mean(x[mask], batch, dim=0)
        x_max = scatter_max(x[mask], batch, dim=0)[0]

        a0 = self.fc_0(x_mean)
        a0 = self.activation(a0)
        a0 = self.fc_1(a0)

        a1 = self.fc_0(x_max)
        a1 = self.activation(a1)
        a1 = self.fc_1(a1)

        a = torch.sigmoid(a0 + a1)

        return x * a[:, None, :]


# %%
if __name__ == '__main__':
    # %%
    x = torch.tensor([
        [
            [0.1, 0.2, 0.3],
            [0.3, 0.4, 0.6],
        ],
        [
            [0.1, 0.2, 0.3],
            [0.3, 0.4, 0.6],
        ],
        [
            [1.1, 1.2, 1.3],
            [0.3, 0.4, 0.6],
        ],
        [
            [1.1, 1.2, 1.3],
            [1.3, 1.4, 1.6],
        ],
    ])
    attention_mask = torch.tensor([
        [
            1,
            0,
        ],
        [
            1,
            1,
        ],
        [
            1,
            0,
        ],
        [
            1,
            1,
        ],
    ])

    out = ChannelAttention(3, reduction=3).forward(x, attention_mask)
    print(x.shape)
    print(out.shape)
