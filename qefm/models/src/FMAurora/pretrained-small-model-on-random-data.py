#https://github.com/microsoft/aurora
from datetime import datetime

import torch

from timm.layers import to_2tuple, make_divisible, GroupNorm1, ConvMlp, DropPath
from aurora import AuroraSmall, Batch, Metadata

model = AuroraSmall()

model.load_checkpoint("microsoft/FMAurora", "FMAurora-0.25-small-pretrained.ckpt")

batch = Batch(
    surf_vars={k: torch.randn(1, 2, 17, 32) for k in ("2t", "10u", "10v", "msl")},
    static_vars={k: torch.randn(17, 32) for k in ("lsm", "z", "slt")},
    atmos_vars={k: torch.randn(1, 2, 4, 17, 32) for k in ("z", "u", "v", "t", "q")},
    metadata=Metadata(
        lat=torch.linspace(90, -90, 17),
        lon=torch.linspace(0, 360, 32 + 1)[:-1],
        time=(datetime(2020, 6, 1, 12, 0),),
        atmos_levels=(100, 250, 500, 850),
    ),
)

prediction = model.forward(batch)

print(prediction.surf_vars["2t"])
