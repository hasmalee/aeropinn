import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Simple PINN MLP: (inputs) -> (u,v,p,nu_tilde)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 4,
        hidden_dim: int = 128,
        num_hidden_layers: int = 8,
        activation: str = "tanh",
    ):
        super().__init__()

        if activation.lower() == "tanh":
            act = nn.Tanh
        elif activation.lower() == "relu":
            act = nn.ReLU
        elif activation.lower() == "silu":
            act = nn.SiLU
        else:
            raise ValueError("activation must be 'tanh', 'relu', or 'silu'")

        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(act())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act())

        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

        # Xavier init helps PINNs
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def split_outputs(y: torch.Tensor):
    """
    y shape: (N,4) -> u,v,p,nu_tilde each shape (N,1)
    """
    u = y[:, 0:1]
    v = y[:, 1:2]
    p = y[:, 2:3]
    nu_tilde = y[:, 3:4]
    return u, v, p, nu_tilde

