import torch
import torch.nn as nn


_scaling_min = 0.001


def set_activation(activation):
    if activation["type"] == "relu":
        return F.relu
    elif activation["type"] == "elu":
        return nn.ELU()
    elif activation["type"] == "srelu":
        return nn.Softplus(beta=activation["BETA"])
    elif activation["type"] == "tanh":
        return nn.Tanh()
    elif activation["type"] == "sin":
        return torch.sin
    elif activation["type"] == "squared_relu":
        return lambda x: F.relu(x).pow(2)
    elif activation["type"] == "swish":
        return Swish()
    else:
        raise ValueError("no such activation %s" % activation["type"])


# noinspection PyUnusedLocal
class ActNorm(torch.nn.Module):
    """ ActNorm layer with data-dependant init."""

    def __init__(self, num_features, logscale_factor=1.0, scale=1.0, learn_scale=True):
        super(ActNorm, self).__init__()
        self.initialized = False
        self.num_features = num_features

        self.register_parameter(
            "b", nn.Parameter(torch.zeros(1, num_features, 1), requires_grad=True)
        )
        self.learn_scale = learn_scale
        if learn_scale:
            self.logscale_factor = logscale_factor
            self.scale = scale
            self.register_parameter(
                "logs",
                nn.Parameter(torch.zeros(1, num_features, 1), requires_grad=True),
            )

    def forward(self, x):
        input_shape = x.size()
        x = x.view(input_shape[0], input_shape[1], -1)

        if not self.initialized:
            self.initialized = True

            # noinspection PyShadowingNames
            def unsqueeze(x):
                return x.unsqueeze(0).unsqueeze(-1).detach()

            # Compute the mean and variance
            sum_size = x.size(0) * x.size(-1)
            b = -torch.sum(x, dim=(0, -1)) / sum_size
            self.b.data.copy_(unsqueeze(b).data)

            if self.learn_scale:
                var = unsqueeze(
                    torch.sum((x + unsqueeze(b)) ** 2, dim=(0, -1)) / sum_size
                )
                logs = (
                    torch.log(self.scale / (torch.sqrt(var) + 1e-6))
                    / self.logscale_factor
                )
                self.logs.data.copy_(logs.data)

        b = self.b
        output = x + b

        if self.learn_scale:
            logs = self.logs * self.logscale_factor
            scale = torch.exp(logs) + _scaling_min
            output = output * scale

            return output.view(input_shape)
        else:
            return output.view(input_shape)


class Swish(nn.Module):
    @staticmethod
    def forward(x):
        return torch.sigmoid(x) * x


class ConcatMLP(nn.Module):
    def __init__(
        self,
        dimx: int = 3,
        dimt: int = 1,
        emb_size: int = 256,
        num_hidden_layers: int = 2,
        actnorm_first=False,
    ):
        super(ConcatMLP, self).__init__()
        layers = [nn.Linear(dimx + dimt, emb_size), Swish()]
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(emb_size, emb_size), Swish()])
        layers.extend([nn.Linear(emb_size, dimx)])

        if actnorm_first:
            layers = [ActNorm(dimx + dimt),] + layers

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x, t):
        if isinstance(t, float) or (isinstance(t, torch.Tensor) and t.dim() == 0):
            t = torch.ones(x.size(0), 1).to(x) * t
        print(x.shape, t.shape)
        return self.net(torch.cat([x, t], 1))


class ConcatMatMLP(nn.Module):
    def __init__(
        self,
        dimx1: int = 3,
        dimx2: int = 3,
        dimt: int = 1,
        emb_size: int = 256,
        num_hidden_layers: int = 2,
        actnorm_first=False,
    ):
        """MLP for matrix-valued data"""
        super(ConcatMatMLP, self).__init__()
        self.dimx1 = dimx1
        self.dimx2 = dimx2
        dimx = dimx1 * dimx2
        layers = [nn.Linear(dimx + dimt, emb_size), Swish()]
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(emb_size, emb_size), Swish()])
        layers.extend([nn.Linear(emb_size, dimx)])

        if actnorm_first:
            layers = [ActNorm(dimx + dimt),] + layers

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x, t):
        if isinstance(t, float) or (isinstance(t, torch.Tensor) and t.dim() == 0):
            t = torch.ones(x.size(0), 1).to(x) * t

            # print(x.shape, t.shape)
        shape = x.size()
        x_flattened = x.view(shape[0], -1)
        # print(x_flattened.shape, t.shape)
        return self.net(torch.cat([x_flattened, t], 1)).view(
            shape[0], self.dimx1, self.dimx2
        )


class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    @staticmethod
    def forward(x):
        return torch.sin(x)


class ConcatSinMLP(nn.Module):
    def __init__(
        self,
        dimx: int = 3,
        dimt: int = 1,
        emb_size: int = 256,
        num_hidden_layers: int = 2,
    ):
        super(ConcatSinMLP, self).__init__()
        layers = [nn.Linear(dimx + dimt, emb_size), Sin()]
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(emb_size, emb_size), Sin()])
        layers.extend([nn.Linear(emb_size, dimx)])
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x, t):
        if isinstance(t, float) or (isinstance(t, torch.Tensor) and t.dim() == 0):
            t = torch.ones(x.size(0), 1).to(x) * t
        return self.net(torch.cat([x, t], 1))
