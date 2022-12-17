from plotting import make_grid_hyp, true_5gaussians_probs, plot_poincare_density
import numpy as np
import torch
from lorentz import Lorentz
import matplotlib.pyplot as plt


def detG(xyz):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    return ((y / x) ** 2 + 1) * ((z / x) ** 2 + 1) - (y * z / x ** 2) ** 2


def detG_Lorentz(xyz):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    return (-((y / x) ** 2) + 1) * (-((z / x) ** 2) + 1) - (y * z / x ** 2) ** 2


def log_standard_normal(x):
    return -0.5 * x ** 2 - np.log(2 * np.pi) / 2


Log2PI = float(np.log(2 * np.pi))


def log_normal(x, mean, log_var, eps=0.00001):
    z = -0.5 * Log2PI
    return -((x - mean) ** 2) / (2.0 * torch.exp(log_var) + eps) - log_var / 2.0 + z


def log_projected_normal(xyz, mean=torch.zeros(1), log_var=torch.zeros(1)):
    yz = xyz[:, 1:]
    return log_normal(yz, mean, log_var).sum(-1) - 0.5 * torch.log(detG(xyz) + 1e-8)


res_npts = 200
on_mani, xy, log_detjac, twodim = make_grid_hyp(res_npts)
probs = torch.exp(
    log_projected_normal(on_mani, log_var=torch.zeros(1) + 0)
    + torch.log((detG(on_mani) / detG_Lorentz(on_mani))) / 2
)
plot_poincare_density(xy, probs, res_npts)


######

# check if lorentz density integrates to 1 after being converted to euclidean density
def phi(xyz):
    return xyz[:, 1:]


def invphi(yz):
    return torch.cat(
        [(yz[:, :1] ** 2 + yz[:, 1:] ** 2 + 1) ** 0.5, yz[:, :1], yz[:, 1:]], 1
    )


M = Lorentz()

n = 1000
bound = 50

x = y = torch.linspace(-bound, bound, n)
XX, YY = torch.meshgrid(x, y)
yz = torch.cat([XX[:, :, None], YY[:, :, None]], 2).view(-1, 2)
xyz = invphi(yz)

twodim = M.squeeze_tangent(M.log0(xyz))

prob = true_5gaussians_probs(xyz, twodim) * detG_Lorentz(xyz) ** 0.5

plt.figure(figsize=(5, 5))
plt.imshow((prob).view(n, n))
print(prob.sum() * (bound * 2 / n) ** 2)
