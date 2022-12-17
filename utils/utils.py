import numpy as np
import torch
import torch.nn as nn
import math
import manifold
import lorentz
import utils
import random
import os

from manifold import Circle
from einops import rearrange, repeat
import ipdb
from torchsde._core import adaptive_stepping
import warnings
from data import orthogonal_group

TWOPI = 2 * math.pi
PI = math.pi
EPS = 1e-9
LOG_SPHERICAL_UNIFORM = -torch.log(torch.tensor(4 * math.pi))
LOG_CIRCLE_UNIFORM = -torch.log(torch.tensor(2 * math.pi))
Log2PI = float(np.log(2 * np.pi))
LOG_SO3_UNIFORM = -torch.log(torch.tensor(8 * math.pi ** 2))
M = lorentz.Lorentz()


class ProjectedNormal(torch.distributions.Distribution):
    support = torch.distributions.constraints.real
    has_rsample = True

    def __init__(self, device, manifold, loc, scale, *args, **kwargs):
        self.manifold = manifold
        self.loc = loc
        self.scale = scale
        self.dev = device
        super().__init__(self.loc, self.scale, *args, **kwargs)

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    def log_standard_normal(self, x):
        return -0.5 * x ** 2 - np.log(2 * np.pi) / 2

    def log_normal(self, x, mean, log_var, eps=0.00001):
        z = -0.5 * Log2PI
        return -((x - mean) ** 2) / (2.0 * torch.exp(log_var) + eps) - log_var / 2.0 + z

    def log_projected_normal(self, xyz, mean=torch.zeros(1), log_var=torch.zeros(1)):
        yz = xyz[:, 1:]
        return self.log_normal(yz, mean, log_var).sum(-1) - 0.5 * torch.log(
            self.manifold.detG(xyz) + 1e-8
        )

    def log_prob(self, xyz, mean=torch.zeros(1), log_var=torch.zeros(1)):
        mean = mean.to(self.dev)
        log_var = log_var.to(self.dev)
        return self.log_projected_normal(xyz, mean, log_var)

    def get_samples(self, num_samples: int):
        return M.invphi(torch.randn(num_samples, 2))


def stratified_uniform(n, r, device="cpu"):
    """
    generate a batch of uniform([0, r]) samples in a stratefied manner (to reduce variance)
    """
    indices = np.arange(n)
    np.random.shuffle(indices)
    t = (
        torch.rand(n, device=device) / n + torch.Tensor(indices).to(device) / n
    ).unsqueeze(-1) * r
    return t


def sample_rademacher(*shape, device="cpu"):
    return (torch.rand(*shape, device=device) > 0.5).float() * 2 - 1


def sample_2d_uniform_patch(n: int = 1):
    theta = (torch.rand(n) * 70 + 40) / 360 * TWOPI
    phi = (torch.rand(n) * 110 + 50) / 360 * TWOPI
    return torch.stack([theta, phi], dim=1)


def sample_spherical_uniform(n: int = 1):
    return manifold.Sphere.proj2manifold(torch.randn(n, 3))


def sample_spherical_uniform_polar(n: int = 1):
    return manifold.Sphere.phi(sample_spherical_uniform(n))


def sample_so3_uniform(n: int = 1):
    rng = orthogonal_group.random.PRNGKey(np.random.randint(9e16))
    return torch.Tensor(np.asarray(orthogonal_group.sample_haar_son(rng, n, 3)))


# noinspection PyUnusedLocal
def zero_func(*x):
    return 0


# noinspection PyUnusedLocal
def identity(x):
    return x


def elbo(
    y,
    s,
    a_func,
    drift=None,
    proj2manifold=None,
    proj2tangent=None,
    T=None,
    method="closest",
    dim=None,
):
    """

    :param y: data
    :param s: time step
    :param a_func: variational degree of freedom
    :param drift: drift variable
    :param proj2manifold: closest-point projection to manifold (used if method=`closest`)
    :param proj2tangent: tangential projection
    :param T: estimation time interval (terminal time)
    :param method: computation method
    :param dim: dimensionality of the manifold (used if method=`qr`)
    :return: elbo estimate
    """
    if method == "closest":
        py = proj2manifold(y)
        a = a_func(py, s)
        if drift is not None:
            drift = proj2tangent(py, drift)
        else:
            drift = 0

        v0 = proj2tangent(py, a) - drift

        div_v0 = 0
        for i in range(y.size(1)):
            div_v0 += torch.autograd.grad(
                v0[:, i].sum(), y, create_graph=True, retain_graph=True
            )[0][:, i]
    elif method == "qr":
        a = a_func(y, s)
        if drift is not None:
            drift = proj2tangent(y, drift)
        else:
            drift = 0

        v0 = proj2tangent(y, a) - drift

        basis = torch.linalg.qr(
            proj2tangent(y, torch.randn(y.shape + (dim,), device=y.device))
        )[0].detach()

        div_v0 = 0
        for i in range(dim):
            q = basis[:, :, i]
            div_v0 += (
                torch.autograd.grad(
                    (v0 * q).sum(), y, create_graph=True, retain_graph=True
                )[0]
                * q
            ).sum(1)
    elif method == "hutchinson-normal":
        a = a_func(y, s)
        if drift is not None:
            drift = proj2tangent(y, drift)
        else:
            drift = 0
        v0 = proj2tangent(y, a) - drift

        q = proj2tangent(y, torch.randn_like(y))
        div_v0 = (
            torch.autograd.grad(
                (v0 * q).sum(), y, create_graph=True, retain_graph=True
            )[0]
            * q
        ).sum(1)
    elif method == "hutchinson-rademacher":
        a = a_func(y, s)
        if drift is not None:
            drift = proj2tangent(y, drift)
        else:
            drift = 0
        v0 = proj2tangent(y, a) - drift

        q = proj2tangent(y, sample_rademacher(y.shape, device=y.device))
        div_v0 = (
            torch.autograd.grad(
                (v0 * q).sum(), y, create_graph=True, retain_graph=True
            )[0]
            * q
        ).sum(1)

    else:
        raise NotImplementedError

    return (-0.5 * (proj2tangent(y, a) ** 2).sum(dim=1) - div_v0) * T


# TODO: refactor the following by adding a `logp0` argument (function of yT) to elbo
#       also pass `method` as an argument


def elbo_tori(y, s, a_func, proj2manifold, proj2tangent, T):
    tori_intrinsic_dim = y.shape[1] // 2
    return (
        elbo(
            y,
            s,
            a_func,
            drift=None,
            proj2manifold=proj2manifold,
            proj2tangent=proj2tangent,
            T=T,
            dim=tori_intrinsic_dim,
        )
        + LOG_CIRCLE_UNIFORM * tori_intrinsic_dim
    )


def elbo_sphere(y, s, a_func, proj2manifold, proj2tangent, T):
    return (
        elbo(
            y,
            s,
            a_func,
            drift=None,
            proj2manifold=proj2manifold,
            proj2tangent=proj2tangent,
            T=T,
            method="qr",
            dim=2,
        )
        + LOG_SPHERICAL_UNIFORM
    )


def elbo_lorentz(prior, sde, x, y, s, a_func, proj2manifold, proj2tangent):
    y_T = sde.sample(x, sde.T)
    prior_log_prob = prior.log_prob(y_T)
    return (
        elbo(
            y,
            s,
            a_func,
            drift=sde.f(y, s),
            proj2manifold=proj2manifold,
            proj2tangent=proj2tangent,
            T=sde.T,
            method="qr",
            dim=2,
        )
        + prior_log_prob
    )


def elbo_so3(y, s, a_func, proj2manifold, proj2tangent, T):
    return (
        elbo(
            y,
            s,
            a_func,
            proj2manifold=proj2manifold,
            proj2tangent=proj2tangent,
            T=T,
            method="hutchinson-rademacher",
        )
        + LOG_SO3_UNIFORM
    )


def exponent_step(
    y, s, a_func, drift, proj2manifold, proj2tangent, ds, db, method, dim
):
    if method == "closest":
        py = proj2manifold(y)
        a = a_func(py, s)
        if drift is not None:
            drift = proj2tangent(py, drift)
        else:
            drift = 0

        v0 = proj2tangent(py, a) - drift

        div_v0 = 0
        for i in range(y.size(1)):
            div_v0 += torch.autograd.grad(
                v0[:, i].sum(), y, create_graph=False, retain_graph=True
            )[0][:, i]

    elif method == "qr":
        a = a_func(y, s)
        if drift is not None:
            drift = proj2tangent(y, drift)
        else:
            drift = 0

        v0 = proj2tangent(y, a) - drift

        basis = torch.linalg.qr(
            proj2tangent(y, torch.randn(y.shape + (dim,), device=y.device))
        )[0].detach()

        div_v0 = 0
        for i in range(dim):
            q = basis[:, :, i]
            div_v0 += (
                torch.autograd.grad(
                    (v0 * q).sum(), y, create_graph=True, retain_graph=True
                )[0]
                * q
            ).sum(1)

    elif method == "hutchinson-normal":
        a = a_func(y, s)
        if drift is not None:
            drift = proj2tangent(y, drift)
        else:
            drift = 0

        v0 = proj2tangent(y, a) - drift

        q = proj2tangent(y, torch.randn_like(y))
        div_v0 = (
            torch.autograd.grad(
                (v0 * q).sum(), y, create_graph=True, retain_graph=True
            )[0]
            * q
        ).sum(1)

    elif method == "hutchinson-rademacher":
        a = a_func(y, s)
        if drift is not None:
            drift = proj2tangent(y, drift)
        else:
            drift = 0

        v0 = proj2tangent(y, a) - drift

        q = proj2tangent(y, sample_rademacher(y.shape, device=y.device))
        div_v0 = (
            torch.autograd.grad(
                (v0 * q).sum(), y, create_graph=True, retain_graph=True
            )[0]
            * q
        ).sum(1)

    else:
        raise NotImplementedError

    A = -(proj2tangent(y, a) * db).sum(axis=-1)
    B = (-0.5 * (proj2tangent(y, a) ** 2).sum(dim=1) - div_v0) * ds
    return A + B


def compute_prior(y0, manifold_name, prior):
    if manifold_name == "Sphere":
        return prior
    elif manifold_name == "Tori":
        return prior
    elif manifold_name == "SO3":
        return prior
    elif manifold_name == "Hyperboloid":
        prior_log_prob = prior.log_prob(y0)
        return prior_log_prob
    else:
        raise NotImplementedError


def kelbo(
    sde,
    y0,
    a_func,
    proj2manifold,
    proj2tangent,
    T,
    K,
    prior,
    steps: int = 1000,
    method: str = "closest",
    dim: int = 2,
):
    # adding K copies of the data so I can parallelize the sampling
    from sdes import heun_step as step

    y0 = repeat(y0, "b d -> b k d", k=K)
    y0 = rearrange(y0, "b k d -> (b k) d")

    y_next = y0
    s = 0
    ds = T / steps
    expo = torch.zeros(y0.shape[0]).to(y0)
    for _ in range(steps):
        s += ds
        y = y_next.detach().requires_grad_(True)
        y_next, increment = step(
            s, ds, y, sde.f, sde.g_increment, proj2manifold, return_increment=True
        )
        expo_step = exponent_step(
            y=y,
            s=s,
            a_func=a_func,
            drift=sde.f(y, s),
            proj2manifold=proj2manifold,
            proj2tangent=proj2tangent,
            ds=ds,
            db=increment,
            method=method,
            dim=dim,
        )
        expo = expo_step.detach() + expo

    # Add Prior Log Prob
    logp0 = compute_prior(y, sde.mani_name, prior)
    expo = expo + logp0
    # disentangling for each data point
    expo = rearrange(expo, "(b k) -> b k", k=K)
    logk = torch.log(torch.tensor(K)).to(y0)

    return torch.logsumexp(expo, dim=1) - logk


def kelbo_sphere(sde, y0, a_func, proj2manifold, proj2tangent, T, K, steps: int = 1000):
    return kelbo(
        sde,
        y0,
        a_func,
        proj2manifold,
        proj2tangent,
        T,
        K,
        LOG_SPHERICAL_UNIFORM,
        steps=steps,
    )


def kelbo_tori(sde, y0, a_func, proj2manifold, proj2tangent, T, K, steps: int = 1000):
    tori_dim = y0.shape[1] // 2
    prior = tori_dim * LOG_CIRCLE_UNIFORM
    return kelbo(sde, y0, a_func, proj2manifold, proj2tangent, T, K, prior, steps=steps)


def kelbo_lorentz(
    sde, y0, a_func, proj2manifold, proj2tangent, T, K, prior, steps: int = 1000
):
    return kelbo(
        sde,
        y0,
        a_func,
        proj2manifold,
        proj2tangent,
        T,
        K,
        prior,
        steps=steps,
        method="qr",
    )


def adaptivekelbo(
    sde,
    y0,
    a_func,
    proj2manifold,
    proj2tangent,
    T,
    K,
    ds: float,
    prior,
    ds_min=1e-5,
    rtol=1e-3,
    atol=1e-3,
    method="closest",
    dim=2,
):
    # adding K copies of the data so I can parallelize the sampling
    from sdes import heun_step as step

    y0 = repeat(y0, "b d -> b k d", k=K)
    y0 = rearrange(y0, "b k d -> (b k) d")

    y = y0
    s = 0.0
    # expo = torch.zeros(y0.shape[0]).to(y0) + logp0
    expo = torch.zeros(y0.shape[0]).to(y0)
    prev_error_ratio = None
    while s < T:
        y = y.detach().requires_grad_(True)

        db1 = torch.randn_like(y0) * (ds / 2) ** 0.5
        db2 = torch.randn_like(y0) * (ds / 2) ** 0.5
        db = db1 + db2

        expo_step = exponent_step(
            y=y,
            s=s,
            a_func=a_func,
            drift=sde.f(y, s),
            proj2manifold=proj2manifold,
            proj2tangent=proj2tangent,
            ds=ds,
            db=db,
            method=method,
            dim=dim,
        )

        # Doing two half steps
        y_half = step(
            s, ds / 2, y, sde.f, sde.g_increment, proj2manifold, increment=db1
        )
        expo_step1 = exponent_step(
            y=y,
            s=s,
            a_func=a_func,
            drift=sde.f(y_half, s + ds / 2),
            proj2manifold=proj2manifold,
            proj2tangent=proj2tangent,
            ds=ds / 2,
            db=db1,
            method=method,
            dim=dim,
        )
        y_next2 = step(
            s + ds / 2,
            ds / 2,
            y_half,
            sde.f,
            sde.g_increment,
            proj2manifold,
            increment=db2,
        )

        expo_step2 = exponent_step(
            y=y_half,
            s=s + ds / 2,
            a_func=a_func,
            drift=sde.f(y_half, s + ds / 2),
            proj2manifold=proj2manifold,
            proj2tangent=proj2tangent,
            ds=ds / 2,
            db=db2,
            method=method,
            dim=dim,
        )
        expo_step_combined = expo_step1 + expo_step2

        with torch.no_grad():
            error_estimate = adaptive_stepping.compute_error(
                expo_step, expo_step_combined, rtol, atol
            )
            ds, prev_error_ratio = adaptive_stepping.update_step_size(
                error_estimate=error_estimate,
                prev_step_size=ds,
                prev_error_ratio=prev_error_ratio,
            )

        if ds < ds_min:
            warnings.warn(
                "Hitting minimum allowed step size in adaptive time-stepping."
            )
            ds = ds_min
            prev_error_ratio = None

        # Accept step.
        if error_estimate <= 1 or ds <= ds_min:
            s, y = s + ds, y_next2

            expo = expo_step_combined.detach() + expo

    # Add Prior Log Prob
    logp0 = compute_prior(y, sde.mani_name, prior)
    expo = expo + logp0

    # disentangling for each data point
    expo = rearrange(expo, "(b k) -> b k", k=K)
    logk = torch.log(torch.tensor(K)).to(y0)

    return torch.logsumexp(expo, dim=1) - logk


def adaptivekelbo_sphere(
    sde,
    y0,
    a_func,
    proj2manifold,
    proj2tangent,
    T,
    K,
    ds: float,
    ds_min=1e-5,
    rtol=1e-3,
    atol=1e-3,
    method="closest",
    dim=2,
):
    return adaptivekelbo(
        sde,
        y0,
        a_func,
        proj2manifold,
        proj2tangent,
        T,
        K,
        ds,
        LOG_SPHERICAL_UNIFORM,
        ds_min=ds_min,
        rtol=rtol,
        atol=atol,
        method=method,
        dim=dim,
    )


def adaptivekelbo_tori(
    sde,
    y0,
    a_func,
    proj2manifold,
    proj2tangent,
    T,
    K,
    ds: float,
    ds_min=1e-5,
    rtol=1e-3,
    atol=1e-3,
    method="closest",
    dim=2,
):
    tori_dim = y0.shape[1] // 2
    return adaptivekelbo(
        sde,
        y0,
        a_func,
        proj2manifold,
        proj2tangent,
        T,
        K,
        ds,
        tori_dim * LOG_CIRCLE_UNIFORM,
        ds_min=ds_min,
        rtol=rtol,
        atol=atol,
        method=method,
        dim=dim,
    )


def adaptivekelbo_lorentz(
    sde,
    y0,
    a_func,
    proj2manifold,
    proj2tangent,
    T,
    K,
    prior,
    ds: float,
    ds_min=1e-5,
    rtol=1e-3,
    atol=1e-3,
    method="qr",
    dim=2,
):
    return adaptivekelbo(
        sde,
        y0,
        a_func,
        proj2manifold,
        proj2tangent,
        T,
        K,
        ds,
        prior,
        ds_min=ds_min,
        rtol=rtol,
        atol=atol,
        method=method,
        dim=dim,
    )


def adaptivekelbo_so3(
    sde,
    y0,
    a_func,
    proj2manifold,
    proj2tangent,
    T,
    K,
    ds: float,
    ds_min=1e-5,
    rtol=1e-3,
    atol=1e-3,
    method="qr",
    dim=3,
):
    # TODO shape issue
    return adaptivekelbo(
        sde,
        y0,
        a_func,
        proj2manifold,
        proj2tangent,
        T,
        K,
        ds,
        LOG_SO3_UNIFORM,
        ds_min=ds_min,
        rtol=rtol,
        atol=atol,
        method=method,
        dim=dim,
    )


def get_drift(a_func, T, manifold_obj, sde):
    # noinspection PyShadowingNames
    def v0(x, t):
        x = manifold_obj.proj2manifold(x)
        return manifold_obj.proj2tangent(
            x, a_func(x, T - t)
        ) - manifold_obj.proj2tangent(x, sde.f(x, T - t))

    return v0


""" Set Random Seed """


def seed_everything(seed):
    if seed:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tori_theta_to_ambient(theta):
    """
    x: (batch_size, dim)
    returns: (batch_size, dim*2) embed n dimensional tori embedded in a dim*2 dimensional ambient space
    """
    b = theta.shape[0]
    theta = rearrange(theta, "b d -> (b d) 1")
    x = Circle.invphi(theta)
    x = rearrange(x, "(b d) two-> b (d two)", b=b)
    return x


def tori_ambient_to_theta(x):
    """
    x: (batch_size, dim*2)
    returns: (batch_size, dim) theta
    """
    b = x.shape[0]
    d = x.shape[1] // 2
    x = rearrange(x, "b (d two) -> (b d) two", d=d)
    theta = Circle.phi(x)
    theta = rearrange(theta, "(b d)-> b d", b=b)
    return theta


def sample_tori_uniform(tori_dim: int, n: int):
    theta0 = torch.rand(n, tori_dim) * 2 - 1.0
    x = tori_theta_to_ambient(theta0)
    return x


def exact_logp(x_ambient, bm_sde, a_func, prior, steps, method="closest"):
    from sdes import ExactLogPODE

    a_func.eval()
    logp = ExactLogPODE(bm_sde, a_func).exact_logp_via_integration(
        x_ambient, bm_sde.T, prior=prior, steps=steps, method=method
    )
    a_func.train()
    return logp
