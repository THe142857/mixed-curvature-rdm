import torch
import numpy as np
from einops import rearrange, repeat
from torch import nn

import manifold
from manifold import Tori
from utils.utils import tori_theta_to_ambient, tori_ambient_to_theta
from utils.utils import identity, compute_prior
from lorentz import Lorentz
from torchdiffeq import odeint


# noinspection PyMethodMayBeStatic,PyUnusedLocal
class CircularBrownianMotion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mani_name = "Circle"

    def f(self, y, t):
        return torch.zeros_like(y)

    def g(self, y, t):
        return torch.ones(list(y.shape))

    def sample(self, y0, t):
        """
        sample yt | y0
        """
        y = (y0 + torch.randn_like(y0) * t ** 0.5) % (2 * np.pi)
        return y


class CircularGenerative(torch.nn.Module):
    def __init__(self, sigma, a, T=0.2):
        # TODO: remove T, and integrate `sigma` & `a` into a single drift func
        super().__init__()
        self.T = T
        self.sigma = sigma
        self.a = a
        self.mani_name = "Circle"

    def mu(self, x, t):
        sa = self.sigma(x, self.T - t) * self.a(x, self.T - t)
        return sa

    def sample(self, x0, T, steps):
        """
        sample xt | x0
        """
        dt = T / steps
        x = x0
        for s in range(steps):
            t = s * dt
            noise = torch.randn_like(x0) * dt ** 0.5
            dx = self.mu(x, t) * dt + noise
            x = (x + dx) % (2 * np.pi)
        return x


def midpoint_step(
    t0,
    dt,
    y0,
    f_func,
    g_func,
    projection,
    return_increment: bool = False,
    increment=None,
):
    half_dt = 0.5 * dt
    t_prime = t0 + half_dt
    if increment is None:
        increment = torch.randn_like(y0) * dt ** 0.5

    f = f_func(y0, t0)
    g = g_func(y0, t0, increment)

    y_prime = y0 + half_dt * f + 0.5 * g
    f_prime = f_func(y_prime, t_prime)
    g_prod_prime = g_func(y_prime, t_prime, increment)

    y1 = y0 + dt * f_prime + g_prod_prime

    if return_increment:
        return projection(y1), increment
    else:
        return projection(y1)


def heun_step(
    t0,
    dt,
    y0,
    f_func,
    g_func,
    projection,
    return_increment: bool = False,
    increment=None,
):
    """
    Stratonovich Heun method
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.205.6327&rep=rep1&type=pdf
    """

    if increment is None:
        # print("y0: ", y0)
        # print("y0 type: ", y0.dtype)
        increment = torch.randn_like(y0) * dt ** 0.5
        # increment = torch.randint(y0) * dt ** 0.5

    f0 = f_func(y0, t0)
    g0 = g_func(y0, t0, increment)

    y_prime = y0 + f0 * dt + g0

    f_prime = f_func(y_prime, t0 + dt)
    g_prime = g_func(y_prime, t0 + dt, increment)

    y1 = y0 + 0.5 * (f0 + f_prime) * dt + 0.5 * (g0 + g_prime)

    if return_increment:
        return projection(y1), increment
    else:
        return projection(y1)


@torch.no_grad()
def integration(x0, steps, sde, T, projection=identity, step=heun_step):
    x = x0
    t = 0
    dt = T / steps
    for _ in range(steps):
        t += dt
        x = step(t, dt, x, sde.f, sde.g_increment, projection)
    # print(x)
    return x


# noinspection PyMethodMayBeStatic,PyUnusedLocal
class AmbientSphericalBrownianMotion(torch.nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T
        self.mani_name = "Sphere"
        self.M = manifold.Sphere

    def f(self, x, t):
        return torch.zeros_like(x)

    def g_increment(self, x, t, increment):
        return manifold.Sphere.proj2tangent(x, increment)

    def sample(self, x, t, steps=100):
        return integration(x, steps, self, t, manifold.Sphere.proj2manifold)


# noinspection PyMethodMayBeStatic,PyUnusedLocal
class AmbientSphericalGenerative(torch.nn.Module):
    def __init__(self, drift_func):
        super().__init__()
        self.f = drift_func
        self.mani_name = "Sphere"

    def g_increment(self, x, t, increment):
        return manifold.Sphere.proj2tangent(x, increment)

    def sample(self, x, t, steps=100):
        return integration(x, steps, self, t, manifold.Sphere.proj2manifold)


# noinspection PyMethodMayBeStatic,PyUnusedLocal
class AmbientHyperbolicBrownianMotion(torch.nn.Module):
    def __init__(self, T, drift_coeff):
        super().__init__()
        self.T = T
        self.M = Lorentz()
        self.drift_coeff = drift_coeff
        self.mani_name = "Hyperboloid"

    def f(self, x, t):
        return self.drift_coeff * self.M.proj2tangent(x, -x)

    def g_increment(self, x, t, increment):
        return self.M.proj2tangent(x, increment)

    def sample(self, x, t, steps=100):
        return integration(
            x, steps, self, t, identity
        )  # self.M.proj2manifold) # TODO: why is this buggy


# noinspection PyMethodMayBeStatic,PyUnusedLocal
class AmbientHyperbolicGenerative(torch.nn.Module):
    def __init__(self, drift_func):
        super().__init__()
        self.f = drift_func
        self.M = Lorentz()
        self.mani_name = "Hyperboloid"

    def g_increment(self, x, t, increment):
        return self.M.proj2tangent(x, increment)

    def sample(self, x, t, steps=100):
        return torch.clamp(
            integration(x, steps, self, t, self.M.proj2manifold), -1e10, 1e10
        )  # TODO: do we need clamping?


class EquivalentInferenceSDE(torch.nn.Module):
    def __init__(self, bm_sde, a_func, lambda_constant: float):
        super().__init__()
        self.bm_sde = bm_sde
        self.a_func = a_func
        self.lambda_constant = lambda_constant

    def f(self, y, s):
        A = self.bm_sde.f(y, s)
        B = self.lambda_constant / 2 * self.a_func(y, s)
        return A - B

    def g_increment(self, y, s, increment):
        return torch.sqrt(1 - self.lambda_constant) * self.bm_sde.g_increment(
            y, s, increment
        )

    def sample(self, y, s, steps=100):
        def integration_projection(x):
            if self.bm_sde.mani_name == "Hyperboloid":
                return identity(x)
            else:
                return self.bm_sde.M.proj2manifold(x)

        return integration(y, steps, self, s, integration_projection)


class ExactLogPODE(torch.nn.Module):
    def __init__(self, bm_sde, a_func):
        super().__init__()
        self.bm_sde = bm_sde
        self.a_func = a_func

    def f(self, y, s):
        # print(s)
        A = self.bm_sde.f(y, s)
        # print(s)
        B = 0.5 * self.a_func(y, s)
        return self.bm_sde.M.proj2tangent(y, A - B)

    def estimate_div_v0_qr(self, y, s):
        y.requires_grad_(True)
        manifold_dim = self.bm_sde.M.manifold_dim
        proj2tangent = self.bm_sde.M.proj2tangent

        v0 = -1 * self.f(y, s)

        random_vectors = torch.randn(
            y.shape + (manifold_dim,), device=y.device
        ).detach()

        def proj2tangent_random_vectors(vs):
            shape = vs.shape
            vs = vs.view(shape[0], -1, shape[-1])
            vs = rearrange(vs, "b ad md -> (b md) ad")
            vs = proj2tangent(
                repeat(y.view(shape[0], -1), "b ad -> (b md) ad", md=manifold_dim).view(
                    shape[0] * manifold_dim, *shape[1:-1]
                ),
                vs,
            )
            vs = rearrange(vs, "(b md) ad -> b ad md", md=manifold_dim)
            return vs.view(*shape)

        basis = torch.linalg.qr(proj2tangent_random_vectors(random_vectors))[0].detach()
        div_v0 = 0
        for i in range(manifold_dim):
            q = basis[:, :, i].view(v0.shape)
            div_v0 += (
                (
                    torch.autograd.grad(
                        (v0 * q).sum(), y, create_graph=False, retain_graph=True
                    )[  # WARNING: create_graph=False
                        0
                    ]
                    * q
                )
                .view(q.size(0), -1)
                .sum(1)
            )

        return div_v0

    def estimate_div_v0_closest(self, y, s):
        y.requires_grad_(True)
        py = self.bm_sde.M.proj2manifold(y)

        v0 = -1 * self.f(py, s)
        div_v0 = 0
        for i in range(y.size(1)):
            div_v0 += torch.autograd.grad(
                v0[:, i].sum(), y, create_graph=False, retain_graph=True
            )[0][:, i]
        return div_v0

    def exact_logp_via_integration(self, x, t, prior, steps=1000, method="closest"):
        logp = torch.zeros(x.shape[0], device=x.device)
        y = x
        y_shape = y.shape

        def f_func_(t_, y_):
            return self.f(y_, t_)

        def f_func_mat_(t_, y_):
            return self.f(y_.view(*y_shape), t_).view(y_shape[0], -1)

        if len(y_shape) == 2:
            f_func = f_func_
        else:
            f_func = f_func_mat_

        s = 0.0
        ds = t / steps
        for _ in range(steps):

            if method == "qr":
                div_v0 = self.estimate_div_v0_qr(y, s).detach()
            elif method == "closest":
                div_v0 = self.estimate_div_v0_closest(y, s).detach()
            else:
                raise ValueError("Unknown method: {}".format(method))

            logp = (logp - div_v0 * ds).detach()
            # take an integration step
            with torch.no_grad():
                ys = odeint(f_func, y, torch.tensor([s, s + ds], device=y.device))
            y = ys[-1].detach()

            y = self.bm_sde.M.proj2manifold(y).detach()
            s += ds
            # print(s / ds)
        logp0 = compute_prior(y, self.bm_sde.mani_name, prior)
        return logp0 + logp


class AmbientToriGenerative(torch.nn.Module):
    def __init__(self, drift_func, tori_dim: int):
        super().__init__()
        self.f = drift_func
        self.mani_name = "Tori"
        self.M = Tori(tori_dim)

    def g_increment(self, x, t, increment):
        return self.M.proj2tangent(x, increment)

    def sample(self, x, t, steps=100):
        return integration(x, steps, self, t, self.M.proj2manifold)


class AmbientToriBrownianMotion(nn.Module):
    def __init__(self, T: float, sample_method: str, tori_dim: int):
        super(AmbientToriBrownianMotion, self).__init__()
        self.T = T
        self.sample_method = sample_method
        self.mani_name = "Tori"
        self.M = Tori(tori_dim)

    def f(self, x, t):
        return torch.zeros_like(x)

    def g_increment(self, x, t, increment):
        return self.M.proj2tangent(x, increment)

    def sample_by_integration(self, x, t, steps):
        return integration(x, steps, self, t, self.M.proj2manifold)

    def sample_by_direct_sampling(self, x, t):
        theta0 = tori_ambient_to_theta(x)
        theta = (theta0 + torch.randn_like(theta0) * t ** 0.5) % (2 * np.pi)
        return tori_theta_to_ambient(theta)

    def sample(self, x, t, steps):
        if self.sample_method == "integration":
            return self.sample_by_integration(x, t, steps)
        elif self.sample_method == "directsampling":
            return self.sample_by_direct_sampling(x, t)
        else:
            raise Exception(f"Unknown sample method {self.sample_method}")


class AmbientHyperbolicPatchNormalLD(torch.nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T
        self.M = Lorentz()
        self.mani_name = "Hyperboloid"

    def f(self, x, t):
        return self.M.riem_score(x) / 2

    def g_increment(self, x, t, increment):
        return self.M.proj2tangent(x, increment)

    def sample(self, x, t, steps=100):
        return integration(x, steps, self, t, self.M.proj2manifold)
