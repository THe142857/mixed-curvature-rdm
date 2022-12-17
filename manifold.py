import torch
import abc

from einops import rearrange
from geoopt.manifolds import PoincareBall as PoincareBallParent
from geoopt.manifolds.stereographic.math import _lambda_x, arsinh, tanh

from utils.mani_utils import cosh, sinh, tanh, arcosh, arsinh, artanh, sinhdiv, divsinh, logsinh
from utils.mani_utils import EPS as MY_EPS


EPS = 1e-9
MIN_NORM = 1e-15


# noinspection PyShadowingNames,PyAbstractClass
class Manifold:
    def __init__(self, ambient_dim, manifold_dim):
        """
        ambient_dim: dimension of ambient space
        manifold_dim: dimension of manifold
        """
        self.ambient_dim = ambient_dim
        self.manifold_dim = manifold_dim

    @staticmethod
    def phi(x):
        """
        x: point on ambient space
        return: point on euclidean patch
        """
        raise NotImplementedError

    @staticmethod
    def invphi(x_tilde):
        """
        x_tilde: point on euclidean patch
        return: point on ambient space
        """
        raise NotImplementedError

    @staticmethod
    def project(x):
        """
        x: manifold point on ambient space
        return: projection of x onto the manifold in the ambient space
        """
        raise NotImplementedError

    @staticmethod
    def g(x):
        """
        x: manifold point on ambient space
        return: differentiable determinant of the metric tensor at point x
        """
        raise NotImplementedError

    def norm(self, x, u, squared=False, keepdim=False):
        norm_sq = self.inner(x, u, u, keepdim)
        norm_sq.data.clamp_(MY_EPS[u.dtype])
        return norm_sq if squared else norm_sq.sqrt()


class Circle:

    @staticmethod
    def phi(x):
        return torch.atan2(x[:, 1], x[:, 0])

    @staticmethod
    def invphi(theta):
        return torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)

    @staticmethod
    def proj2manifold(x):
        return x / (x.norm(dim=1, keepdim=True) + EPS)

    @staticmethod
    def proj2tangent(x, v):
        return v - x * (x * v).sum(dim=1, keepdim=True)

    @staticmethod
    def g(x):
        batch_size = x.shape[0]
        # determinant of metric tensor is constant everywhere in the unit circle and it is 1
        return torch.ones(batch_size, 1).to(x)


def stabilize(x, eps=EPS):
    x = torch.where(x < 0., x+eps, x-eps)
    return x


def nozero(x, eps=EPS):
    x = torch.where(x < 0., x-eps, x+eps)
    return x


class Sphere(Manifold):
    @staticmethod
    def phi(x):
        theta = torch.acos(stabilize(x[:, 2]))  # [batch_size]
        phi = torch.atan2(x[:, 1], nozero(x[:, 0]))  # [batch_size]
        return torch.stack([theta, phi], dim=1)

    @staticmethod
    def invphi(x_tilde):
        theta = x_tilde[:, 0]
        phi = x_tilde[:, 1]
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        return torch.stack([x, y, z], dim=1)  # TODO: check if this is correct


    @staticmethod
    def proj2manifold(x):
        return x / (x.norm(dim=1).unsqueeze(-1) + EPS)

    @staticmethod
    def g(x):
        # for math check
        # https://math.stackexchange.com/questions/2527765/how-does-a-metric-tensor-describe-geometry-on-a-manifold
        x_tilde = Sphere.phi(x)
        theta = x_tilde[:, 0]
        return torch.sin(theta) ** 2

    @staticmethod
    def proj2tangent(x, v):
        shape = v.shape
        if len(shape) == 2:
            v = v.unsqueeze(2)
        x = x.unsqueeze(2)
        return (v - x * (x * v).sum(dim=1, keepdim=True)).view(shape)

    @staticmethod
    def orthogonal_projection_matrix(x):
        return torch.eye(3) - x[:, :, None] * x[:, None, :]

    def __init__(self):
        super().__init__(ambient_dim=3, manifold_dim=2)


class Tori(Manifold):
    def __init__(self, tori_dim: int):
        super(Tori, self).__init__(ambient_dim=tori_dim * 2, manifold_dim=tori_dim)

    @staticmethod
    def proj2manifold(x):
        b = x.shape[0]
        x = rearrange(x, 'b (d two) -> (b d) two', two=2)
        x = Circle.proj2manifold(x)
        x = rearrange(x, '(b d) two -> b (d two)', b=b)
        return x

    @staticmethod
    def proj2tangent(x, v):
        b = x.shape[0]
        x = rearrange(x, 'b (d two) -> (b d) two', two=2)
        v = rearrange(v, 'b (d two) -> (b d) two', two=2)
        v = Circle.proj2tangent(x, v)
        v = rearrange(v, '(b d) two-> b (d two)', b=b, two=2)
        return v