import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np


DELTA = 1e-7


def softplus(x, delta=DELTA):
    return F.softplus(x) + delta


def sigmoid(x, delta=DELTA):
    return torch.sigmoid(x) * (1 - delta) + 0.5 * delta


def logsigmoid(x):
    return -softplus(-x)


def log(x, delta=DELTA):
    return torch.log(x * 1e2 + DELTA) - np.log(1e2 + delta)


def logit(x):
    return log(x) - log(1 - x)


def act_a(x):
    return F.softplus(x)


def act_b(x):
    return x


def act_w(x):
    return F.softmax(x, dim=2)


def oper(array, operetor, axis=-1, keepdims=False):
    a_oper = operetor(array)
    if keepdims:
        shape = []
        for j, s in enumerate(array.size()):
            shape.append(s)
        shape[axis] = -1
        a_oper = a_oper.view(*shape)
    return a_oper


def log_sum_exp(a, axis=-1, sum_op=torch.sum):
    def maximum(x):
        return x.max(axis)[0]

    a_max = oper(a, maximum, axis, True)

    def summation(x):
        return sum_op(torch.exp(x - a_max), axis)

    b = torch.log(oper(a, summation, axis, True)) + a_max
    return b


def sigmoid_flow(x, logdet=0, ndim=4, params=None, delta=DELTA, logit_end=True):
    """
    element-wise sigmoidal flow described in `Neural Autoregressive Flows` (https://arxiv.org/pdf/1804.00779.pdf)
    :param x: input
    :param logdet: accumulation of log-determinant of jacobian
    :param ndim: number of dimensions of the transform
    :param params: parameters of the transform (batch_size x dimensionality of features x ndim*3 parameters)
    :param delta: small value to deal with numerical stability
    :param logit_end: whether to logit-transform it back to the real space
    :return:
    """
    assert params is not None, 'parameters not provided'
    assert params.size(2) == ndim*3, 'params shape[2] does not match ndim * 3'

    a = act_a(params[:, :, 0 * ndim: 1 * ndim])
    b = act_b(params[:, :, 1 * ndim: 2 * ndim])
    w = act_w(params[:, :, 2 * ndim: 3 * ndim])

    pre_sigm = a * x[:, :, None] + b
    sigm = torch.sigmoid(pre_sigm)
    x_pre = torch.sum(w * sigm, dim=2)

    logj = F.log_softmax(
      params[:, :, 2 * ndim: 3 * ndim], dim=2) + logsigmoid(pre_sigm) + logsigmoid(-pre_sigm) + log(a)
    logj = log_sum_exp(logj, 2).sum(2)
    if not logit_end:
        return x_pre, logj.sum(1) + logdet

    x_pre_clipped = x_pre * (1 - delta) + delta * 0.5
    x_ = log(x_pre_clipped) - log(1 - x_pre_clipped)
    xnew = x_

    logdet_ = logj + np.log(1 - delta) - (log(x_pre_clipped) + log(-x_pre_clipped + 1))
    logdet = logdet_.sum(1) + logdet

    return xnew, logdet


class SequentialFlow(torch.nn.Module):

    def __init__(self, flows):
        super(SequentialFlow, self).__init__()
        self.flows = torch.nn.ModuleList(flows)

    def forward_transform(self, x, logdet=0):
        for flow in self.flows:
            x, logdet = flow.forward_transform(x, logdet)
        return x, logdet


class ElementwiseDSF(torch.nn.Module):

    def __init__(self, dim, ndim, logit_end=True):
        super().__init__()

        self.dim = dim
        self.ndim = ndim
        self.logit_end = logit_end
        self.params = nn.Parameter(torch.zeros(1, dim, 3 * ndim))

        # init params
        self.params.data[:, :, :ndim].zero_().add_(np.log(np.exp(1) - 1))

    def forward_transform(self, x, logdet):
        return sigmoid_flow(x,
                            logdet=logdet,
                            ndim=self.ndim,
                            params=self.params,
                            delta=DELTA,
                            logit_end=self.logit_end)


class FixedScalingFlow(torch.nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
        self.logdet = np.log(scale)

    def forward_transform(self, x, logdet):
        return self.scale * x, logdet + np.prod(x.shape[1:]) * self.logdet


def log_standard_logistic(x):
    return logsigmoid(x) + logsigmoid(-x)


def stratified_uniform(n, r):
    """
    generate a batch of uniform([0, r]) samples in a stratefied manner (to reduce variance)
    """
    indices = np.arange(n)
    np.random.shuffle(indices)
    t = (torch.rand(n) / n + torch.Tensor(indices) / n).unsqueeze(-1) * r
    return t


class ImportanceSampler(nn.Module):

    def __init__(self, T=1.0, ndim=8, stratified=False, prior=None):
        super().__init__()

        self.T = T
        self.stratified = stratified
        self.flow = SequentialFlow(
            [ElementwiseDSF(1, ndim, False), FixedScalingFlow(T)])

        self.optim = torch.optim.Adam(self.parameters(), lr=0.001)
        self.is_cuda = False
        self.prior = prior

    # noinspection PyUnresolvedReferences
    def sample(self, n):
        if self.stratified:
            t0 = logit(stratified_uniform(n, 1))
        else:
            t0 = logit(torch.rand(n, 1))

        if self.is_cuda:
            t0 = t0.cuda()

        logq0 = log_standard_logistic(t0).sum(1)
        t, logdet = self.flow.forward_transform(t0, 0)

        return t, logq0 - logdet

    def estimate(self, func, n_samples):
        t, logq = self.sample(n_samples)
        return func(t.detach().requires_grad_(False), self.prior) / (torch.exp(logq) + DELTA) / self.T

    def loss(self, func, n_samples):
        """
        Eq[f/q] = int f
        var_q(f/q) = E_q[(f/q)^2] - mu^2
        grad var_q(f/q) = grad E_q[(f/q)^2]
        """
        t, logq = self.sample(n_samples)
        return (func(t, self.prior) / (torch.exp(logq) + DELTA) / self.T) ** 2 + 0.0001 * logq  # regularization

    def step(self, func, n_samples):
        loss = self.loss(func, n_samples).mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss
