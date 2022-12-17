import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as st
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.utils import tori_ambient_to_theta
from wrappednormal import WrappedNormal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal
from PIL import Image
import cartopy.crs as ccrs
from cartopy import config
import os
from data import orthogonal_group
import plotly.graph_objects as go
import io
import ipdb
from tqdm import tqdm


def plot_circular_hist(y, bins=40, fig=None):
    """
    :param y: angle values in [0, 1]
    :param bins: number of bins
    :param fig: matplotlib figure object
    """

    theta = np.linspace(0, 2 * np.pi, num=bins, endpoint=False)
    radii = np.histogram(y, bins, range=(0, 2 * np.pi), density=True)[0]

    # # Display width
    width = (2 * np.pi) / (bins * 1.25)

    # Construct ax with polar projection
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    # Set Orientation
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_xlim(0, 2 * np.pi)  # workaround for a weird issue
    ax.set_xticks(np.pi / 180.0 * np.linspace(0, 360, 8, endpoint=False))

    # Plot bars:
    _ = ax.bar(x=theta, height=radii, width=width, color="gray")

    # Grid settings
    ax.set_rgrids([])

    return fig, ax


def fig2pil(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = fig.canvas.tostring_rgb()
    pil_im = Image.frombytes("RGB", (w, h), buf)
    plt.close("all")
    return pil_im


def plot_scatter3D(xyz, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), zlim=(-1.0, 1.0)):

    fig = plt.figure(figsize=(8, 8), dpi=80)
    ax = plt.axes(projection="3d")
    ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.5)
    ax.axes.set_xlim3d(xlim[0], xlim[1])
    ax.axes.set_ylim3d(ylim[0], ylim[1])
    ax.axes.set_zlim3d(zlim[0], zlim[1])
    return fig


def eulerAnglesToRotationMatrix(theta):
    """https://learnopencv.com/rotation-matrix-to-euler-angles/"""

    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta[0]), -math.sin(theta[0])],
            [0, math.sin(theta[0]), math.cos(theta[0])],
        ]
    )

    R_y = np.array(
        [
            [math.cos(theta[1]), 0, math.sin(theta[1])],
            [0, 1, 0],
            [-math.sin(theta[1]), 0, math.cos(theta[1])],
        ]
    )

    R_z = np.array(
        [
            [math.cos(theta[2]), -math.sin(theta[2]), 0],
            [math.sin(theta[2]), math.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def isRotationMatrix(R):
    """https://learnopencv.com/rotation-matrix-to-euler-angles/"""
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    """https://learnopencv.com/rotation-matrix-to-euler-angles/"""

    # assert(isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def plot_so3(x):
    return plot_scatter3D(
        np.array([rotationMatrixToEulerAngles(x[i]) for i in range(len(x))]),
        (-math.pi, math.pi),
        (-math.pi / 2, math.pi / 2),
        (-math.pi, math.pi),
    )


def plot_so3_multimodal_density(log_p_fn=orthogonal_group.log_multimodal):
    npoints = 40j
    X, Y, Z = np.mgrid[-3.14:3.14:npoints, -1.57:1.57:npoints, -3.14:3.14:npoints]
    points = np.concatenate(
        [
            eulerAnglesToRotationMatrix([x, y, z])[None]
            for x, y, z in zip(X.flatten(), Y.flatten(), Z.flatten())
        ],
        0,
    )

    values = np.exp(log_p_fn(points)).flatten()

    vol = 4 * np.pi ** 2 / abs(npoints) ** 3
    values /= values.sum() * vol

    fig = go.Figure(
        data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=values.flatten(),
            isomin=0.1,
            isomax=0.8,
            opacity=0.1,
            surface_count=20,
        )
    )

    buf = io.BytesIO()
    fig.write_image(buf, width=800, height=600, scale=2)
    buf.seek(0)

    return Image.open(buf)


def euc2lonlat(xyz):
    theta = np.arccos(xyz[:, 2])
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    return phi / np.pi * 180, -(theta / np.pi * 2 * 90) + 90


def lonlat2euc(lon, lat):
    phi = lon / 180.0 * np.pi
    theta = (90 - lat) / 180.0 * np.pi
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=1)


def plot_scatter_plate_carree(xyz, superpose=True):
    x, y = euc2lonlat(xyz)

    proj = ccrs.PlateCarree()
    fname = os.path.join(
        config["repo_data_dir"],
        "raster",
        "natural_earth",
        "50-natural-earth-1-downsampled.png",
    )

    fig = plt.figure(figsize=(7.75, 4))
    plt.subplot(111, projection=proj)
    if superpose:
        plt.imshow(
            plt.imread(fname),
            origin="upper",
            transform=proj,
            extent=[-180, 180, -90, 90],
            alpha=1.0,
        )

    plt.scatter(x, y, s=2, color="r", alpha=1.0)
    plt.axis("off")
    plt.pause(1)
    plt.tight_layout()
    return fig


def plot_hist(x, xlabel=None, ylabel=None, **kwargs):
    fig = plt.figure(figsize=(7, 7))
    plt.hist(x, **kwargs)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(xlabel)
    plt.pause(1)
    plt.tight_layout()
    return fig


def plot_scatter3D_poincare(xyz):
    try:
        xyz = xyz.detach().cpu().numpy()
    except:
        pass
    fig = plt.figure(figsize=(8, 8), dpi=80)
    ax = plt.axes(projection="3d")
    ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.5)
    ax.axes.set_xlim3d()
    ax.axes.set_ylim3d()
    ax.axes.set_zlim3d()
    return fig


def plot_poincare_gen_samples(data, npts=500):
    from lorentz import Lorentz

    fig = plt.figure()
    plt.clf()
    M = Lorentz()
    data = torch.Tensor(data)
    data = M.to_poincare(data).numpy()
    data_x = data[:, 0]
    data_y = data[:, 1]

    bp = torch.linspace(-1, 1, npts)
    xx, yy = torch.meshgrid((bp, bp))
    twodim = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    threedim = M.unsqueeze_tangent(twodim)
    on_mani = M.exp0(threedim)
    xy_poincare = M.to_poincare(on_mani).t().numpy()
    # Peform the kernel density estimate
    positions = np.vstack([xx.numpy().ravel(), yy.numpy().ravel()])

    x_poincare = xy_poincare.T[:, 0].reshape(npts, npts)
    y_poincare = xy_poincare.T[:, 1].reshape(npts, npts)
    values = np.vstack([data_x, data_y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    plt.pcolormesh(x_poincare, y_poincare, f, cmap="magma")
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.axis("off")
    plt.axis("equal")
    # print("Saved to: %s" % (filename))
    # plt.savefig(filename)

    plt.clf()
    plt.pcolormesh(xx.numpy(), yy.numpy(), f, cmap="magma")
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    # plt.axis('off')
    plt.axis("equal")
    return fig


def plot_poincare_density(xy_poincare, prob, npts, cmap="magma", uniform=False):
    prob = torch.tensor(prob)
    fig = plt.figure()
    plt.clf()
    x = xy_poincare[:, 0].cpu().numpy().reshape(npts, npts)
    y = xy_poincare[:, 1].cpu().numpy().reshape(npts, npts)
    prob = prob.detach().cpu().numpy().reshape(npts, npts)
    if not uniform:
        plt.pcolormesh(x, y, prob, cmap=cmap)
    else:  # uniform color
        colormap = plt.cm.get_cmap(cmap)
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        prob[prob > 0] = 0.5
        plt.pcolormesh(x, y, prob, cmap=cmap, norm=norm)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    # plt.axis('off')
    plt.axis("equal")
    # print("Saved to: %s" % (namestr))
    # plt.savefig(namestr)
    return fig


## Plot true and model code
def make_grid_hyp(npts):
    from lorentz import Lorentz

    M = Lorentz()
    bp = torch.linspace(-5, 5, npts)
    xx, yy = torch.meshgrid((bp, bp))
    twodim = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    threedim = M.unsqueeze_tangent(twodim)
    on_mani = M.exp0(threedim)
    xy = M.to_poincare(on_mani)
    dummy = -1
    log_detjac = -M.logdetexp(dummy, threedim)
    return on_mani, xy, log_detjac, twodim


def plot_hyp_distr(distr=None, res_npts=500):
    on_mani, xy, log_detjac, twodim = make_grid_hyp(res_npts)
    if distr == "1wrapped":
        probs = true_1wrapped_probs(on_mani)
    elif distr == "5gaussians":
        probs = true_5gaussians_probs(on_mani, twodim)
    elif distr == "bigcheckerboard":
        probs = true_bigcheckerboard_probs(on_mani, twodim)
    elif distr == "mult_wrapped":
        probs = true_mult_wrapped_probs(on_mani)
    elif distr == "model":
        probs = true_mult_wrapped_probs(on_mani)

    return plot_poincare_density(
        xy, probs, res_npts, uniform=True if distr == "bigcheckerboard" else False
    )


def true_1wrapped_probs(on_mani):
    from lorentz import Lorentz

    M = Lorentz()
    mu = torch.Tensor([-1.0, 1.0]).unsqueeze(0)
    std_v = 0.75
    std_1 = torch.Tensor([[std_v], [std_v]]).T
    radius = torch.ones(1)
    mu_h = M.exp0(M.unsqueeze_tangent(mu))
    device = torch.device("cuda:0")
    distr = WrappedNormal(device, M, mu_h, std_1)
    prob = torch.exp(distr.log_prob(on_mani))
    return prob


def true_5gaussians_probs(on_mani, twodim):
    scale = 3
    centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
    centers = torch.tensor([(scale * x, scale * y) for x, y in centers])

    prob = 0
    for c in centers:
        loc = torch.ones_like(twodim) * torch.tensor(c)
        distr = MultivariateNormal(loc, 0.25 * torch.eye(2))
        prob += torch.exp(distr.log_prob(twodim))
    prob /= len(centers)
    return prob


def true_bigcheckerboard_probs(on_mani, twodim):
    s = 1.5  # side length

    def in_board(z, s):
        """Whether z is in the checkerboard of side length s"""
        if 0 <= z[0] < s or -2 * s <= z[0] < -s:
            return 0 <= z[1] < s or -2 * s <= z[1] < -s
        elif -2 * s <= z[0] < 2 * s:
            return s <= z[1] < 2 * s or -s <= z[1] < 0
        else:
            return 0

    prob = torch.zeros(twodim.shape[0])
    for i in range(twodim.shape[0]):
        prob[i] = in_board(twodim[i, :], s)

    prob /= torch.sum(prob)
    return prob


def true_mult_wrapped_probs(on_mani):
    from lorentz import Lorentz

    M = Lorentz()
    s = 1.3
    centers = [
        torch.tensor([[0.0, s, s]]),
        torch.tensor([[0, -s, -s]]),
        torch.tensor([[0.0, -s, s]]),
        torch.tensor([[0, s, -s]]),
    ]
    centers = [M.projx(center) for center in centers]
    n = on_mani.shape[0]
    var1 = 0.3
    var2 = 1.5
    scales = [
        torch.tensor([[var1, var2]]),
        torch.tensor([[var1, var2]]),
        torch.tensor([[var2, var1]]),
        torch.tensor([[var2, var1]]),
    ]
    distrs = []
    for i in range(len(centers)):
        loc = centers[i].repeat(n, 1)
        scale = scales[i].repeat(n, 1)
        distrs.append(WrappedNormal(M, loc, scale))
    prob = torch.zeros(on_mani.shape[0])
    for distr in distrs:
        prob += torch.exp(distr.log_prob(on_mani))
    prob /= len(centers)
    return prob


def plot_theta(theta):
    theta = theta * 180.0 / np.pi
    return scatter_plot(theta[:, 0], theta[:, 1])


def plot_theta_of_ambient(x):
    theta = tori_ambient_to_theta(x)
    theta = theta.detach().cpu().numpy()
    return plot_theta(theta)


def scatter_plot(x, y):
    fig = plt.figure(figsize=(4, 4))

    plt.scatter(x, y, s=10, alpha=0.1)
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$\psi$")
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.tight_layout()

    return fig


def plot_density_on_theta(exact_logp, device, test_samples=None, title=None):
    """
    test_samples: samples from the test set to overlay on the density
    """
    phi = torch.linspace(-180, 180, 100, device=device)
    psi = torch.linspace(-180, 180, 100, device=device)
    tv, pv = torch.meshgrid([phi, psi], indexing="ij")

    points = torch.stack([tv.reshape(-1), pv.reshape(-1)], dim=1)
    logps = []
    inner_batch_size = 640
    for i in range(math.ceil(points.shape[0] / inner_batch_size)):
        point = points[
            i * inner_batch_size : min((i + 1) * inner_batch_size, points.shape[0])
        ]
        logp = exact_logp(point, steps=100)
        logps.append(logp.detach())
    logps = torch.cat(logps)

    ps = torch.exp(logps)
    ps = ps.reshape(tv.shape)

    fig_colormesh = plt.figure(figsize=(4, 4))
    plt.pcolormesh(
        tv.detach().cpu().numpy(),
        pv.detach().cpu().numpy(),
        ps.detach().cpu().numpy(),
        cmap="Blues",
        zorder=0,
    )
    if test_samples is not None:
        plt.scatter(
            test_samples[:, 0], test_samples[:, 1], s=0.2, alpha=1.0, zorder=1, c="r"
        )
    plt.xlabel(r"$\phi$", fontsize=15)
    plt.ylabel(r"$\psi$", fontsize=15)
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)

    fig_contour, ax = plt.subplots(1, 1, figsize=(6, 6))
    if title is not None:
        ax.set_title(title, fontsize=15)
    cp = ax.contourf(
        tv.detach().cpu().numpy(),
        pv.detach().cpu().numpy(),
        ps.reshape(tv.shape).detach().cpu().numpy(),
        norm=matplotlib.colors.LogNorm(),
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig_contour.colorbar(cp, cax=cax)
    cbar.ax.set_yticklabels([f"{s:.1f}" for s in np.log(cbar.ax.get_yticks())])
    cbar.set_label("log likelihood", rotation=270)
    if test_samples is not None:
        ax.scatter(
            test_samples[:, 0], test_samples[:, 1], s=0.2, alpha=1.0, zorder=1, c="r"
        )
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\phi$", fontsize=15)
    ax.set_ylabel(r"$\psi$", fontsize=15)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)

    return fig_colormesh, fig_contour


def plot_ambient(x):
    # noinspection PyTypeChecker
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)

    axs[0].set_aspect("equal")
    axs[1].set_aspect("equal")
    axs[0].set_xlim(-1, 1)
    axs[0].set_ylim(-1, 1)
    axs[1].set_xlim(-1, 1)
    axs[1].set_ylim(-1, 1)

    axs[0].scatter(x[:, 0], x[:, 1], s=10, alpha=0.4)
    axs[0].set_xlabel("x0")
    axs[0].set_ylabel(r"x1")
    axs[1].set_xlabel("x2")
    axs[1].set_ylabel(r"x3")
    axs[1].scatter(x[:, 2], x[:, 3], s=10, alpha=0.4)
    plt.tight_layout()
    return fig


def plot_density_on_map(exact_logp, device, test_samples=None, title=None):
    lon = torch.linspace(-180.0, 180.0, 100, device=device)
    lat = torch.linspace(-90.0, 90.0, 100, device=device)
    tv, pv = torch.meshgrid([lon, lat], indexing="ij")
    points = lonlat2euc(tv.reshape(-1), pv.reshape(-1))
    logps = []
    inner_batch_size = 640
    for i in tqdm(range(math.ceil(points.shape[0] / inner_batch_size))):
        point = points[
            i * inner_batch_size : min((i + 1) * inner_batch_size, points.shape[0])
        ]
        logp = exact_logp(point, steps=10000)
        logps.append(logp.detach())

    logps = torch.cat(logps)
    ps = torch.exp(logps)
    ps = ps.reshape(tv.shape)
    proj = ccrs.PlateCarree()
    fname = os.path.join(
        config["repo_data_dir"],
        "raster",
        "natural_earth",
        "50-natural-earth-1-downsampled-dark.png",
    )
    fig = plt.figure(figsize=(7.75, 4))
    ax = plt.subplot(111, projection=proj)
    # first show the colormesh of the density
    cm = ax.contourf(
        tv.detach().cpu().numpy(),
        pv.detach().cpu().numpy(),
        ps.detach().cpu().numpy(),
        cmap="viridis",
        zorder=0,
    )
    cbar = fig.colorbar(cm, shrink=0.79, pad=0.01)
    cbar.set_label("likelihood", rotation=270, labelpad=10)
    # overlay the image of the earth
    ax.imshow(
        plt.imread(fname),
        origin="upper",
        transform=proj,
        extent=[-180, 180, -90, 90],
        alpha=0.5,
        zorder=1,
    )
    # overlay then the test samples
    test_lon, test_lat = euc2lonlat(test_samples)
    if test_samples is not None:
        ax.scatter(test_lon, test_lat, s=0.2, alpha=1.0, zorder=2, c="r")
    ax.set_xlabel(r"lon$", fontsize=15)
    ax.set_ylabel(r"lat$", fontsize=15)
    if title is not None:
        ax.set_title(title, fontsize=15)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    plt.axis("off")
    plt.pause(1)
    plt.tight_layout()

    return fig


def plot_density_on_hyperbolic(exact_logp, device, dataset):
    res_npts = 200
    on_mani, xy, _, _ = make_grid_hyp(res_npts)
    on_mani = on_mani.to(device)

    logps = []
    batchsize = 640
    for i in range(math.ceil(on_mani.shape[0] / batchsize)):
        batch = on_mani[i * batchsize : (i + 1) * batchsize]
        logp = exact_logp(batch, steps=100)
        logps.append(logp.detach())
    logps = torch.cat(logps, dim=0)
    ps = torch.exp(logps)

    fig = plot_poincare_density(
        xy, ps, res_npts, uniform=True if dataset == "bigcheckerboard" else False
    )

    return fig
