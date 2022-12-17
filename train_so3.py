import matplotlib

matplotlib.use("Agg")
import torch
import argparse

from torch import nn

import wandb
from torch.utils.data import DataLoader
from data import datasets
import utils.utils as utils
from dsf import ImportanceSampler
import plotting
import os
import json
from helpers import logging, create
from manifold import Manifold
from models import ConcatMatMLP
from sdes import integration
import ipdb
from utils import exact_logp


_folder_name_keys = [
    "dataset",
    "batch_size",
    "lr",
    "num_iterations",
    "T0",
    "emb_size",
    "hidden_layers",
    "imp",
    "seed",
]


def get_args():
    parser = argparse.ArgumentParser()

    # i/o
    parser.add_argument("--dataset", type=str, choices=["so3"], default="so3")
    parser.add_argument("--dataroot", type=str, default="data")
    parser.add_argument("--saveroot", type=str, default="saved")
    parser.add_argument("--expname", type=str, default="so3")
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--sample_every", type=int, default=500)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--evaluate_every", type=int, default=5000)
    parser.add_argument("--evaluation_K", type=int, default=10)
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="number of integration steps for sampling",
    )
    parser.add_argument(
        "--inference_num_steps",
        type=int,
        default=100,
        help="number of integration steps for sampling for training",
    )
    parser.add_argument(
        "--evaluation_num_steps",
        type=int,
        default=100,
        help="number of integration steps for evaluation",
    )
    parser.add_argument("--seed", type=int, default=12345, help="random seed")

    # hparam
    parser.add_argument("--T0", type=float, default=2.0, help="integration time")
    parser.add_argument(
        "--emb_size", type=int, default=256, help="embedding size for hidden layers"
    )
    parser.add_argument(
        "--hidden_layers", type=int, default=2, help="num of hidden layers"
    )
    parser.add_argument(
        "--imp", type=int, default=0, help="importance sampling for time index"
    )

    # optimization
    parser.add_argument(
        "--div", type=str, choices=["deterministic"],
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=20000,
        help="number of training iterations",
    )
    parser.add_argument(
        "--sch", type=int, default=0, choices=[0, 1], help="using cosine lr scheduler"
    )

    return parser.parse_args()


args = get_args()
folder_tag = "riemannian-diffusion"
folder_name = "-".join([str(getattr(args, k)) for k in _folder_name_keys])
create(args.saveroot, folder_tag, args.expname, folder_name)
folder_path = os.path.join(args.saveroot, folder_tag, args.expname, folder_name)
print_ = lambda s: logging(s, folder_path)
print_(f"folder path: {folder_path}")
print_(str(args))
with open(os.path.join(folder_path, "args.txt"), "w") as out:
    out.write(json.dumps(args.__dict__, indent=4))

wandb.init(
    project="riemannian_diffusion",
    name=f"{args.expname}-{folder_name}",
    config=args.__dict__,
)
args = wandb.config

# load data
if args.dataset in ["so3"]:
    trainset = datasets.SpecialOrthogonalGroup(split="train")
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    valset = datasets.SpecialOrthogonalGroup(split="valid")
    valloader = DataLoader(
        valset, batch_size=args.val_batch_size, shuffle=False, num_workers=0
    )

    testset = datasets.SpecialOrthogonalGroup(split="test")
    testloader = DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0
    )
else:
    raise Exception(f"Dataset {args.dataset} not implemented")


# noinspection PyShadowingNames
def proj2manifold(x):
    u, _, vT = torch.linalg.svd(x)
    return u @ vT


# noinspection PyShadowingNames
def proj2tangent(x, v):
    shape = v.shape
    m = x.size(1)
    if v.ndim == 2:
        v = v.view(-1, m, m)
    return 0.5 * (v - x @ v.permute(0, 2, 1) @ x).view(shape)


# noinspection PyShadowingNames,PyAbstractClass
class OrthogonalGroup(Manifold):
    def __init__(self):
        super(OrthogonalGroup, self).__init__(ambient_dim=9, manifold_dim=3)

    @staticmethod
    def proj2manifold(x):
        return proj2manifold(x)

    @staticmethod
    def proj2tangent(x, v):
        return proj2tangent(x, v)


# noinspection PyShadowingNames,PyMethodMayBeStatic,PyUnusedLocal
class AmbientOrthogonalGroupBrownianMotion(nn.Module):
    def __init__(self, T: float):
        super(AmbientOrthogonalGroupBrownianMotion, self).__init__()
        self.T = T
        self.mani_name = "SO3"
        self.M = OrthogonalGroup()

    def f(self, x, t):
        return torch.zeros_like(x)

    def g_increment(self, x, t, increment):
        return proj2tangent(x, increment)

    def sample(self, x, t, steps):
        if len(t.size()) == 2:
            t = t.unsqueeze(2)
        return integration(x, steps, self, t, proj2manifold)


# noinspection PyShadowingNames,PyMethodMayBeStatic,PyUnusedLocal
class AmbientOrthogonalGroupGenerative(torch.nn.Module):
    def __init__(self, drift_func):
        super().__init__()
        self.f = drift_func
        self.mani_name = "SO3"

    def g_increment(self, x, t, increment):
        return proj2tangent(x, increment)

    def sample(self, x, t, steps=100):
        return integration(x, steps, self, t, proj2manifold)


a = ConcatMatMLP(
    dimx1=3,
    dimx2=3,
    dimt=1,
    emb_size=args.emb_size,
    num_hidden_layers=args.hidden_layers,
)
print_(a)
cuda = torch.cuda.is_available()
if cuda:
    print_("cuda available")
    a = a.cuda()
    device = "cuda"
else:
    print_("cuda unavailable")
    device = "cpu"


sde = AmbientOrthogonalGroupBrownianMotion(args.T0)
gsde = AmbientOrthogonalGroupGenerative(
    utils.get_drift(a, args.T0, OrthogonalGroup, sde)
)
opt = torch.optim.Adam(a.parameters(), lr=args.lr)

if args.sch:
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.num_iterations)
else:
    sch = None


if args.imp:
    print_("Initializing importance sampler")
    imp_sampler = ImportanceSampler(args.T0, ndim=10, stratified=True)
    if cuda:
        imp_sampler = imp_sampler.cuda()
        imp_sampler.is_cuda = True

    def get_estimand(x_):
        def estimand(t_):
            y_ = sde.sample(x_, t_, steps=args.inference_num_steps).requires_grad_(True)
            loss_ = -utils.elbo_so3(y_, t_, a, proj2manifold, proj2tangent, sde.T)
            return loss_

        return estimand

    for _ in range(100):
        x = iter(trainloader).next()
        if cuda:
            x = x.cuda()
        imp_sampler.step(get_estimand(x), x.size(0))

    wandb.log(
        {
            "q(t)": wandb.Image(
                plotting.fig2pil(
                    plotting.plot_hist(
                        imp_sampler.sample(1000)[0].view(-1, 1).cpu().data.numpy(),
                        xlabel="t",
                        ylabel="q(t)",
                        density=True,
                        bins=20,
                    )
                )
            )
        },
        step=0,
    )
    imp_sampler.eval()
else:
    imp_sampler = None
    get_estimand = None

if os.path.exists(os.path.join(folder_path, "checkpoint.pt")):
    print_("loading checkpoint")
    a, opt, sch, not_finished, count, imp_sampler = torch.load(
        os.path.join(folder_path, "checkpoint.pt")
    )
    gsde = AmbientOrthogonalGroupGenerative(
        utils.get_drift(a, args.T0, OrthogonalGroup, sde)
    )
else:
    not_finished = True
    count = 0

    a.eval()
    wandb.log(
        {
            "samples": wandb.Image(
                plotting.fig2pil(
                    plotting.plot_so3(
                        gsde.sample(
                            utils.sample_so3_uniform(n=5000).to(device),
                            args.T0,
                            args.num_steps,
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                )
            )
        },
        step=count,
    )
    a.train()

    wandb.log(
        {
            "test data": wandb.Image(
                plotting.fig2pil(plotting.plot_so3(testset.data[:5000]))
            )
        },
        step=count,
    )
    wandb.log(
        {"test density": wandb.Image(plotting.plot_so3_multimodal_density())},
        step=count,
    )


def evaluate(dataloader):
    a.eval()
    kelbos = []
    for x_val in dataloader:
        print(".", end="")
        if cuda:
            x_val = x_val.cuda()
        # if adaptive:
        #     kelbo = utils.adaptivekelbo_so3(sde=sde,
        #                                     y0=x_val,
        #                                     a_func=a,
        #                                     proj2manifold=proj2manifold,
        #                                     proj2tangent=proj2tangent,
        #                                     T=sde.T,
        #                                     K=args.evaluation_K,
        #                                     ds=sde.T / args.evaluation_num_steps,
        #                                     ds_min=1e-5,
        #                                     rtol=1e-5,
        #                                     atol=1e-5)
        # else:
        #     kelbo = utils.kelbo_so3(sde=sde,
        #                             y0=x_val,
        #                             a_func=a,
        #                             proj2manifold=proj2manifold,
        #                             proj2tangent=proj2tangent,
        #                             T=sde.T,
        #                             K=args.evaluation_K,
        #                             steps=args.evaluation_num_steps)

        if args.imp:
            kelbo = -imp_sampler.estimate(get_estimand(x_val), x.size(0))
        else:
            t_val = utils.stratified_uniform(x_val.size(0), args.T0)
            if cuda:
                t_val = t_val.cuda()
            y_val = sde.sample(
                x_val, t_val, steps=args.inference_num_steps
            ).requires_grad_(True)
            kelbo = utils.elbo_so3(
                y_val,
                t_val,
                a,
                OrthogonalGroup.proj2manifold,
                OrthogonalGroup.proj2tangent,
                args.T0,
            )
        kelbos.append(kelbo.detach())

        # kelbos.append(kelbo.detach())
    kelbo = torch.cat(kelbos).mean()
    a.train()
    return kelbo


def get_fig_density(a_func, bm_sde, prior_):
    """
    :test_samples: samples from the test set to overlay on the density
    """

    def func(xyz, steps=100):
        xyz = torch.Tensor(xyz)
        if cuda:
            xyz = xyz.cuda()
        return exact_logp(xyz, bm_sde, a_func, prior_, steps, method="qr")

    return plotting.plot_so3_multimodal_density(func)


while not_finished:
    for x in trainloader:
        if cuda:
            x = x.cuda()
        if args.imp:
            loss = imp_sampler.estimate(
                get_estimand(x), x.size(0)
            ).mean()  # TODO normalizing constant
        else:
            t = utils.stratified_uniform(x.size(0), args.T0)
            if cuda:
                t = t.cuda()
            y = sde.sample(x, t, steps=args.inference_num_steps).requires_grad_(True)
            loss = -utils.elbo_so3(
                y,
                t,
                a,
                OrthogonalGroup.proj2manifold,
                OrthogonalGroup.proj2tangent,
                args.T0,
            ).mean()

        if loss.isnan():
            ipdb.set_trace()

        opt.zero_grad()
        loss.backward()
        opt.step()
        if args.sch:
            sch.step()

        count += 1
        wandb.log({"loss": loss.item()}, step=count)

        if count % args.print_every == 0 or count == 1:
            print_(f"Iteration {count} \tloss {loss.item()}")

        if count % args.sample_every == 0:
            a.eval()
            wandb.log(
                {
                    "samples": wandb.Image(
                        plotting.fig2pil(
                            plotting.plot_so3(
                                gsde.sample(
                                    utils.sample_so3_uniform(n=5000).to(device),
                                    args.T0,
                                    args.num_steps,
                                )
                                .detach()
                                .cpu()
                                .numpy()
                            )
                        )
                    )
                },
                step=count,
            )
            a.train()

        if count % args.checkpoint_every == 0:
            torch.save(
                [a, opt, sch, not_finished, count, imp_sampler],
                os.path.join(folder_path, "checkpoint.pt"),
            )

        if count % args.evaluate_every == 0:
            # train_kelbo = evaluate(trainloader)
            # wandb.log({'train_kelbo': train_kelbo.item()}, step=count)
            # print_(f'Iteration {count} \t Training KELBO: {train_kelbo.item()}')
            val_kelbo = evaluate(valloader)
            wandb.log({"val_kelbo": val_kelbo.item()}, step=count)
            print_(f"Iteration {count} \t Validation KELBO: {val_kelbo.item()}")
            test_kelbo = evaluate(
                testloader
            )  # TODO uncomment after hyperparameter tuning with validation set
            wandb.log({"test_kelbo": test_kelbo.item()}, step=count)
            print_(f"Iteration {count} \t Test KELBO: {test_kelbo.item()}")

        if count >= args.num_iterations:
            not_finished = False
            print_("Finished training")
            torch.save(
                [a, opt, sch, not_finished, count, imp_sampler],
                os.path.join(folder_path, "checkpoint.pt"),
            )
            break

        if args.imp >= 1 and count % args.imp == 1:
            imp_sampler.train()
            for _ in range(10):
                x = iter(trainloader).next()
                if cuda:
                    x = x.cuda()
                imp_sampler.step(get_estimand(x), x.size(0))
            wandb.log(
                {
                    "q(t)": wandb.Image(
                        plotting.fig2pil(
                            plotting.plot_hist(
                                imp_sampler.sample(1000)[0]
                                .view(-1, 1)
                                .cpu()
                                .data.numpy(),
                                xlabel="t",
                                ylabel="q(t)",
                                density=True,
                                bins=20,
                            )
                        )
                    )
                },
                step=count,
            )
            imp_sampler.eval()

wandb.log(
    {"model density": wandb.Image(get_fig_density(a, sde, prior_=0))}, step=count,
)

val_kelbo = evaluate(valloader)
print_(f"Final Validation KELBO: {val_kelbo.item()}")
test_kelbo = evaluate(testloader)
print_(f"Final Test KELBO: {test_kelbo.item()}")
