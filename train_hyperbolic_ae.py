import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd 
import random 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib

matplotlib.use("Agg")
import torch

torch.set_default_dtype(torch.float64)
import argparse
import ipdb
import math
import wandb
from torch.utils.data import DataLoader
from data import datasets
import utils.utils as utils
from dsf import ImportanceSampler
import plotting
import os
import json
from helpers import logging, create
from lorentz import Lorentz
from sdes import (
    AmbientHyperbolicBrownianMotion,
    AmbientHyperbolicGenerative,
    AmbientHyperbolicPatchNormalLD,
)
from models import ConcatMLP

from ae import train_ae, Encoder, Decoder


_folder_name_keys = [
    "dataset",
    "batch_size",
    "lr",
    "num_iterations",
    "T0",
    "emb_size",
    "hidden_layers",
    "imp",
]


def get_args():
    parser = argparse.ArgumentParser()

    # i/o
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["1wrapped", "5gaussians", "bigcheckerboard", "mult_wrapped"],
        default="5gaussians",
    )
    parser.add_argument("--dataroot", type=str, default="data")
    parser.add_argument("--saveroot", type=str, default="saved")
    parser.add_argument("--expname", type=str, default="hyperbolic")
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--sample_every", type=int, default=500)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument(
        "--reload", action="store_true", default=False, help="Reload checkpoints"
    )
    parser.add_argument("--evaluate_every", type=int, default=5000)
    parser.add_argument("--evaluation_K", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="number of integration steps for sampling",
    )
    parser.add_argument(
        "--evaluation_num_steps",
        type=int,
        default=100,
        help="number of integration steps for evaluation",
    )

    # hparam
    parser.add_argument("--T0", type=float, default=2.0, help="integration time")
    parser.add_argument("--emb_size", type=int, default=512, help="num of hiddens")
    parser.add_argument("--hidden_layers", type=int, default=3, help="num of hiddens")
    parser.add_argument(
        "--imp", type=int, default=0, help="importance sampling for time index"
    )
    parser.add_argument(
        "--drift_coeff", type=float, default=1.0, help="Drift Coefficient"
    )

    # optimization
    parser.add_argument(
        "--div", type=str, choices=["deterministic"],
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--num_iterations", type=int, default=10000)
    parser.add_argument(
        "--wandb", action="store_true", default=True, help="Use wandb for logging"
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
utils.seed_everything(args.seed)

with open(os.path.join(folder_path, "args.txt"), "w") as out:
    out.write(json.dumps(args.__dict__, indent=4))

if args.wandb:
    wandb.init(
        project="riemannian_diffusion",
        name=f"{args.expname}-{folder_name}",
        config=args.__dict__,
    )
    args = wandb.config

cuda = torch.cuda.is_available()

# ################### autoencoder ###################
data_dir = 'dataset'

train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
test_dataset  = torchvision.datasets.MNIST(data_dir, train=False, download=True)

train_transform = transforms.Compose([
transforms.ToTensor(),
])

test_transform = transforms.Compose([
transforms.ToTensor(),
])

train_dataset.transform = train_transform
test_dataset.transform = test_transform

m=len(train_dataset)

train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])
batch_size=256

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

d = 3
encoder = Encoder(encoded_space_dim=d,fc2_input_dim=128)
decoder = Decoder(encoded_space_dim=d,fc2_input_dim=128)
train_ae(encoder, decoder)
encoder.eval()
decoder.eval()

#############################################################

if cuda:
    print_("cuda available")
    device = "cuda"
else:
    print_("cuda unavailable")
    device = "cpu"

# if args.dataset in ["1wrapped", "5gaussians", "bigcheckerboard", "mult_wrapped"]:
#     trainset = datasets.HyperbolicSynthDataset(
#         dataset=args.dataset, split="train", device=device
#     )
#     trainloader = DataLoader(
#         trainset, batch_size=args.batch_size, shuffle=True, num_workers=0
#     )

#     valset = datasets.HyperbolicSynthDataset(
#         dataset=args.dataset, split="valid", device=device
#     )
#     valloader = DataLoader(
#         valset, batch_size=args.val_batch_size, shuffle=True, num_workers=0
#     )

#     testset = datasets.HyperbolicSynthDataset(
#         dataset=args.dataset, split="test", device=device
#     )
#     testloader = DataLoader(
#         testset, batch_size=args.test_batch_size, shuffle=True, num_workers=0
#     )
# else:
#     raise Exception(f"Dataset {args.dataset} not implemented")

a = ConcatMLP(
    3, 1, args.emb_size, args.hidden_layers, actnorm_first=True
)  # TODO why adding actnorm blows up the loss
print_(a)
cuda = torch.cuda.is_available()

if cuda:
    a = a.cuda()

M = Lorentz()
loc = M.zero((3,)).to(device)
scale = torch.ones(1, M.dim).to(device)
sde = AmbientHyperbolicPatchNormalLD(args.T0)
gsde = AmbientHyperbolicGenerative(utils.get_drift(a, args.T0, M, sde))
prior = utils.ProjectedNormal(device, M, torch.zeros(1), torch.zeros(1))
opt = torch.optim.Adam(a.parameters(), lr=args.lr)


def gen_samples(gsde_: AmbientHyperbolicGenerative, num_samples=100, steps=100):
    prior_samples = prior.get_samples(num_samples).detach()
    samples_ = gsde_.sample(prior_samples, sde.T, steps=steps).detach()
    return samples_


# INITIALIZE ACTNORM
# x = iter(trainloader).next().to(device)
b = iter(train_loader).next()[0].to(device)
x = encoder(b).detach()
print("initialize actnorm: ", x.size())
t = utils.stratified_uniform(x.size(0), args.T0, device=device)
y = sde.sample(x, t).requires_grad_(True)

if args.imp:
    print_("Initializing importance sampler")
    imp_sampler = ImportanceSampler(args.T0, ndim=10, stratified=True, prior=prior)
    if cuda:
        imp_sampler = imp_sampler.cuda()
        imp_sampler.is_cuda = True

    def get_estimand(x_):
        def estimand(t_, prior):
            y_ = sde.sample(x_, t_).requires_grad_(True)
            loss_ = -utils.elbo_lorentz(
                prior=prior,
                sde=sde,
                x=x_,
                y=y_,
                s=t_,
                a_func=a,
                proj2manifold=M.proj2manifold,
                proj2tangent=M.proj2tangent,
            ).mean()
            return loss_

        return estimand

    for _ in range(100):
        x = iter(trainloader).next()
        if cuda:
            x = x.cuda()
        imp_sampler.step(get_estimand(x), x.size(0))

    if args.wandb:
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

if args.reload and os.path.exists(os.path.join(folder_path, "checkpoint.pt")):
    print_("loading checkpoint")
    a, opt, not_finished, count, imp_sampler = torch.load(
        os.path.join(folder_path, "checkpoint.pt")
    )
    gsde = AmbientHyperbolicGenerative(utils.get_drift(a, args.T0, M, sde))
else:
    not_finished = True
    count = 0

    a.eval()
    if args.wandb:
        samples = gen_samples(gsde_=gsde, num_samples=1024, steps=args.num_steps)
        # wandb.log(
        #     {
        #         "samples": wandb.Image(
        #             plotting.fig2pil(plotting.plot_scatter3D_poincare(samples))
        #         ),
        #         "poincare_samples": wandb.Image(
        #             plotting.fig2pil(plotting.plot_poincare_gen_samples(samples))
        #         ),
        #     },
        #     step=count,
        # )
    a.train()
    if args.wandb:
        # wandb.log(
        #     {
        #         "test data": wandb.Image(
        #             plotting.fig2pil(
        #                 plotting.plot_scatter3D_poincare(testset.data[0:1024])
        #             )
        #         ),
        #         "test_density": wandb.Image(
        #             plotting.fig2pil(plotting.plot_hyp_distr(args.dataset))
        #         ),
        #     },
        #     step=count,
        # )

        # wandb.log(
        #     {
        #         "agg posterior": wandb.Image(
        #             plotting.fig2pil(
        #                 plotting.plot_scatter3D_poincare(
        #                     sde.sample(
        #                         testset.data[0:1024].to(device),
        #                         args.T0 * torch.ones(1024, 1, device=device),
        #                     )
        #                 )
        #             )
        #         ),
        #         "test_density": wandb.Image(
        #             plotting.fig2pil(plotting.plot_hyp_distr(args.dataset))
        #         ),
        #     },
        #     step=count,
        # )

        # wandb.log(
        #     {
        #         "test kde": wandb.Image(
        #             plotting.fig2pil(
        #                 plotting.plot_poincare_gen_samples(testset.data[0:1024])
        #             )
        #         )
        #     },
        #     step=count,
        # )
        # prior_yz = torch.randn(10000, 2)
        # prior_xyz = M.invphi(prior_yz)
        # wandb.log(
        #     {
        #         "prior 3d": wandb.Image(
        #             plotting.fig2pil(plotting.plot_scatter3D_poincare(prior_xyz))
        #         )
        #     },
        #     step=count,
        # )
        pass


def plot_model_kelbo(adaptive: bool = False, batchsize: int = 64):
    a.eval()
    res_npts = 200
    on_mani, xy, _, _ = plotting.make_grid_hyp(res_npts)
    if cuda:
        on_mani = on_mani.cuda()

    model_kelbos = []
    for i in range(math.ceil(on_mani.shape[0] / batchsize)):
        batch = on_mani[i * batchsize : (i + 1) * batchsize]
        if adaptive:
            model_kelbo = utils.adaptivekelbo_lorentz(
                sde=sde,
                y0=batch,
                a_func=a,
                proj2manifold=M.proj2manifold,
                proj2tangent=M.proj2tangent,
                T=sde.T,
                K=args.evaluation_K,
                prior=prior,
                ds=sde.T / args.evaluation_num_steps,
                ds_min=1e-5,
                rtol=1e-2,
                atol=1e-2,
            )

        else:
            model_kelbo = utils.kelbo_lorentz(
                sde=sde,
                y0=batch,
                a_func=a,
                proj2manifold=M.proj2manifold,
                proj2tangent=M.proj2tangent,
                T=sde.T,
                K=args.evaluation_K,
                prior=prior,
                steps=args.evaluation_num_steps,
            )
        model_kelbos.append(model_kelbo.detach())
    model_kelbos = torch.cat(model_kelbos, dim=0)
    model_kelbos = torch.exp(model_kelbos)

    wandb.log(
        {
            "Model KELBO": wandb.Image(
                plotting.fig2pil(
                    plotting.plot_poincare_density(
                        xy,
                        model_kelbos,
                        res_npts,
                        uniform=True if args.dataset == "bigcheckerboard" else False,
                    )
                )
            )
        },
        step=count,
    )
    a.train()


def evaluate(dataloader, adaptive=False):
    kelbos = []
    a.eval()
    for x_val in dataloader:
        if cuda:
            x_val = x_val.cuda()
        if adaptive:
            kelbo = utils.adaptivekelbo_lorentz(
                sde=sde,
                y0=x_val,
                a_func=a,
                proj2manifold=M.proj2manifold,
                proj2tangent=M.proj2tangent,
                T=sde.T,
                K=args.evaluation_K,
                prior=prior,
                ds=sde.T / args.evaluation_num_steps,
                ds_min=1e-5,
                rtol=1e-2,
                atol=1e-2,
            )
        else:
            kelbo = utils.kelbo_lorentz(
                sde=sde,
                y0=x_val,
                a_func=a,
                proj2manifold=M.proj2manifold,
                proj2tangent=M.proj2tangent,
                T=sde.T,
                K=args.evaluation_K,
                prior=prior,
                steps=args.evaluation_num_steps,
            )
        kelbos.append(kelbo.detach())
    kelbo = torch.cat(kelbos).mean()
    a.train()
    return kelbo


def evaluate_exact_logp(dataloader, a_func, bm_sde, prior):
    logps = []
    counter = 0
    for x_ in dataloader:
        if cuda:
            x_ = x_.cuda()
        counter += 1
        if counter % 100 == 0:
            print(counter)
        logp = utils.exact_logp(
            x_ambient=x_,
            bm_sde=bm_sde,
            a_func=a_func,
            prior=prior,
            steps=100,
            method="qr",
        )
        logps.append(logp.detach())
        print(counter, "logp.mean until now:", torch.cat(logps).mean())
    logps = torch.cat(logps, axis=0)
    return logps


def get_fig_density(a_func, bm_sde, prior_):
    def func(x_, steps=100):
        return utils.exact_logp(
            x_ambient=x_,
            bm_sde=bm_sde,
            a_func=a_func,
            prior=prior_,
            steps=steps,
            method="qr",
        )

    fig = plotting.plot_density_on_hyperbolic(func, device=device, dataset=args.dataset)
    return fig


while not_finished:
    # for x in trainloader:
    for image_batch, _ in train_loader:
        image_batch = image_batch.to(device)
        print("image batch: ", image_batch.size())
        x = encoder(image_batch).detach()

        if cuda:
            x = x.to(device)
        if args.imp:
            loss = imp_sampler.estimate(get_estimand(x), x.size(0)).mean()
        else:
            t = utils.stratified_uniform(x.size(0), args.T0)
            if cuda:
                t = t.cuda()
            # print(x)
            y = sde.sample(x, t).requires_grad_(True)
            # print(y)
            loss = -utils.elbo_lorentz(
                prior, sde, x, y, t, a, M.proj2manifold, M.proj2tangent
            ).mean()
            # TODO method
        

        opt.zero_grad()
        loss.backward()
        opt.step()

        count += 1
        if args.wandb:
            wandb.log({"loss": loss.item()}, step=count)
        if count % args.print_every == 0 or count == 1:
            print_(f"Iteration {count} \tloss {loss.item()}")

        if count % 50 == 0:
            # Bonnie's samples :)
            n = 10
            samples = gsde.sample(utils.sample_spherical_uniform(n=n).to(device),
                                args.T0,
                                args.num_steps).detach()
            # print("Bonnie's samples: ", samples.size(), samples[0])
            plt.figure(figsize=(16,4.5))
            # targets = test_dataset.targets.numpy()
            # t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
            for i in range(n):
                ax = plt.subplot(2,n,i+1)
                #img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
                # encoder.eval()
                # decoder.eval()
                # with torch.no_grad():
                #     rec_img  = decoder(encoder(img))
                enc_img = samples[i]
                # print(enc_img)
                rec_img = decoder(torch.unsqueeze(enc_img, dim=0)).detach()
                # plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
                # ax.get_xaxis().set_visible(False)
                # ax.get_yaxis().set_visible(False)  
                # if i == n//2:
                #     ax.set_title('Original images')
                # ax = plt.subplot(2, n, i + 1 + n)
                plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)  
                if i == n//2:
                    ax.set_title('Reconstructed images')
            print("SAVING IMAGES")
            plt.savefig("hyperbolic_ae.png")
            plt.show()

        if count % args.sample_every == 0:
            a.eval()
            if args.wandb:
                samples = gen_samples(
                    gsde_=gsde, num_samples=1024, steps=args.num_steps
                )
                wandb.log(
                    {
                        "samples": wandb.Image(
                            plotting.fig2pil(plotting.plot_scatter3D_poincare(samples))
                        ),
                        "poincare_samples": wandb.Image(
                            plotting.fig2pil(
                                plotting.plot_poincare_gen_samples(samples)
                            )
                        ),
                    },
                    step=count,
                )
            a.train()
            wandb.log(
                {
                    "density plot": wandb.Image(
                        plotting.fig2pil(get_fig_density(a, sde, prior)),
                    )
                },
                step=count,
            )

        if count % args.checkpoint_every == 0:
            torch.save(
                [a, opt, not_finished, count, imp_sampler],
                os.path.join(folder_path, "checkpoint.pt"),
            )

        if count % args.evaluate_every == 0:
            train_exact_logp = evaluate_exact_logp(
                dataloader=trainloader, a_func=a, bm_sde=sde, prior=prior
            )
            train_exact_logp = train_exact_logp.mean()
            wandb.log({"train_exact_logp": train_exact_logp.item()}, step=count)
            print_(f"Train exact logp: {train_exact_logp.item()}")
            valid_exact_logp = evaluate_exact_logp(
                dataloader=valloader, a_func=a, bm_sde=sde, prior=prior
            )
            valid_exact_logp = valid_exact_logp.mean()
            wandb.log({"valid_exact_logp": valid_exact_logp.item()}, step=count)
            print_(f"Valid exact logp: {valid_exact_logp.item()}")
            test_exact_logp = evaluate_exact_logp(
                dataloader=testloader, a_func=a, bm_sde=sde, prior=prior
            )
            test_exact_logp = test_exact_logp.mean()
            wandb.log({"test_exact_logp": test_exact_logp.item()}, step=count)
            # train_kelbo = evaluate(trainloader, adaptive=False)
            # print_(f'Iteration {count} \t Training KELBO: {train_kelbo.item()}')
            plot_model_kelbo()
            val_kelbo = evaluate(valloader, adaptive=False)
            print_(f"Iteration {count} \t Validation KELBO: {val_kelbo.item()}")
            test_kelbo = evaluate(
                testloader, adaptive=False
            )  # TODO uncomment after hyperparameter tuning with validation set
            print_(f"Iteration {count} \t Test KELBO: {test_kelbo.item()}")
            # wandb.log({'Train NLL': -1*train_kelbo.item()}, step=count)
            wandb.log({"Val NLL": -1 * val_kelbo.item()}, step=count)
            wandb.log({"Test NLL": -1 * test_kelbo.item()}, step=count)

        if count >= args.num_iterations:
            not_finished = False
            print_("Finished training")
            torch.save(
                [a, opt, not_finished, count, imp_sampler],
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
            if args.wandb:
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

val_kelbo = evaluate(valloader, count, visualize=False)
print_(f"Final Validation KELBO: {val_kelbo.item()}")
test_kelbo = evaluate(testloader, count, visualize=False)
print_(f"Final Test KELBO: {test_kelbo.item()}")
