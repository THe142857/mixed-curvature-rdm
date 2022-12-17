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
matplotlib.use('Agg')
import torch
import argparse
import wandb
from torch.utils.data import DataLoader
from data import datasets
import utils.utils as utils
from dsf import ImportanceSampler
import plotting
import os
import json
from helpers import logging, create, ExponentialMovingAverage
from manifold import Sphere
from sdes import AmbientSphericalBrownianMotion, AmbientSphericalGenerative
from models import ConcatSinMLP as ConcatMLP

from ae import train_ae, Encoder, Decoder

def get_args():
    parser = argparse.ArgumentParser()

    # i/o
    parser.add_argument('--dataset', type=str,
                        choices=['UniformSlice', 'volcano', 'earthquake', 'flood', 'fire'],
                        default='UniformSlice')
    parser.add_argument('--dataroot', type=str, default='data')
    parser.add_argument('--saveroot', type=str, default='saved')
    parser.add_argument('--expname', type=str, default='sphere')
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--sample_every', type=int, default=500)
    parser.add_argument('--checkpoint_every', type=int, default=500)
    parser.add_argument('--evaluate_every', type=int, default=5000)
    parser.add_argument('--evaluation_K', type=int, default=10)
    parser.add_argument('--num_steps', type=int, default=100,
                        help='number of integration steps for sampling')
    parser.add_argument('--inference_num_steps', type=int, default=100,
                        help='number of integration steps for sampling for training')
    parser.add_argument('--evaluation_num_steps', type=int, default=100,
                        help='number of integration steps for evaluation')
    parser.add_argument('--seed', type=int, default=12345,
                        help='seed for dataset split')

    # hparam
    parser.add_argument('--T0', type=float, default=2.0,
                        help='integration time')
    parser.add_argument('--emb_size', type=int, default=256,
                        help='num of hiddens')
    parser.add_argument('--hidden_layers', type=int, default=2,
                        help='num of hiddens')
    parser.add_argument('--imp', type=int, default=0,
                        help='importance sampling for time index')

    # optimization
    parser.add_argument('--div', type=str, choices=['deterministic'], )
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--num_iterations', type=int, default=10000)
    parser.add_argument('--sch', type=int, default=0, choices=[0, 1],
                        help='using cosine lr scheduler')
    parser.add_argument('--warmup_iters', type=int, default=0,
                        help='lr warmup')
    parser.add_argument('--ema_decay', type=float, default=0.995, help='ema decay (polyak averaging)')

    # mode selection
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'report'])

    return parser.parse_args()


# noinspection PyShadowingNames
def update_lr(optimizer, itr, args):
    iter_frac = min(float(itr + 1) / max(args.warmup_iters, 1), 1.0)
    lr = args.lr * iter_frac
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_estimand(x_):
    def estimand(t_, prior=None):
        assert prior is None  # just for compatibility with the estimand for other manifolds so imp_sampler can work with them
        y_ = sde.sample(x_, t_, args.inference_num_steps).requires_grad_(True)
        loss_ = -1 * utils.elbo_sphere(y_, t_, a, proj2manifold=Sphere.proj2manifold,
                                        proj2tangent=Sphere.proj2tangent, T=sde.T)  # TODO method
        return loss_

    return estimand


def evaluate(dataloader, a_module, adaptive=True):
    a_module.eval()
    kelbos = []
    for x_val in dataloader:
        if cuda:
            x_val = x_val.cuda()
        if adaptive:
            kelbo = utils.adaptivekelbo_sphere(sde=sde,
                                               y0=x_val,
                                               a_func=a_module,
                                               proj2manifold=Sphere.proj2manifold,
                                               proj2tangent=Sphere.proj2tangent,
                                               T=sde.T,
                                               K=args.evaluation_K,
                                               ds=sde.T / args.evaluation_num_steps,
                                               ds_min=1e-5,
                                               rtol=1e-2,
                                               atol=1e-2)
        else:
            kelbo = utils.kelbo_sphere(sde=sde,
                                       y0=x_val,
                                       a_func=a_module,
                                       proj2manifold=Sphere.proj2manifold,
                                       proj2tangent=Sphere.proj2tangent,
                                       T=sde.T,
                                       K=args.evaluation_K,
                                       steps=args.evaluation_num_steps)
        kelbos.append(kelbo.detach())
    kelbo = torch.cat(kelbos).mean()
    a_module.train()
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
        logp = utils.exact_logp(x_ambient=x_, bm_sde=bm_sde, a_func=a_func, prior=prior, steps=args.evaluation_num_steps)
        logps.append(logp.detach())
        print(counter, 'logp.mean until now:', torch.cat(logps).mean())
    logps = torch.cat(logps, axis=0)
    return logps


def get_fig_density(a_func, bm_sde, prior_, test_samples=None, title=None):
    def func(x_, steps=100):
        return utils.exact_logp(x_ambient=x_, bm_sde=bm_sde, a_func=a_func, prior=prior_, steps=steps)

    fig = plotting.plot_density_on_map(func, test_samples=test_samples, device=device, title=title)
    return fig


def evaluate_step():
    train_exact_logp = evaluate_exact_logp(dataloader=trainloader, a_func=ema_a, bm_sde=sde, prior=prior)
    train_exact_logp = train_exact_logp.mean()
    wandb.log({'train_exact_logp': train_exact_logp.item()}, step=count)
    print_(f'Train exact logp: {train_exact_logp.item()}')
    valid_exact_logp = evaluate_exact_logp(dataloader=valloader, a_func=ema_a, bm_sde=sde, prior=prior)
    valid_exact_logp = valid_exact_logp.mean()
    wandb.log({'valid_exact_logp': valid_exact_logp.item()}, step=count)
    print_(f'Valid exact logp: {valid_exact_logp.item()}')
    test_exact_logp = evaluate_exact_logp(dataloader=testloader, a_func=ema_a, bm_sde=sde, prior=prior)
    test_exact_logp = test_exact_logp.mean()
    wandb.log({'test_exact_logp': test_exact_logp.item()}, step=count)
    val_kelbo = evaluate(valloader, a_module=ema_a, adaptive=True)
    wandb.log({'val_kelbo': val_kelbo.item()}, step=count)
    print_(f'Iteration {count} \t Validation KELBO: {val_kelbo.item()}')
    test_kelbo = evaluate(testloader, a_module=ema_a, adaptive=True)
    wandb.log({'test_kelbo': test_kelbo.item()}, step=count)
    print_(f'Iteration {count} \t Test KELBO: {test_kelbo.item()}')


def produce_samples_step():
    a.eval()
    wandb.log(
        {'samples': wandb.Image(
            plotting.fig2pil(
                plotting.plot_scatter_plate_carree(
                    gsde.sample(utils.sample_spherical_uniform(n=1024).to(device),
                                args.T0,
                                args.num_steps).detach().cpu().numpy())))},
        step=count)
    a.train()
    test_samples = testset.data[:1024]
    wandb.log(
        {'density plot': wandb.Image(
            plotting.fig2pil(get_fig_density(ema_a, sde, prior, test_samples=test_samples, title=args.dataset)),
        )}, step=count)


if __name__ == "__main__":
    _folder_name_keys = ['dataset', 'batch_size', 'lr', 'num_iterations', 'sch', 'warmup_iters',
                        'T0', 'emb_size', 'hidden_layers', 'imp', 'seed', 'ema_decay', 'inference_num_steps']
    args = get_args()
    folder_tag = 'riemannian-diffusion'
    folder_name = '-'.join([str(getattr(args, k)) for k in _folder_name_keys])
    create(args.saveroot, folder_tag, args.expname, folder_name)
    folder_path = os.path.join(args.saveroot, folder_tag, args.expname, folder_name)
    print_ = lambda s: logging(s, folder_path)
    print_(f'folder path: {folder_path}')
    print_(str(args))
    with open(os.path.join(folder_path, 'args.txt'), 'w') as out:
        out.write(json.dumps(args.__dict__, indent=4))

    wandb.init(project="riemannian_diffusion", name=f'{args.expname}-{folder_name}', config=args.__dict__)
    args = wandb.config

    # ################### autoencoder ###################
    data_dir = 'dataset'

    trainset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
    testset  = torchvision.datasets.MNIST(data_dir, train=False, download=True)

    train_transform = transforms.Compose([transforms.ToTensor()])

    test_transform = transforms.Compose([transforms.ToTensor()])

    trainset.transform = train_transform
    testset.transform = test_transform

    m=len(trainset)

    train_data, val_data = random_split(trainset, [int(m-m*0.2), int(m*0.2)])
    batch_size=256

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=True)

    d = 3
    # encoder = Encoder(encoded_space_dim=d,fc2_input_dim=128)
    # decoder = Decoder(encoded_space_dim=d,fc2_input_dim=128)
    # train_ae(encoder, decoder, 20)
    # encoder.eval()
    # decoder.eval()

    encoder = Encoder(encoded_space_dim=d,fc2_input_dim=128)
    decoder = Decoder(encoded_space_dim=d,fc2_input_dim=128)
    encoder.load_state_dict(torch.load("ae_encoder.ckpt"))
    decoder.load_state_dict(torch.load("ae_decoder.ckpt"))
    decoder.eval()

    a = ConcatMLP(3, 1, args.emb_size, args.hidden_layers)
    ema_a = ConcatMLP(3, 1, args.emb_size, args.hidden_layers)
    ema_a = ExponentialMovingAverage(ema_module=ema_a, decay=args.ema_decay).init(target_module=a)  # TODO add decay to args
    print_(a)
    cuda = torch.cuda.is_available()

    if cuda:
        print_('cuda available')
        a = a.cuda()
        ema_a = ema_a.cuda()
        device = 'cuda'
    else:
        print_('cuda unavailable')
        device = 'cpu'

    sde = AmbientSphericalBrownianMotion(args.T0)
    gsde = AmbientSphericalGenerative(utils.get_drift(a, args.T0, Sphere, sde))
    opt = torch.optim.Adam(a.parameters(), lr=args.lr)
    prior = utils.LOG_SPHERICAL_UNIFORM

    if args.sch:
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.num_iterations - args.warmup_iters)
    else:
        sch = None

    if args.imp and args.mode == 'train':
        print_('Initializing importance sampler')
        imp_sampler = ImportanceSampler(args.T0, ndim=10, stratified=True)
        if cuda:
            imp_sampler = imp_sampler.cuda()
            imp_sampler.is_cuda = True

        for _ in range(100):
            x = iter(trainloader).next()
            if cuda:
                x = x.cuda()
            imp_sampler.step(get_estimand(x), x.size(0))

        wandb.log({'q(t)': wandb.Image(plotting.fig2pil(
            plotting.plot_hist(imp_sampler.sample(1000)[0].view(-1, 1).cpu().data.numpy(),
                            xlabel='t', ylabel='q(t)', density=True, bins=20)))}, step=0)
        imp_sampler.eval()
    else:
        imp_sampler = None
        get_estimand = None

    if os.path.exists(os.path.join(folder_path, 'checkpoint.pt')):
        print_('loading checkpoint')
        a, opt, sch, not_finished, count, imp_sampler, ema_a = torch.load(os.path.join(folder_path, 'checkpoint.pt'), map_location=torch.device(device))
        gsde = AmbientSphericalGenerative(utils.get_drift(a, args.T0, Sphere, sde))
    else:
        not_finished = True
        count = 0

        a.eval()
        wandb.log(
            {'samples': wandb.Image(
                plotting.fig2pil(
                    plotting.plot_scatter_plate_carree(gsde.sample(utils.sample_spherical_uniform(n=1024).to(device),
                                                                args.T0, args.num_steps).detach().cpu().numpy())))},
            step=count)
        a.train()
    
    if args.mode == 'train':
        while not_finished:
            # for x in trainloader:
            for image_batch, _ in train_loader:
                image_batch = image_batch.to(device)
                x = encoder(image_batch).detach()
                x = Sphere.proj2manifold(x)

                if count <= args.warmup_iters:
                    update_lr(opt, count, args)

                if cuda:
                    x = x.cuda()
                if args.imp:
                    loss = imp_sampler.estimate(get_estimand(x), x.size(0)).mean() - utils.LOG_SPHERICAL_UNIFORM
                else:
                    t = utils.stratified_uniform(x.size(0), args.T0)
                    if cuda:
                        t = t.cuda()
                    y = sde.sample(x, t, args.inference_num_steps).requires_grad_(True)
                    # print("y: ", y.size())

                    loss = - utils.elbo_sphere(y, t, a, Sphere.proj2manifold, Sphere.proj2tangent, args.T0).mean()
                    # TODO method

                opt.zero_grad()
                loss.backward()
                opt.step()

                ema_a = ExponentialMovingAverage(ema_module=ema_a, decay=args.ema_decay).step(a)

                if args.sch and count >= args.warmup_iters:
                    sch.step()

                count += 1
                wandb.log({'loss': loss.item()}, step=count)
                if count % args.print_every == 0 or count == 1:
                    print_(f'Iteration {count} \tloss {loss.item()}')

                if count % 500 == 0:
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
                        # if i == n//2:
                        #     ax.set_title('Reconstructed images')
                        plt.subplots_adjust(wspace=0, hspace=0)
                    print("SAVING IMAGES")
                    plt.savefig("sphere_ae.png")
                    # plt.show()

                # if count % args.sample_every == 0:
                #     produce_samples_step()

                # if count % args.checkpoint_every == 0:
                #     torch.save([a, opt, sch, not_finished, count, imp_sampler, ema_a],
                #             os.path.join(folder_path, 'checkpoint.pt'))

                # if count % args.evaluate_every == 0:
                #     evaluate_step()

                # if count >= args.num_iterations:
                #     not_finished = False
                #     print_('Finished training')
                #     torch.save([a, opt, sch, not_finished, count, imp_sampler, ema_a],
                #             os.path.join(folder_path, 'checkpoint.pt'))
                #     break

                # if args.imp >= 1 and count % args.imp == 1:
                #     imp_sampler.train()
                #     for _ in range(10):
                #         x = iter(trainloader).next()
                #         if cuda:
                #             x = x.cuda()
                #         imp_sampler.step(get_estimand(x), x.size(0))
                #     wandb.log({'q(t)': wandb.Image(plotting.fig2pil(
                #         plotting.plot_hist(imp_sampler.sample(1000)[0].view(-1, 1).cpu().data.numpy(),
                #                         xlabel='t', ylabel='q(t)', density=True, bins=20)))}, step=count)
                #     imp_sampler.eval()

        val_kelbo = evaluate(valloader, a_module=ema_a, adaptive=True)
        print_(f'Final Validation KELBO: {val_kelbo.item()}')
        test_kelbo = evaluate(testloader, a_module=ema_a, adaptive=True)
        print_(f'Final Test KELBO: {test_kelbo.item()}')


    if args.mode == 'report':
        produce_samples_step()
        evaluate_step()

        # wandb.log({'test data': wandb.Image(plotting.fig2pil(plotting.plot_scatter_plate_carree(testset.data[:1024])))},
        #           step=count)