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

# matplotlib.use('Agg')
import numpy as np
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
from helpers import logging, create
from manifold import Tori
from models import ConcatMLP
from plotting import plot_theta_of_ambient
from sdes import AmbientToriGenerative, AmbientToriBrownianMotion
from utils.utils import tori_theta_to_ambient, sample_tori_uniform, exact_logp
from tqdm import tqdm

from ae import train_ae, Encoder, Decoder

_folder_name_keys = ['dataset', 'batch_size', 'lr', 'num_iterations', 'T0', 'emb_size', 'hidden_layers', 'imp',
                     'bm_sampling_method', 'bm_sampling_steps', 'seed']


def get_args():
    parser = argparse.ArgumentParser()

    # i/o
    parser.add_argument('--dataset', type=str, choices=['General', 'Glycine', 'Proline', 'Pre-Pro', 'RNA'],
                        default='General')
    parser.add_argument('--dataroot', type=str, default='data')
    parser.add_argument('--saveroot', type=str, default='saved')
    parser.add_argument('--expname', type=str, default='tori_ambient')
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--sample_every', type=int, default=500)
    parser.add_argument('--checkpoint_every', type=int, default=500)
    parser.add_argument('--evaluate_every', type=int, default=5000)
    parser.add_argument('--evaluation_K', type=int, default=10)
    parser.add_argument('--num_steps', type=int, default=100,
                        help='number of integration steps for sampling')
    parser.add_argument('--evaluation_num_steps', type=int, default=100,
                        help='number of integration steps for evaluation')
    parser.add_argument('--seed', type=int, default=12345, help='random seed')

    # hparam
    parser.add_argument('--T0', type=float, default=2.0,
                        help='integration time')
    parser.add_argument('--emb_size', type=int, default=256,
                        help='embedding size for hidden layers')
    parser.add_argument('--hidden_layers', type=int, default=2,
                        help='num of hidden layers')
    parser.add_argument('--imp', type=int, default=0,
                        help='importance sampling for time index')

    # optimization
    parser.add_argument('--div', type=str, choices=['deterministic'], )
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--num_iterations', type=int, default=10000, help='number of training iterations')
    parser.add_argument('--sch', type=int, default=0, choices=[0, 1],
                        help='using cosine lr scheduler')
    parser.add_argument('--bm_sampling_method', type=str, default='integration',
                        choices=['integration', 'directsampling'],
                        help='brownian motion sampling method')
    parser.add_argument('--bm_sampling_steps', type=int, default=100, help='number of integration steps when sampling '
                                                                           'brownian motion with integration.'
                                                                           ' It has no effect when using direct sampling')
    parser.add_argument('--observe_loss_with_bm_direct_sampling', type=int, default=True,
                        help='track and log loss with bm direct sampling')

    # mode selection
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'report'])

    return parser.parse_args()


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

train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
test_dataset  = torchvision.datasets.MNIST(data_dir, train=False, download=True)

train_transform = transforms.Compose([transforms.ToTensor()])

test_transform = transforms.Compose([transforms.ToTensor()])

train_dataset.transform = train_transform
test_dataset.transform = test_transform

m=len(train_dataset)

train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])
batch_size=256

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# load data
# if args.dataset in ['General', 'Glycine', 'Proline', 'Pre-Pro']:
#     trainset = datasets.Top500(amino=args.dataset, split='train')
#     trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

#     valset = datasets.Top500(amino=args.dataset, split='valid')
#     valloader = DataLoader(valset, batch_size=args.val_batch_size, shuffle=False, num_workers=0)

#     testset = datasets.Top500(amino=args.dataset, split='test')
#     testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)
# elif args.dataset in ['RNA']:
#     trainset = datasets.RNA(split='train', seed=args.seed)
#     trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

#     valset = datasets.RNA(split='valid', seed=args.seed)
#     valloader = DataLoader(valset, batch_size=args.val_batch_size, shuffle=False, num_workers=0)

#     testset = datasets.RNA(split='test', seed=args.seed)
#     testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)
# else:
#     raise Exception(f'Dataset {args.dataset} not implemented')



def transform(theta_):
    return (theta_ - 360. * (theta_ > 180)) / 180. * np.pi

d = 4
encoder = Encoder(encoded_space_dim=d,fc2_input_dim=128)
decoder = Decoder(encoded_space_dim=d,fc2_input_dim=128)
train_ae(encoder, decoder, 20, d=d)
encoder.eval()
decoder.eval()

# tori_dim = next(iter(train_loader))[0].shape[1]
# print(next(iter(train_loader))[0].shape)
# print("TORI")
# print(tori_dim)
tori_dim = 2
tori = Tori(tori_dim)
dimx = tori_dim * 2  # ambient dim is twice the manifold intrinsic dim
a = ConcatMLP(dimx=dimx, dimt=1, emb_size=args.emb_size, num_hidden_layers=args.hidden_layers)
print_(a)
cuda = torch.cuda.is_available()
if cuda:
    print_('cuda available')
    a = a.cuda()
    device = 'cuda'
else:
    print_('cuda unavailable')
    device = 'cpu'

# theta = transform(next(iter(train_loader)))
# x = tori_theta_to_ambient(theta)
sde = AmbientToriBrownianMotion(args.T0, args.bm_sampling_method, tori_dim)
gsde = AmbientToriGenerative(drift_func=utils.get_drift(a, args.T0, Tori, sde), tori_dim=tori_dim)
prior = tori_dim * utils.LOG_CIRCLE_UNIFORM
opt = torch.optim.Adam(a.parameters(), lr=args.lr)


if args.sch:
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.num_iterations)
else:
    sch = None

print_(f'bm_sampling_method: {args.bm_sampling_method}.')
if args.bm_sampling_method == 'directsampling':
    print_(f'bm_sampling_steps: {args.bm_sampling_steps} is ignored')

if args.imp and args.mode=='train':
    print_('Initializing importance sampler')
    imp_sampler = ImportanceSampler(args.T0, ndim=10, stratified=True)
    if cuda:
        imp_sampler = imp_sampler.cuda()
        imp_sampler.is_cuda = True


    def get_estimand(x_, force_direct_sampling=False):
        def estimand(t_, prior=None):
            assert prior is None  # just for compatibility with the estimand for other manifolds so imp_sampler can work with them
            if force_direct_sampling:
                y_ = sde.sample_by_direct_sampling(x_, t_).requires_grad_(True)
            else:
                y_ = sde.sample(x_, t_, steps=args.bm_sampling_steps).requires_grad_(True)
            loss_ = -utils.elbo_tori(y_, t_, a, tori.proj2manifold, tori.proj2tangent, sde.T)
            return loss_

        return estimand


    for _ in range(100):
        x = tori_theta_to_ambient(transform(iter(train_loader).next()))
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

# load model if it exists
if os.path.exists(os.path.join(folder_path, 'checkpoint.pt')):
    print_('loading checkpoint')
    a, opt, sch, not_finished, count, imp_sampler = torch.load(os.path.join(folder_path, 'checkpoint.pt'), map_location=torch.device(device))
    gsde = AmbientToriGenerative(drift_func=utils.get_drift(a, args.T0, Tori, sde), tori_dim=tori_dim)
else:
    not_finished = True
    count = 0

    a.eval()
    tori_dim = next(iter(train_loader)).shape[1]  # dataset is expressed in theta(intrisic coordinates)
    wandb.log(
        {'samples': wandb.Image(
            plotting.fig2pil(
                plot_theta_of_ambient(
                    gsde.sample(
                        sample_tori_uniform(tori_dim=tori_dim, n=1024).to(device), args.T0,
                        args.num_steps)
                )))},
        step=count)
    a.train()

    wandb.log({'test data': wandb.Image(
        plotting.fig2pil(
            plot_theta_of_ambient(
                tori_theta_to_ambient(transform(torch.tensor(testset.data[:1024])))
            )))}, step=count)


def evaluate(dataloader, adaptive=False):
    a.eval()
    kelbos = []
    for theta_val in tqdm(dataloader):
        x_val = tori_theta_to_ambient(transform(theta_val))
        if cuda:
            x_val = x_val.cuda()
        if adaptive:
            kelbo = utils.adaptivekelbo_tori(sde=sde,
                                             y0=x_val,
                                             a_func=a,
                                             proj2manifold=tori.proj2manifold,
                                             proj2tangent=tori.proj2tangent,
                                             T=sde.T,
                                             K=args.evaluation_K,
                                             ds=sde.T / args.evaluation_num_steps,
                                             ds_min=1e-5,
                                             rtol=1e-5,
                                             atol=1e-5)
        else:
            kelbo = utils.kelbo_tori(sde=sde,
                                     y0=x_val,
                                     a_func=a,
                                     proj2manifold=tori.proj2manifold,
                                     proj2tangent=tori.proj2tangent,
                                     T=sde.T,
                                     K=args.evaluation_K,
                                     steps=args.evaluation_num_steps)
        kelbos.append(kelbo.detach())
    kelbo = torch.cat(kelbos).mean()
    a.train()
    return kelbo


def evaluate_exact_logp(dataloader, a_func, bm_sde, prior):
    logps = []
    counter = 0
    for theta_ in tqdm(dataloader):
        counter += 1
        x_ = tori_theta_to_ambient(transform(theta_))
        if cuda:
            x_ = x_.cuda()
        logp = exact_logp(x_ambient=x_, bm_sde=bm_sde, a_func=a_func, prior=prior, steps=args.evaluation_num_steps, method='closest')
        logps.append(logp.detach())
    logps = torch.cat(logps, axis=0)
    return logps


def get_fig_density(a_func, bm_sde, prior_, test_samples=None, title=None):
    """
    :test_samples: samples from the test set to overlay on the density
    """
    def func(theta_, steps=100):
        return exact_logp(tori_theta_to_ambient(transform(theta_)), bm_sde, a_func, prior_, steps, method='qr')

    fig_colormesh, fig_contour = plotting.plot_density_on_theta(func, device=device, test_samples=test_samples, title=title)
    return fig_colormesh, fig_contour


def evaluate_step():
    train_exact_logp = evaluate_exact_logp(dataloader=train_loader, a_func=a, bm_sde=sde, prior=prior)
    train_exact_logp = train_exact_logp.mean()
    wandb.log({'train_exact_logp': train_exact_logp.item()}, step=count)
    print_(f'Train exact logp: {train_exact_logp.item()}')
    valid_exact_logp = evaluate_exact_logp(dataloader=valloader, a_func=a, bm_sde=sde, prior=prior)
    valid_exact_logp = valid_exact_logp.mean()
    wandb.log({'valid_exact_logp': valid_exact_logp.item()}, step=count)
    print_(f'Valid exact logp: {valid_exact_logp.item()}')
    test_exact_logp = evaluate_exact_logp(dataloader=testloader, a_func=a, bm_sde=sde, prior=prior)
    test_exact_logp = test_exact_logp.mean()
    wandb.log({'test_exact_logp': test_exact_logp.item()}, step=count)
    print_(f'Test exact logp: {test_exact_logp.item()}')
    val_kelbo = evaluate(valloader)
    wandb.log({'val_kelbo': val_kelbo.item()}, step=count)
    print_(f'Iteration {count} \t Validation KELBO: {val_kelbo.item()}')
    test_kelbo = evaluate(testloader)  # TODO uncomment after hyperparameter tuning with validation set
    wandb.log({'test_kelbo': test_kelbo.item()}, step=count)
    print_(f'Iteration {count} \t Test KELBO: {test_kelbo.item()}')


def produce_samples_step():
    a.eval()
    wandb.log(
        {'samples': wandb.Image(
            plotting.fig2pil(
                plot_theta_of_ambient(
                    gsde.sample(
                        sample_tori_uniform(tori_dim, n=1024).to(device), args.T0,
                        args.num_steps))),
        )}, step=count)
    a.train()
    test_samples = transform(torch.tensor(testset.data[:1024]))*180./np.pi
    fig_colormesh, fig_contour = get_fig_density(a, sde, prior, test_samples=test_samples, title=args.dataset)
    wandb.log(
        {'density plot colormesh': wandb.Image(
            plotting.fig2pil(fig_colormesh)
        )}, step=count)
    wandb.log(
        {'density plot contour': wandb.Image(
            plotting.fig2pil(fig_contour)
        )}, step=count)



if __name__ == "__main__":
    if args.mode == 'train':
        while not_finished:
            # for theta in trainloader:
            for image_batch, _ in train_loader:
                image_batch = image_batch.to(device)
                x = encoder(image_batch).detach()
                # transfer to ambient space
                # x = tori_theta_to_ambient(transform(theta))
                if cuda:
                    x = x.cuda()
                if args.imp:
                    loss = imp_sampler.estimate(get_estimand(x, force_direct_sampling=False), x.size(0)).mean()
                else:
                    t = utils.stratified_uniform(x.size(0), args.T0)
                    if cuda:
                        t = t.cuda()
                    y = sde.sample(x, t, steps=args.bm_sampling_steps).requires_grad_(True)
                    loss = -utils.elbo_tori(y, t, a, Tori.proj2manifold, Tori.proj2tangent, args.T0).mean()
        
                opt.zero_grad()
                loss.backward()
                opt.step()
                if args.sch:
                    sch.step()
        
                count += 1
                wandb.log({'loss': loss.item()}, step=count)
        
                if args.observe_loss_with_bm_direct_sampling:
                    if args.imp:
                        direct_loss = imp_sampler.estimate(get_estimand(x, force_direct_sampling=True), x.size(0)).mean()
                    else:
                        t = utils.stratified_uniform(x.size(0), args.T0)
                        if cuda:
                            t = t.cuda()
                        y = sde.sample_by_direct_sampling(x, t).requires_grad_(True)
                        direct_loss = -utils.elbo_tori(y, t, a, Tori.proj2manifold, Tori.proj2tangent, args.T0).mean()
                    wandb.log({'direct_loss': direct_loss.item()}, step=count)
                    del direct_loss
        
                if count % args.print_every == 0 or count == 1:
                    print_(f'Iteration {count} \tloss {loss.item()}')
                
                if count % 500 == 0:
                    # Bonnie's samples :)
                    n = 10
                    # print("TORI_DIM")
                    samples = gsde.sample(utils.sample_tori_uniform(tori_dim=tori_dim, n=n).to(device),
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
                        plt.savefig("sphere_ae.png")
                    print("SAVING IMAGES")
                    plt.savefig("tori_ae.png")
                    # plt.show()
        
                # if count % args.sample_every == 0:
                #     produce_samples_step()
        
                # if count % args.checkpoint_every == 0:
                #     torch.save([a, opt, sch, not_finished, count, imp_sampler], os.path.join(folder_path, 'checkpoint.pt'))
        
                # if count % args.evaluate_every == 0:
                #     evaluate_step()
        
                # if count >= args.num_iterations:
                #     not_finished = False
                #     print_('Finished training')
                #     torch.save([a, opt, sch, not_finished, count, imp_sampler], os.path.join(folder_path, 'checkpoint.pt'))
                #     break
        
                # if args.imp >= 1 and count % args.imp == 1:
                #     imp_sampler.train()
                #     for _ in range(10):
                #         x = tori_theta_to_ambient(transform(iter(train_loader).next()))
                #         if cuda:
                #             x = x.cuda()
                #         imp_sampler.step(get_estimand(x), x.size(0))
                #     wandb.log({'q(t)': wandb.Image(plotting.fig2pil(
                #         plotting.plot_hist(imp_sampler.sample(1000)[0].view(-1, 1).cpu().data.numpy(),
                #                            xlabel='t', ylabel='q(t)', density=True, bins=20)))}, step=count)
                #     imp_sampler.eval()
                    
        val_kelbo = evaluate(valloader, adaptive=True)
        print_(f'Final Validation KELBO: {val_kelbo.item()}')
        test_kelbo = evaluate(testloader, adaptive=True)
        print_(f'Final Test KELBO: {test_kelbo.item()}')

    # generate report if in report mode
    if args.mode == 'report':
        produce_samples_step()
        evaluate_step()
        
        
