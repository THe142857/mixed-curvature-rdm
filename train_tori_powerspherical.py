from torch.utils.data import DataLoader

import plotting
from data import datasets
from utils.von_mises_fisher import *
from utils.power_spherical import *
import argparse
from einops import rearrange
from manifold import Circle, Manifold
import torch
import matplotlib.pyplot as plt
import wandb

_folder_name_keys = ['dataset', 'batch_size', 'lr', 'num_iterations', 'num_mixture','num_iterations', 'seed']


def get_args():
    parser = argparse.ArgumentParser()

    # i/o
    parser.add_argument('--dataset', type=str, choices=['General', 'Glycine', 'Proline', 'Pre-Pro', 'RNA'],
                        default='General')
    parser.add_argument('--expname', type=str, default='train_tori_powerspherical')
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--sample_every', type=int, default=500)
    parser.add_argument('--evaluate_every', type=int, default=100)

    # hparam
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)

    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--num_mixture', type=int, default=32)
    parser.add_argument('--num_iterations', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=12345, help='random seed just for RNA dataset, does not affect torch or other datasets')

    return parser.parse_args()


# noinspection PyShadowingNames
def to_ambient(theta):
    """
    x: (batch_size, dim)
    returns: (batch_size, dim*2) embed n dimensional tori embedded in a dim*2 dimensional ambient space
    """
    b = theta.shape[0]
    theta = rearrange(theta, 'b d -> (b d) 1')
    x = Circle.invphi(theta)
    x = rearrange(x, '(b d) two-> b (d two)', b=b)
    return x


# noinspection PyShadowingNames
def to_theta(x):
    """
    x: (batch_size, dim*2)
    returns: (batch_size, dim) theta
    """
    b = x.shape[0]
    d = x.shape[1] // 2
    x = rearrange(x, 'b (d two) -> (b d) two', d=d)
    theta = Circle.phi(x)
    theta = rearrange(theta, '(b d)-> b d', b=b)
    return theta


# noinspection PyShadowingNames
def transform(theta):
    return (theta - 360. * (theta > 180)) / 180. * np.pi


def evaluate(dataloader):
    global device
    nlls = []
    for theta_val in dataloader:
        theta_val = theta_val.to(device)
        x_val = to_ambient(transform(theta_val))
        with torch.no_grad():
            nll = -v.log_prob(x_val).detach()
        nlls.append(nll)
    return torch.cat(nlls).mean()


# noinspection PyShadowingNames
def scatter_plot(x, y):
    fig = plt.figure(figsize=(4, 4))

    plt.scatter(x, y, s=10, alpha=0.1)
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\psi$')
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.tight_layout()

    return fig


# noinspection PyShadowingNames
def plot_theta(theta):
    theta = theta * 180. / np.pi
    return scatter_plot(theta[:, 0], theta[:, 1])


# noinspection PyShadowingNames
def plot_theta_of_ambient(x):
    theta = to_theta(x)
    theta = theta.detach().cpu().numpy()
    return plot_theta(theta)

args = get_args()
folder_name = '-'.join([str(getattr(args, k)) for k in _folder_name_keys])
wandb.init(project="riemannian_diffusion", name=f'{args.expname}-{folder_name}', config=args.__dict__)
args = wandb.config

if args.dataset in ['General', 'Glycine', 'Proline', 'Pre-Pro']:
    trainset = datasets.Top500(amino=args.dataset, split='train')
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    valset = datasets.Top500(amino=args.dataset, split='valid')
    valloader = DataLoader(valset, batch_size=args.val_batch_size, shuffle=False, num_workers=0)

    testset = datasets.Top500(amino=args.dataset, split='test')
    testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)
elif args.dataset in ['RNA']:
    trainset = datasets.RNA(split='train', seed=args.seed)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    valset = datasets.RNA(split='valid', seed=args.seed)
    valloader = DataLoader(valset, batch_size=args.val_batch_size, shuffle=False, num_workers=0)

    testset = datasets.RNA(split='test', seed=args.seed)
    testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)
else:
    raise Exception(f'Dataset {args.dataset} not implemented')

v = MixtureProductPowerSpherical(n=trainset[0].shape[0], p=args.num_mixture)

cuda = torch.cuda.is_available()
if cuda:
    print('cuda available')
    v = v.cuda()
    device = 'cuda'
else:
    print('cuda unavailable')
    device = 'cpu'

opt = torch.optim.Adam(v.parameters(), lr=args.lr)

iteration_counter = 0

wandb.log({'test data': wandb.Image(
        plotting.fig2pil(
            plot_theta_of_ambient(
                to_ambient(transform(torch.tensor(testset.data[:1024])))
            )))}, step=0)

while iteration_counter < args.num_iterations:
    for theta in trainloader:
        theta = theta.to(device)
        iteration_counter += 1
        x = to_ambient(transform(theta))
        opt.zero_grad()
        loss = -v.log_prob(x).mean()
        loss.backward()
        opt.step()

        # logging
        print('step:', iteration_counter, 'loss', loss.item())
        wandb.log({'training loss(nll)': loss.item()}, step=iteration_counter)

        if iteration_counter % args.evaluate_every == 0:
            wandb.log({'validation loss(nll)': evaluate(valloader).item()}, step=iteration_counter)
            wandb.log({'test loss(nll)': evaluate(testloader).item()}, step=iteration_counter)

        if iteration_counter % args.sample_every == 0:
            wandb.log(
                {'samples': wandb.Image(
                    plotting.fig2pil(
                        plot_theta_of_ambient(
                            v.sample(1024)
                            )),
                )}, step=iteration_counter)


