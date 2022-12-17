import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from ae import train_ae, Encoder, Decoder

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
encoder = Encoder(encoded_space_dim=d,fc2_input_dim=128)
decoder = Decoder(encoded_space_dim=d,fc2_input_dim=128)
train_ae(encoder, decoder, 30)
encoder.eval()
decoder.eval()
torch.save(encoder.state_dict(), "ae_encoder.ckpt")
torch.save(decoder.state_dict(), "ae_decoder.ckpt")