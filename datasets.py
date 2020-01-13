# Josh Dunning
# For Python 3.X

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
import math

class OneClassSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, mask, data_source):
        self.mask = mask
        self.data_source = data_source

    def __iter__(self):
        return iter([i.item() for i in torch.nonzero(self.mask)])

    def __len__(self):
        return len(self.data_source)

def loadData(target_num):
	#data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((32, 32)), transforms.ToTensor()])
	#label_transform = transforms.Compose([])
	data_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
	label_transform = lambda lbl: random.uniform(0.8, 0.99) if lbl == target_num else 0

	trainset = datasets.MNIST(root='./data', train=True, download=False, transform=data_transform, target_transform=label_transform)
	testset = datasets.MNIST(root='./data', train=False, download=False, transform=data_transform, target_transform=label_transform)

	train_mask = torch.tensor([math.ceil(trainset[i][1]) for i in range(len(trainset))])
	test_mask = torch.tensor([math.ceil(testset[i][1]) for i in range(len(testset))])

	train_sampler = OneClassSampler(train_mask, trainset)
	test_sampler = OneClassSampler(test_mask, testset)

	return (trainset, train_sampler, testset, test_sampler)
