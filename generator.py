# Josh Dunning
# For Python 3.X
# Adapted from https://github.com/activatedgeek/LeNet-5/blob/master/lenet.py

import torch.nn as nn
from collections import OrderedDict

class LeNet5Rev(nn.Module):

	def __init__(self):
		super(LeNet5Rev, self).__init__()

		self.linear = nn.Sequential(OrderedDict([
			('l1', nn.Linear(100, 4096)),
			('n1', nn.BatchNorm1d(4096)),
			('relu1', nn.ReLU()),
		]))

		self.convnet = nn.Sequential(OrderedDict([
			('c2', nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)),
			('n2', nn.BatchNorm2d(64)),
			('c3', nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)),
			('relu3', nn.Tanh()),
		]))

	def forward(self, img):
		#print ("G1:", img.shape)
		output = self.linear(img)
		#print ("G2:", output.shape)
		output = output.view(output.size(0), 64, 8, 8)
		#print ("G3:", output.shape)
		output = self.convnet(output)
		#print ("G4:", output.shape)
		return output

def buildGenerator():
	return LeNet5Rev()