# Josh Dunning
# For Python 3.X
# Adapted from https://github.com/activatedgeek/LeNet-5/blob/master/lenet.py

import torch.nn as nn
from collections import OrderedDict

class LeNet5(nn.Module):
	def __init__(self):
		super(LeNet5, self).__init__()

		# o = [i + 2*p - k]/s + 1

		self.convnet = nn.Sequential(OrderedDict([
			('c1', nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)), # 64 x 16
			('n1', nn.BatchNorm2d(64)),
			('relu1', nn.LeakyReLU()),
			('d1', nn.Dropout(0.4)),
			('c2', nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)), # 64 x 8
			('n2', nn.BatchNorm2d(64)),
			('relu2', nn.LeakyReLU()),
			('d2', nn.Dropout(0.4)),
		]))

		self.linear = nn.Sequential(OrderedDict([
			('l3', nn.Linear(4096, 1)),
			('relu3', nn.LeakyReLU())
		]))

	def forward(self, img):
		#print ("D1:", img.shape)
		output = self.convnet(img)
		#print ("D2:", output.shape)
		output = output.view(output.size(0), -1)
		#print ("D3:", output.shape)
		output = self.linear(output)
		#print ("D4:", output.shape)
		return output

def buildDiscriminator():
	return LeNet5()