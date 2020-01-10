# Josh Dunning
# For Python 3.X
# Adapted from:
#   https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
#   https://github.com/activatedgeek/LeNet-5/blob/master/run.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import loadData
from discriminator import buildDiscriminator
from generator import buildGenerator

from matplotlib import pyplot as plt
import numpy as np
import sys
import signal
import random

# Global variables
discriminator, d_optim, trainset_batches, trainset_batches_it, generator, g_optim, testset, testset_batches, loaded = (None,)*9

##################################
######## HELPER FUNCTIONS ########

def yes_or_no(question):
    while "the answer is invalid":
        reply = str(raw_input(question+' (y/n): ')).lower().strip()
        if reply[0] == 'y':
            return True
        if reply[0] == 'n':
            return False

def save(save_objects):
    for model, save_path in save_objects:
        torch.save(model.state_dict(), save_path)
    print "Models saved."

def prompt_save(save_objects):
    if yes_or_no("Save models?"):
        save(save_objects)
    else:
        print "Models discarded."
    print ("")

def signal_handler(sig, frame):
    if loaded:
        print ("\n")
        prompt_save([(discriminator, discriminator_save_path), (generator, generator_save_path)])
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

##################################
######## MODEL PARAMETERS ########

# Overall parameters
epochs = 30
trainset_batch_size = 256
testset_batch_size = 1024
generator_input_noise = 100
generator_input_sampler = lambda m, n: torch.empty(m, n).normal_(mean=0.5,std=0.5)

# Discriminator parameters
load_discriminator_state = True
discriminator_save_path = './saves/discriminator/mnist8_full'
d_steps = 100
dlr = 1e-6  # (DLR) - Discriminator learning rate
d_loss_fn = nn.MSELoss()

# Generator parameters
load_generator_state = True
generator_save_path = './saves/generator/mnist8_full'
g_steps = 200
glr = 1e-5  # (GLR) - Generator learning rate
g_loss_fn = nn.MSELoss()

##################################
######### CORE FUNCTIONS #########

def getTrainsetBatch():
    global trainset_batches_it 

    try:
        (images, labels) = next(trainset_batches_it)
    except StopIteration:
        trainset_batches_it = iter(trainset_batches)
        (images, labels) = next(trainset_batches_it)
    return (images, labels.float())

def trainDiscriminator(e):
    discriminator.train()
    for i in range(d_steps):
        real_images, labels = getTrainsetBatch()

        d_optim.zero_grad()
        
        # Run on real images
        real_output = discriminator(real_images)
        real_loss = d_loss_fn(real_output, labels)

        # Run on fake images
        fake_images = generator(generator_input_sampler(trainset_batch_size, generator_input_noise)).detach()
        fake_output = discriminator(fake_images.view(trainset_batch_size, 1, 32, 32))
        fake_loss = d_loss_fn(fake_output, torch.add(torch.zeros([trainset_batch_size,]), random.uniform(0.0, 0.2)))

        if i % 10 == 0 or i + 1 == d_steps:
            print('Discriminator Train - Epoch %d, Batch: %d, Real Loss: %f, Fake Loss: %f' % (e + 1, i, real_loss.detach().cpu().item(), fake_loss.detach().cpu().item()))

        if i == 0 and not e % 2:
            f, axarr = plt.subplots(3, 2)
            axarr[0, 0].imshow(real_images[0].reshape(32, 32))
            axarr[0, 1].imshow(fake_images[0].reshape(32, 32))
            axarr[1, 0].imshow(real_images[1].reshape(32, 32))
            axarr[1, 1].imshow(fake_images[1].reshape(32, 32))
            axarr[2, 0].imshow(real_images[2].reshape(32, 32))
            axarr[2, 1].imshow(fake_images[2].reshape(32, 32))
            plt.show(block=False)
            plt.pause(1)
            plt.close()

        real_loss.backward()
        fake_loss.backward()
        d_optim.step()

def trainGenerator(e):
    generator.train()
    for i in range(g_steps):

        real_images, labels = getTrainsetBatch()

        g_optim.zero_grad()

        # Create fake images
        fake_images = generator(generator_input_sampler(trainset_batch_size, generator_input_noise))
        fake_output = discriminator(fake_images.view(trainset_batch_size, 1, 32, 32))
        fake_loss = g_loss_fn(fake_output, torch.add(torch.zeros([trainset_batch_size,]), random.uniform(0.8, 0.99)))

        if i % 10 == 0 or i + 1 == g_steps:
            print('Generator Train - Epoch %d, Batch: %d, Loss: %f' % (e + 1, i, fake_loss.detach().cpu().item()))

        fake_loss.backward()
        g_optim.step()

def train():
    for epoch in range(epochs):
        trainDiscriminator(epoch)
        trainGenerator(epoch)
        save([(discriminator, discriminator_save_path), (generator, generator_save_path)])

def test():
    discriminator.eval()
    generator.eval()
    d_total_correct = 0
    g_total_correct = 0
    avg_d_loss = 0.0
    avg_g_loss = 0.0
    for i, (images, labels) in enumerate(testset_batches):
        labels = labels.float()
        real_output = discriminator(images)
        avg_d_loss += d_loss_fn(real_output, labels).sum()
        real_pred = real_output.detach().max(1)[1]
        d_total_correct += real_pred.eq(labels.view_as(real_pred)).sum()

        fake_images = generator(generator_input_sampler(testset_batch_size, generator_input_noise))
        fake_output = discriminator(fake_images.view(testset_batch_size, 1, 32, 32))
        ones_labels = torch.ones([testset_batch_size,])
        avg_g_loss += g_loss_fn(fake_output, ones_labels)
        fake_pred = fake_output.detach().max(1)[1]
        g_total_correct += fake_pred.eq(ones_labels.view_as(fake_pred)).sum()

    avg_d_loss /= len(testset)
    print('Test Avg. Discriminator - Loss: %f, Accuracy: %f' % (avg_d_loss.detach().cpu().item(), float(d_total_correct) / len(testset)))
    print('Test Avg. Generator - Loss: %f, Accuracy: %f' % (avg_g_loss.detach().cpu().item(), float(g_total_correct) / len(testset)))

def main():
    global discriminator, d_optim, generator, g_optim, trainset_batches, trainset_batches_it, testset, testset_batches, loaded

    # Load data and setup models
    trainset, trainsampler, testset, testsampler = loadData()
    discriminator = buildDiscriminator()
    generator = buildGenerator()
    loaded = True

    # Restore models to previous state if desired
    if load_discriminator_state:
        discriminator.load_state_dict(torch.load(discriminator_save_path))
    if load_generator_state:
        generator.load_state_dict(torch.load(generator_save_path))

    # Prepare training and test data batches
    trainset_batches = DataLoader(trainset, batch_size=trainset_batch_size, shuffle=False, sampler=trainsampler, num_workers=8)
    trainset_batches_it = iter(trainset_batches)
    testset_batches = DataLoader(testset, batch_size=testset_batch_size, sampler=testsampler,  num_workers=8)

    # Setup optimizers
    d_optim = torch.optim.Adam(discriminator.parameters(), lr=dlr)
    g_optim = torch.optim.Adam(generator.parameters(), lr=glr)

    # Train then test models
    train()
    test()

    # Save model
    prompt_save([(discriminator, discriminator_save_path), (generator, generator_save_path)])

if __name__ == '__main__':
    main()







