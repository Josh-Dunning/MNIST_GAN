# Josh Dunning
# For Python 3.X

import torch
from generator import buildGenerator
from matplotlib import pyplot as plt

###################################
########### PARAMETERS ############

#generator_input_sampler = lambda m, n: torch.empty(m, n).normal_(mean=0.5,std=0.5)
generator_input_sampler = lambda m, n: torch.rand(m, n)
#generator_input_sampler = lambda m, n: torch.zeros(m, n)
generator_save_path = './saves/generator/mnist8_full'
generator_input_noise = 100
rows = 5
cols = 8

###################################


def main():
    generator = buildGenerator()
    generator.load_state_dict(torch.load(generator_save_path))

    fake_images = generator(generator_input_sampler(rows * cols, generator_input_noise)).detach()

    f, axarr = plt.subplots(rows, cols)

    for row in range(rows):
        for col in range(cols):
            axarr[row, col].imshow(fake_images[row * cols + col].reshape(32, 32), cmap='gray')
    
    plt.show()

if __name__ == '__main__':
    main()







