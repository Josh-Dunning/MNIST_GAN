# Josh Dunning
# Includes code from "https://pytorch.org/tutorials/beginner/pytorch_with_examples.html"
# For Python 3.X

import torch
import sys
import signal
import math
import time
sys.path.append("Sim")
from TestGenes import GenerateGenePairs

def signal_handler(sig, frame):
    print ("\n")
    prompt_save(model, model_save_path)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

raw_net_diagram = """
 _     _     _
|_|   |_|   |_|
 | \\ / | \\ / |
 |  \\  |  \\  |
 \033[4m|\033[0m / \\ \033[4m|\033[0m / \\ \033[4m|\033[0m
|_|   |_|   |_|
 | \\ / | \\ / |
 |  \\  |  \\  |
 \033[4m|\033[0m / \\ \033[4m|\033[0m / \\ \033[4m|\033[0m
|\033[4m#\033[0m|   |\033[4m#\033[0m|   |\033[4m#\033[0m|

"""

################################
###### HELPER FUNCTIONS ########

def get_device(dtype):
    """
    Check if a valid GPU with drivers is available,
    otherwise just run on the CPU.
    """
    try:
        device = torch.device("cuda:0")
        test_gpu = torch.randn(0, 0, device=device, dtype=dtype)
        print ("  Running on GPU")
    except AssertionError:
        device = torch.device("cpu")
        print ("  Running on CPU")
    print ("  ==============")
    return device

def print_batch_start(idx, size):
    print("{:<3} ===============================".format(""))
    print("{:<3} || Starting batch {:<3} of {:<3} ||".format("", idx + 1, size))
    print("{:<3} ===============================".format(""))

def print_training_status(t, itr, loss, avg_loss, diagram):
    """
    Print the current state of the training.
    """
    percent_done = ((t * 1.0) / (itr * 1.0 / 100))
    if not percent_done % 10:
        buf = ''.join(["###" if i < (percent_done / 10) else "___" for i in range(10)])
        progress_str = "   " + str(buf) + " " + str(int(percent_done)) + "%"
        sys.stdout.write('\r{:<1} {:>7} {:<1} {:<15} {:<14} {} \n {}'.format("", t, "", round(loss, 5), round(avg_loss, 5), next(diagram), progress_str))
        sys.stdout.flush()

def train(x, y, model, loss_fn, optimizer, lr, itr, n):
    """
    Train the network.
    """
    print ('{:<1} {:>7} {:<1} {:<13} {:<1} {:<15} {:<5} {:<5} {}'.format("", "Itr", "", "Loss", "", "Avg Loss", "|", "|", "|"))
    print ('{:<1} {:>7} {:<1} {:<13} {:<1} {:<15} {:<5} {:<5} {}'.format("", "---", "", "----", "", "--------", "V", "V", "V"))
    net_diagram = iter(raw_net_diagram.splitlines())
    next(net_diagram)
    for t in range(itr + 1):
        # Forward Propagation
        y_pred = model(x)

        # Find loss
        loss = loss_fn(y_pred, y)
        print_training_status(t, itr, loss.item(), (loss.item() / n), net_diagram)

        # Zero gradients then back propagate
        optimizer.zero_grad()
        loss.backward()

        # Update weights according to gradients
        optimizer.step()

    print ("\n")

def test(x, y, model, n):
    """
    Test the network.
    """
    # Forward pass: compute predicted y
    y_pred = model(x)

    # Compute and print loss
    loss = loss_fn(y_pred, y)
    
    print "   Average Test Loss:", round((loss.item() / n), 5), "\n"

def yes_or_no(question):
    while "the answer is invalid":
        reply = str(raw_input(question+' (y/n): ')).lower().strip()
        if reply[0] == 'y':
            return True
        if reply[0] == 'n':
            return False

def prompt_save(model, path):
    if yes_or_no("Save model?"):
        torch.save(model.state_dict(), path)
        print "Model saved."
    else:
        print "Model discarded."
    print ("")

#### END HELPER FUNCTIONS ######
################################

print("")

dtype = torch.float
device = get_device(dtype)

model_save_path = "./saves/256_only_winner"
load_model = False
load_test_data = False

# B is number of training batches, N is batch size, T_test is test batch size
B, N, N_test = 1, 64, 512
# D_in is input dimension, H is hidden dimensions, D_out is output dimension
D_in, H, D_out = 400, [256], 21
learning_rate = 1e-7
iterations = 500

# Create random gene codes for training and test data
start_data_time = time.time()
random_gene_codes, pacwar_results = GenerateGenePairs(B, N, "training")
one_hot_gene_codes = torch.flatten(torch.nn.functional.one_hot(random_gene_codes), start_dim=2, end_dim=3)

x_train = one_hot_gene_codes.type(torch.FloatTensor)
y_train = pacwar_results.type(torch.FloatTensor)

if load_test_data:
    x_test = torch.load('x_test_tensor.pt')
    y_test = torch.load('y_test_tensor.pt')

else:
    random_gene_codes, pacwar_results = GenerateGenePairs(1, N_test, "test")
    one_hot_gene_codes = torch.flatten(torch.nn.functional.one_hot(random_gene_codes), start_dim=2, end_dim=3)

    x_test = one_hot_gene_codes.type(torch.FloatTensor)
    y_test = pacwar_results.type(torch.FloatTensor)

    torch.save(x_test, 'x_test_tensor.pt')
    torch.save(y_test, 'y_test_tensor.pt')

# Setup model
layers = []
for i, layer in enumerate([D_in] + H):
    layers.append(torch.nn.Linear(layer, H[i] if i < len(H) else D_out))
    layers.append(torch.nn.ReLU())
model = torch.nn.Sequential(*(layers[:-1]))

# Load model if resuming training
if load_model:
    model.load_state_dict(torch.load(model_save_path))

# Define loss function and optimizer
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train
print ("")
start_training_time = time.time()
for batch_idx in range(B):
    print_batch_start(batch_idx, B)
    train(x_train[batch_idx], y_train[batch_idx], model, loss_fn, optimizer, learning_rate, iterations, N)

# Test
test(x_test, y_test, model, N_test)

print "Finished at", time.time()
print "Total time:", ((time.time() - start_data_time) / 3600)
print "Data creation time:", ((start_training_time - start_data_time) / 3600)
print "Training time:", ((time.time() - start_training_time) / 3600)

# Save model
prompt_save(model, model_save_path)










