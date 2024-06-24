"""
Training file for the models we implemented 
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils
from torch.utils.data import DataLoader
from einops import rearrange
import wandb

from model import BigramLanguageModel, MiniGPT
from dataset import TinyStoriesDataset
from config import BigramConfig, MiniGPTConfig


MODEL = "bigram"  # bigram or minigpt

if MODEL == "bigram":
    config = BigramConfig
    model = BigramLanguageModel(config)
elif MODEL == "minigpt":
    config = MiniGPTConfig
    model = MiniGPT(config)
else:
    raise ValueError("Invalid model name")


# Initialize wandb if you want to use it
# if config.to_log:
#     wandb.init(project="dl2_proj3")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


train_dataset = TinyStoriesDataset(
    config.path_to_data,
    mode="train",
    context_length=config.context_length,
)
eval_dataset = TinyStoriesDataset(
    config.path_to_data, mode="test", context_length=config.context_length
)

train_dataloader = DataLoader(
    train_dataset, batch_size=config.batch_size, pin_memory=True
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=config.batch_size, pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("number of trainable parameters: %.2fM" % (count_parameters(model) / 1e6,))


if not Path.exists(config.save_path):
    Path.mkdir(MiniGPTConfig.save_path, parents=True, exist_ok=True)


### ==================== START OF YOUR CODE ==================== ###
"""
You are required to implement the training loop for the model.

Please keep the following in mind:
- You will need to define an appropriate loss function for the model.
- You will need to define an optimizer for the model.
- You are required to log the loss (either on wandb or any other logger you prefer) every `config.log_interval` iterations.
- It is recommended that you save the model weights every `config.save_iterations` iterations you can also just save the model with the best training loss.

Please check the config file to see the different configurations you can set for the model.
NOTE : 
The MiniGPT config has params that you do not need to use, these were added to scale the model but are 
not a required part of the assignment. 
Feel free to experiment with the parameters and I would be happy to talk to you about them if interested :)
"""

import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Define loss function
criterion = nn.CrossEntropyLoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Initialize SummaryWriter for TensorBoard logging
writer = SummaryWriter()

# Initialize variables for tracking training and validation loss
train_losses = []
eval_losses = []

# Define the number of batches for training and validation
num_train_batches = 6000
num_val_batches = 6000

# Training loop
for epoch in range(1):
    model.train()
    train_loss_epoch = 0  # Initialize epoch training loss
    for i, batch in enumerate(train_dataloader, 0):
        if i >= num_train_batches:
            break
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.squeeze(dim=1).to(device)  # Move data to device
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Log training loss to TensorBoard
        writer.add_scalar('Training Loss', loss.item(), epoch * num_train_batches + i)
        train_losses.append(loss.item())  # Append training loss to list
        train_loss_epoch += loss.item()  # Accumulate epoch training loss

    train_loss_epoch /= num_train_batches  # Average training loss for the epoch

    # Validate the model
    model.eval()
    val_loss_epoch = 0  # Initialize epoch validation loss
    val_losses_epoch = []  # Initialize list for epoch validation losses
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader, 0):
            if i >= num_val_batches:
                break
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.squeeze(dim=1).to(device)  # Move data to device
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss_epoch += loss.item()  # Accumulate epoch validation loss
            val_losses_epoch.append(loss.item())  # Append validation loss for this batch to list
            # Debugging print statements
            print(f"Validation batch {i}: loss = {loss.item()}")
    
    val_loss_epoch /= num_val_batches  # Average validation loss for the epoch
    eval_losses.extend(val_losses_epoch)  # Append epoch validation losses to overall list
    # Debugging print statement
    print(f"Epoch {epoch}: Validation Loss = {val_loss_epoch}")
    # Log validation loss to TensorBoard
    writer.add_scalar('Validation Loss', val_loss_epoch, epoch)

# Close the SummaryWriter
writer.close()

# Function to compute moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Apply moving average to smooth the curves
window_size = 100  # You can adjust this value to smooth the curves more or less
smoothed_train_losses = moving_average(train_losses, window_size)
smoothed_eval_losses = moving_average(eval_losses, window_size)

# Plot training and validation loss over the same number of steps
plt.figure(figsize=(10, 5))
num_steps = min(len(smoothed_train_losses), len(smoothed_eval_losses))  # Ensure both have the same length
plt.plot(np.arange(1, num_steps + 1), smoothed_train_losses[:num_steps], label="Training Loss")
plt.plot(np.arange(1, num_steps + 1), smoothed_eval_losses[:num_steps], label="Validation Loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("training_validation_loss.png") 
plt.show()