#%%

from google.colab import drive
drive.flush_and_unmount()
drive.mount('/content/drive', force_remount=True)
import sys
!ls /content/drive/MyDrive/'Colab Notebooks'/conformal-prediction-introduction
sys.path.append('/content/drive/MyDrive/Colab Notebooks/conformal-prediction-introduction')

#%%
import torch
from torchvision import datasets, transforms, models
from src.data import IndexedDataset
from src.train import train_model, evaluate_and_save

import torch.nn as nn
from functools import partial


# def main():
print("Running with",  "cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_folder = "dataset"
results_folder = "results"
num_workers = 6
val_frac = 0.2
holdout_frac = 0.2
epochs = 1
lr = 1e-3

# Data transforms
transform_train = transforms.Compose(
    [
        transforms.Resize(224), # Resize image
        transforms.RandomHorizontalFlip(), # Randomly mirror images
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
transform_test = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

# Datasets and loaders
full_trainset = IndexedDataset(
    datasets.CIFAR10(
        root=data_folder, train=True, download=True, transform=transform_train
    )
)
val_size = int(val_frac * len(full_trainset))
holdout_size = int(holdout_frac * len(full_trainset))
train_size = len(full_trainset) - val_size - holdout_size
trainset, valset, holdoutset = torch.utils.data.random_split(
    full_trainset, [train_size, val_size, holdout_size]
)
testset = IndexedDataset(
    datasets.CIFAR10(
        root=data_folder, train=False, download=True, transform=transform_test
    )
)

dataloader_settings = partial(
    torch.utils.data.DataLoader,
    batch_size=64,
    num_workers=num_workers,
    pin_memory=True if device.type == "cuda" else False,
    persistent_workers=True,
)

trainloader = dataloader_settings(trainset, shuffle=True, drop_last=True)
valloader = dataloader_settings(valset, shuffle=False)
testloader = dataloader_settings(testset, shuffle=False)
holdoutloader = dataloader_settings(holdoutset, shuffle=False)

# Model setup
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

#%%

# Train the model
model = train_model(model, trainloader, valloader, device, epochs=epochs, lr=lr)

# Evaluation on validation set
evaluate_and_save(model, valloader, device, results_folder, "val_predictions.pth")

# Evaluation on test set
evaluate_and_save(model, testloader, device, results_folder, "test_predictions.pth")

# Save the model and predictions on the holdout set
torch.save(model.state_dict(), results_folder + "/cifar10_resnet18.pth")
evaluate_and_save(model, holdoutloader, device, results_folder, "holdout_predictions.pth")
#%%

model2 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model2.fc = nn.Identity()
model2 = model2.to(device)

with torch.no_grad():
        for idx, inputs, targets in valloader:
            inputs = inputs.to(device)
            outputs = model2(inputs)
            print(outputs.shape)
            break