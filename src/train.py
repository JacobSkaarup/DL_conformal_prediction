import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm


def evaluate_and_save(model, dataloader, device, folder, filename):
    """Evaluates the model on the given dataloader and saves the predictions."""
    
   

    model.eval()
    results = {
        "indexes": [],
        "labels": [],
        "outputs": [],
        "latents": [],
    }

    latents_batch = []
    correct = 0
    total = 0

    def hook(module, input, output):
        latents_batch.append(output.squeeze().cpu())

    handle = model.avgpool.register_forward_hook(hook)

    with torch.no_grad():
        for idx, inputs, targets in dataloader:
            latents_batch.clear()
            inputs = inputs.to(device)
            outputs = model(inputs)

            results["outputs"].append(outputs.cpu())
            results["latents"].append(latents_batch[0])

            results["labels"].extend(targets.cpu().tolist())
            results["indexes"].extend(idx.cpu().tolist())
            
            targets = targets.to(device)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


    handle.remove()

    # Concatenate lists into arrays
    results["outputs"] = torch.cat(results["outputs"], dim=0).numpy()
    results["latents"] = torch.cat(results["latents"], dim=0).numpy()
    results["labels"] = np.array(results["labels"])
    results["indexes"] = np.array(results["indexes"])


    torch.save(results, f"{folder}/{filename}")
    accuracy = 100.0 * correct / total
    print(f"Accuracy on {filename.split('_')[0]} set: {accuracy:.2f}%")


def train_model(model, trainloader, valloader, device, epochs=10, lr=1e-3):
    """Trains the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        for _, inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(trainloader.dataset)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _, inputs, targets in valloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_accuracy = 100.0 * correct / total
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%"
        )

    return model
