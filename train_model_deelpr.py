import torch
import torch.optim as optim
from network_1 import FatigNet
#from dataloader import AudioDataset, DataLoader
from dataloader import EMGDataset, DataLoader
from torch.nn import BCELoss
from tqdm import tqdm

# Number of epochs
epochs = 100

# device E:\an_5_sem1\an_5_sem1\TB\Lab_TB\cod\networks\network_1.py

device = 'cpu'

# Load datasets
train_dataset = EMGDataset('dataset/DATASET_DEEPLR/dataset_labeled_divided_nonoveralped_TRAIN_DEEPLR')
test_dataset = EMGDataset('dataset/DATASET_DEEPLR/dataset_labeled_divided_nonoveralped_VAL_DEEPLR')

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Initialize model and optimizer
model = FatigNet()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = BCELoss()

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss_train = 0

    for inputs, targets in tqdm(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs).float()
        loss = criterion(outputs[:, 0], targets)
        total_loss_train += loss
        loss.backward()
        optimizer.step()

    avg_loss_train = total_loss_train / len(train_loader)
    print(f"Epoch {epoch}, Avg Train Loss: {avg_loss_train}")

    # Testing loop
    model.eval()
    total_loss_test = 0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            total_loss_test += criterion(outputs, targets).item()
            predicted_classes = outputs.round()
            # Update correct predictions count
            correct_predictions += (predicted_classes == targets).sum().item()
            total_predictions += targets.size(0)
    
    avg_loss_test = total_loss_test / len(test_loader)
    accuracy = correct_predictions / total_predictions
    print(f"Epoch {epoch}, Avg Test Loss: {avg_loss_test}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save the model
    if epoch % 5 == 0:
        torch.save(model.state_dict(), f'./models/audio_model_epoch{epoch}.pth')

