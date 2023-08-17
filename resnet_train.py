import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

class CustomClassifier(nn.Module):
    def __init__(self):
        super(CustomClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.resnet(x)

class LungImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # This should be a numpy array
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        image = image.repeat(3, 1, 1)

        return image, label

print("load")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_df = pd.read_csv('sorted_output.csv')

# Instantiate your custom classifier
model = CustomClassifier()
model.to(device)

# Map class names to numerical labels (0 for Normal, 1 for anything else)
class_name_to_label = { 'No finding': 0, 'Abnormal' : 1 }
# labels = image_df['class_name'].apply(lambda x: 0 if x == 'No finding' else 1).values
labels = image_df['class_name'].apply(lambda x: 1 if x == 'Cardiomegaly' or x == 'Aortic enlargement' else 0).values
print("Labels:", labels)

images = np.load("rois.npy")
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Split the dataset into training and validation sets
print("Splitting dataset into training and validation sets...")
train_size = int(0.8 * len(images))
val_size = len(images) - train_size
train_files, val_files = images[:train_size], images[train_size:]
train_labels, val_labels = labels[:train_size], labels[train_size:]

train_dataset = LungImageDataset(train_files, train_labels, transform=transform)
val_dataset = LungImageDataset(val_files, val_labels, transform=transform)

# Calculate class weights
total_samples = 200 + 1540
weight_normal = total_samples / (2 * 1540)  # Inversely proportional to normal class
weight_abnormal = torch.tensor([total_samples / (2 * 200)])  # Inversely proportional to abnormal class

# Initialize the model, loss function, and optimizer
print("Initializing model, loss function, and optimizer...")
model = CustomClassifier()
criterion = nn.BCEWithLogitsLoss(pos_weight=weight_abnormal)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create data loaders for training and validation
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Train the model
print("Training the model...")
num_epochs = 20
best_accuracy = 0.0  # Initialize best accuracy

for epoch in tqdm(range(num_epochs), desc='Training', total=num_epochs, unit='epoch'):
    model.train()
    epoch_loss = 0.0
    total_accuracy = 0.0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch', leave=False) as pbar:
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            print(labels.shape)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predicted = torch.round(torch.sigmoid(outputs))
            accuracy = (predicted == labels.unsqueeze(1)).sum().item() / labels.size(0)
            total_accuracy += accuracy

            # Update inner tqdm description
            pbar.set_postfix({'Loss': loss.item(), 'Accuracy': accuracy})
            pbar.update()

    epoch_loss /= len(train_loader)
    avg_accuracy = total_accuracy / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}")

    # Evaluate the model on the validation data
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in tqdm(val_loader, desc='Validation', total=len(val_loader), unit='batch', leave=False):
            outputs = model(images)
            predicted = torch.round(torch.sigmoid(outputs))
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()

        accuracy = correct / total
        print(f'Validation Accuracy: {accuracy:.2f}')

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            saved_model_path = 'best_model.pth'
            torch.save(model.state_dict(), saved_model_path)
            print("Best model saved!")

print("Training complete.")
