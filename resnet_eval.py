import torch
import pandas as pd
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score
from PIL import Image
from tqdm import tqdm
from captum.attr import GuidedBackprop

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

def guided_backpropagation(model, image):
    model.eval()
    image.requires_grad_()

    # Forward pass
    prediction = model(image)
    target_class = prediction.argmax()

    # Zero out gradients
    model.zero_grad()

    # Calculate gradients of the prediction with respect to the input image
    prediction[:, target_class].backward()

    # Get the gradients from the input image
    gradients = image.grad.cpu().numpy()
    gradients = np.max(gradients, axis=1)

    return gradients

print("load")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_df = pd.read_csv('sorted_output.csv')

# Instantiate your custom classifier
model = CustomClassifier()
model.load_state_dict(torch.load('best_model.pth'))  # Load the saved weights
model.to(device)

# Map class names to numerical labels (0 for Normal, 1 for anything else)
class_name_to_label = { 'No finding': 0, 'Abnormal' : 1 }
labels = image_df['class_name'].apply(lambda x: 0 if x == 'No finding' else 1).values
# labels = image_df['class_name'].apply(lambda x: 1 if x == 'Cardiomegaly' or x == 'Aortic enlargement' else 0).values
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

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Evaluate the model
model.eval()
all_labels = []
all_predictions = []
guided_backprop = GuidedBackprop(model)

# Prepare a few test images for visualization
sample_indices = list(range(30))  # Replace with the indices of the samples you want to visualize

with torch.no_grad():
    for i, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.unsqueeze(1).float().to(device)
        outputs = model(images)

        # Convert logits to probabilities using sigmoid activation
        predictions = torch.sigmoid(outputs)

        all_labels.append(labels.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

    # for image in sample_images:

        # image = torch.tensor(image).unsqueeze(0).to(device)  # Convert to tensor and add batch and channel dimensions
        # output = model(image)
        if i > 30 and i < 60:
            images = images.to(device)
            labels = labels.unsqueeze(1).float().to(device)

            # Perform guided backpropagation on the images
            guided_gradients = guided_backprop.attribute(images, target=0)  # Assuming binary classification

            # Normalize gradients for visualization
            gradients_norm = (guided_gradients - guided_gradients.min()) / (guided_gradients.max() - guided_gradients.min())
            gradients_norm = gradients_norm.mean(dim=1).cpu().numpy()

            predicted_prob = predictions.item()
            predicted_perc = predicted_prob * 100
            grayscale_image = np.mean(images.cpu().numpy().squeeze(), axis=0)  # Compute the mean along the RGB channels
            ground_truth_label = labels.item() if labels.numel() == 1 else labels[0].item()
            ground_truth_label = "Normal" if ground_truth_label == 0 else "Abnormal"
            print(ground_truth_label)

            plt.imshow(grayscale_image, cmap='gray')  # Display grayscale image
            plt.imshow(gradients_norm[0], cmap="hot", alpha=0.7, vmin=0.2, vmax=0.8)  # Overlay gradients heatmap
            plt.title(f'Probability of Disease: {predicted_perc:.2f}%')
            plt.text(70, -20, f'Ground Truth: {ground_truth_label}', fontsize=12, color='black')
            plt.show()

# Concatenate the results
all_labels = np.concatenate(all_labels)
all_predictions = np.concatenate(all_predictions)

# Calculate AUROC score
# auroc = roc_auc_score(all_labels, all_predictions)

# Compute ROC curve and AUROC score
fpr, tpr, _ = roc_curve(all_labels, all_predictions)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Convert probabilities to binary predictions
binary_predictions = (all_predictions > 0.5).astype(int)

accuracy = accuracy_score(all_labels, binary_predictions)
tn, fp, fn, tp = confusion_matrix(all_labels, binary_predictions).ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

# Print the metrics
print(f'AUROC Score: {auroc:.4f}')
print(f'Binary Accuracy: {accuracy:.4f}')
print(f'Specificity: {specificity:.4f}')
print(f'Sensitivity: {sensitivity:.4f}')
print('Confusion Matrix:')
print(confusion_matrix(all_labels, binary_predictions))
