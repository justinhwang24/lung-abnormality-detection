import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def binary_accuracy(outputs, targets):
    predicted_labels = outputs > 0.5
    correct = (predicted_labels == targets).sum().item()
    total = targets.numel()
    accuracy = correct / total
    return accuracy

def dice_loss(outputs, targets, epsilon=1e-6):
    intersection = torch.sum(outputs * targets)
    union = torch.sum(outputs) + torch.sum(targets)
    dice = 1.0 - (2.0 * intersection + epsilon) / (union + epsilon)
    return dice

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

""" Encoder block:
    It consists of an conv_block followed by a max pooling.
    Here the number of filters doubles and the height and width half after every block.
"""
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

""" Decoder block:
    The decoder block begins with a transpose convolution, followed by a concatenation with the skip
    connection from the encoder block. Next comes the conv_block.
    Here the number filters decreases by half and the height and width doubles.
"""
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x

class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(1, 32)
        self.e2 = encoder_block(32, 64)
        self.e3 = encoder_block(64, 128)
        self.e4 = encoder_block(128, 256)

        """ Bottleneck """
        self.b = conv_block(256, 512)

        """ Decoder """
        self.d1 = decoder_block(512, 256)
        self.d2 = decoder_block(256, 128)
        self.d3 = decoder_block(128, 64)
        self.d4 = decoder_block(64, 32)

        """ Classifier """
        self.outputs = nn.Conv2d(32, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        # inputs = inputs.unsqueeze(1)  # Shape: [num_samples, 1, height, width]

        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Classifier """
        outputs = self.outputs(d4)

        return torch.sigmoid(outputs)


# Custom Dataset class for training data
class CustomDataset(data.Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        return image, mask

if __name__ == "__main__":
    images = np.load('lung_images.npy')
    masks = np.load('lung_masks.npy')
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)  # Add channel dimension
    masks = torch.tensor(masks, dtype=torch.float32).permute(0, 3, 1, 2)   # Add channel dimension
    print(images.shape)

    train_size = int(0.8 * len(images))
    images_train, images_val = images[:train_size], images[train_size:]
    masks_train, masks_val = masks[:train_size], masks[train_size:]

    train_dataset = CustomDataset(images_train, masks_train)
    val_dataset = CustomDataset(images_val, masks_val)
    batch_size = 16
    num_epochs = 20

    # Create DataLoader for training and validation data
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet().to(device)
    criterion = dice_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    best_weights = None

    # Lists to store the sample indices for plotting
    sample_indices = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        total_accuracy = 0.0
        total_dice_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
            for i, (images, masks) in enumerate(train_loader):
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                target_masks = F.interpolate(masks, size=outputs.shape[2:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, target_masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                accuracy = binary_accuracy(outputs, target_masks)
                total_accuracy += accuracy
                total_dice_loss += loss.item()

                pbar.set_postfix({'Dice Loss': loss.item(), 'Binary Accuracy': accuracy})
                pbar.update()

                 # For plotting, save indices of some samples from the last batch of the last epoch
                if epoch == num_epochs - 1 and i == len(train_loader) - 1:
                    sample_indices.extend(np.random.choice(images.size(0), size=min(9, images.size(0)), replace=False))

        epoch_loss /= len(train_loader)
        avg_accuracy = total_accuracy / len(train_loader)
        avg_dice_loss = total_dice_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Binary Accuracy: {avg_accuracy:.4f}, Dice Loss: {avg_dice_loss:.4f}")

        # After computing the validation loss
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                target_masks = F.interpolate(masks, size=outputs.shape[2:], mode='bilinear', align_corners=False)
                val_loss += criterion(outputs, target_masks).item()

            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = model.state_dict()
                print("Saving best weights!")
                torch.save(best_weights, 'best_model_weights.pth')

    print("Training finished!")