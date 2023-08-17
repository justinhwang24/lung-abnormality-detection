import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import cv2
from skimage.morphology import remove_small_objects
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import cv2
import numpy as np

def clean_up_mask(mask):
    mask = mask.squeeze()

    # Use morphological operations to separate connected components
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, cv2.UMat(kernel), iterations=10)
    mask = cv2.dilate(mask, cv2.UMat(kernel), iterations=10)
    mask = cv2.morphologyEx(cv2.UMat(mask), cv2.MORPH_OPEN, cv2.UMat(kernel), iterations=1)

    # Remove small objects
    mask = remove_small_objects(mask.get(), min_size=500)
    return mask

def check_valid_images(image, pred_mask):
    mask = pred_mask.squeeze()
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))

    # Get the areas of all connected components
    areas = stats[:, 4]
    bg_area = areas[0]
    areas = areas[1:]

    largest = np.max(areas)

    similarity_count = 0
    for num in areas:
        if (largest - num) / largest < 0.6:
          similarity_count = similarity_count + 1

    return similarity_count <= 2 and largest <= 32768

def get_top_two_largest_segments(mask, min_area_threshold=100):
    # Ensure mask is 2-dimensional
    mask = mask.squeeze()
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))

    # Get the areas of all connected components
    areas = stats[:, 4]
    areas[0] = 0

    print("Areas:", areas)

    # Sort the areas in descending order and get the indices of the two largest areas
    sorted_indices = np.argsort(areas)[::-1][:2]
    print("Sorted Indices:", sorted_indices)

    # Create a binary mask with only the two largest segments
    segmented_mask = np.zeros_like(mask, dtype=np.uint8)
    for index in sorted_indices:
        if areas[index] >= min_area_threshold:
            segmented_mask[labels == index] = 1

    # Invert the segmented mask to have 0s for background and 1s for the two largest segments
    # segmented_mask = 1 - segmented_mask

    return segmented_mask

def create_roi(image, mask, roi_size=(256, 256), padding=25):
    mask = mask.squeeze()  # Remove the channel dimension if present

    # Find the bounding box that tightly encloses the segmented mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add padding to the bounding box
    rmin = max(0, rmin - padding)
    rmax = min(mask.shape[0], rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(mask.shape[1], cmax + padding)

    # Calculate the scaling factors to resize the bounding box to the desired ROI size
    scale_factor_r = roi_size[0] / (rmax - rmin)
    scale_factor_c = roi_size[1] / (cmax - cmin)

    print(f"scale_factor_r: {scale_factor_r}, scale_factor_c: {scale_factor_c}")

    # Resize the mask to fit inside the ROI using the calculated scaling factors
    resized_mask = cv2.resize(mask[rmin:rmax, cmin:cmax], (roi_size[1], roi_size[0]), interpolation=cv2.INTER_LINEAR)

    # Calculate the crop positions to center the resized mask within the ROI
    rstart = roi_size[0] // 2 - resized_mask.shape[0] // 2
    rend = rstart + resized_mask.shape[0]
    cstart = roi_size[1] // 2 - resized_mask.shape[1] // 2
    cend = cstart + resized_mask.shape[1]

    print(f"resized_mask shape: {resized_mask.shape}")
    print(f"rstart: {rstart}, rend: {rend}, cstart: {cstart}, cend: {cend}")

    # Create a fixed-size ROI and align the resized mask inside the ROI
    aligned_roi = np.zeros(roi_size, dtype=mask.dtype)
    aligned_roi[rstart:rend, cstart:cend] = resized_mask

    cropped_image = image[:, :, rmin:rmax, cmin:cmax]
    cropped_image = cv2.resize(cropped_image.squeeze(), (roi_size[1], roi_size[0]), interpolation=cv2.INTER_LINEAR)

    return cropped_image, aligned_roi

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

def load_model(model, weights_file):
    model.load_state_dict(torch.load(weights_file))
    model.eval()
    return model

def calculate_auroc(model, val_loader, device):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predicted_mask = torch.sigmoid(outputs).cpu().numpy()
            predicted_mask_binary = (predicted_mask >= 0.7).astype(np.uint8)
            # predicted_mask_binary = clean_up_mask(predicted_mask_binary)
            if check_valid_images(images.cpu().numpy(), predicted_mask_binary):
              updated_masks = predicted_mask_binary
              # updated_masks = get_top_two_largest_segments(predicted_mask_binary)
              target_masks = F.interpolate(masks, size=outputs.shape[2:], mode='bilinear', align_corners=False)
              y_true.extend(target_masks.cpu().detach().numpy().flatten())
              y_pred.extend(updated_masks.flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_test_binary = (y_true > 0.5).astype(int)
    preds_binary = (y_pred > 0.5).astype(int)

    auroc = roc_auc_score(np.ravel(y_test_binary), np.ravel(preds_binary))
    return auroc

def plot_invalid_images(model, val_loader, device, num_samples=9):
    invalid_samples = []
    with torch.no_grad():
        model.eval()
        for i, (images, masks) in enumerate(val_loader):
            # if (len(invalid_samples) == num_samples):
            #   break
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predicted_mask = torch.sigmoid(outputs).cpu().numpy()
            predicted_mask_binary = (predicted_mask >= 0.7).astype(np.uint8)
            predicted_mask_binary = clean_up_mask(predicted_mask_binary)
            if not check_valid_images(images.cpu().numpy(), predicted_mask_binary):
                invalid_samples.append((images, masks))

    print("Invalid count:", len(invalid_samples))
    plt.figure(figsize=(12, 9))

    # ground truth vs predicted
    with torch.no_grad():
        for i, (images, masks) in enumerate(invalid_samples):
            # images, masks = images.squeeze(0).to(device), masks.squeeze(0).to(device)

            outputs = model(images)
            predicted_mask = torch.sigmoid(outputs).cpu().numpy()
            predicted_mask_binary = (predicted_mask >= 0.7).astype(np.uint8)
            predicted_mask_binary = clean_up_mask(predicted_mask_binary)

            print(check_valid_images(images.cpu().numpy(), predicted_mask_binary))
            # predicted_mask_binary = get_top_two_largest_segments(predicted_mask_binary)

            plt.subplot(3, num_samples, i + 1)
            plt.imshow(np.squeeze(images.cpu().numpy()), cmap='gray')
            plt.xlabel("Base Image")

            plt.subplot(3, num_samples, num_samples + i + 1)
            plt.imshow(np.squeeze(masks.cpu().numpy()), cmap='gray')
            plt.xlabel("Ground Truth Mask")

            plt.subplot(3, num_samples, 2 * num_samples + i + 1)
            plt.imshow(np.squeeze(predicted_mask_binary), cmap='gray')
            plt.xlabel("Predicted Mask")

            if i == num_samples - 1:
              break

    plt.tight_layout()
    plt.show()

def plot_sample_predictions(model, val_loader, device, num_samples=9):
    sample_indices = np.random.choice(len(val_loader.dataset), size=num_samples, replace=False)
    losses = []
    with torch.no_grad():
        model.eval()
        # Worst losses including invalid images
        # for i, (images, masks) in enumerate(val_loader):
        #     images, masks = images.to(device), masks.to(device)
        #     outputs = model(images)
        #     target_masks = F.interpolate(masks, size=outputs.shape[2:], mode='bilinear', align_corners=False)
        #     loss = dice_loss(outputs, target_masks)
        #     losses.extend(np.array([loss.cpu()]))
        #     print(i, loss)

        # Worst losses excluding invalid images
        # for i, (images, masks) in enumerate(val_loader):
        #     images, masks = images.to(device), masks.to(device)
        #     outputs = model(images)
        #     predicted_mask = torch.sigmoid(outputs).cpu().numpy()
        #     predicted_mask_binary = (predicted_mask >= 0.7).astype(np.uint8)
        #     predicted_mask_binary = clean_up_mask(predicted_mask_binary)
        #     if check_valid_images(images.cpu().numpy(), predicted_mask_binary):
        #         target_masks = F.interpolate(masks, size=outputs.shape[2:], mode='bilinear', align_corners=False)
        #         loss = dice_loss(outputs, target_masks)
        #         losses.extend(np.array([loss.cpu()]))
        #         print(i, loss)

    # Find the indices of masks with the highest loss values
    # sample_indices = np.argsort(losses)[-num_samples:] # worst loss
    # worst_losses = np.array(losses)[sample_indices]
    # print("Worst Losses:", worst_losses)

    plt.figure(figsize=(12, 9))

    # ground truth vs predicted
    # with torch.no_grad():
    #     for i, idx in enumerate(sample_indices):
    #         images, masks = val_loader.dataset[idx]
    #         images, masks = images.unsqueeze(0).to(device), masks.unsqueeze(0).to(device)

    #         outputs = model(images)
    #         predicted_mask = torch.sigmoid(outputs).cpu().numpy()
    #         predicted_mask_binary = (predicted_mask >= 0.7).astype(np.uint8)
    #         predicted_mask_binary = clean_up_mask(predicted_mask_binary)
    #         predicted_mask_binary = get_top_two_largest_segments(predicted_mask_binary)

    #         plt.subplot(3, num_samples, i + 1)
    #         plt.imshow(np.squeeze(images.cpu().numpy()), cmap='gray')
    #         plt.xlabel("Base Image")

    #         plt.subplot(3, num_samples, num_samples + i + 1)
    #         plt.imshow(np.squeeze(masks.cpu().numpy()), cmap='gray')
    #         plt.xlabel("Ground Truth Mask")

    #         plt.subplot(3, num_samples, 2 * num_samples + i + 1)
    #         plt.imshow(np.squeeze(predicted_mask_binary), cmap='gray')
    #         plt.xlabel("Predicted Mask")

    # roi mask and image
    with torch.no_grad():
      for i, idx in enumerate(sample_indices):
          images, masks = val_loader.dataset[idx]
          images, masks = images.unsqueeze(0).to(device), masks.unsqueeze(0).to(device)

          outputs = model(images)
          predicted_mask = torch.sigmoid(outputs).cpu().numpy()
          predicted_mask_binary = (predicted_mask >= 0.7).astype(np.uint8)
          predicted_mask_binary = get_top_two_largest_segments(predicted_mask_binary)

          print("Masks", masks.cpu().numpy())
          print("PM", predicted_mask)

          roi_image, roi_mask = create_roi(images.cpu().numpy(), predicted_mask_binary)
          print(f"Cropped Image Shape: {roi_image.shape}")

          plt.subplot(3, num_samples, i + 1)
          plt.imshow(np.squeeze(images.cpu().numpy()), cmap='gray')
          plt.xlabel("Base Image")

          plt.subplot(3, num_samples, num_samples + i + 1)
          plt.imshow(np.squeeze(roi_mask), cmap='gray')
          plt.xlabel("ROI Mask")

          plt.subplot(3, num_samples, 2 * num_samples + i + 1)
          plt.imshow(np.squeeze(roi_image), cmap='gray')
          plt.xlabel("ROI Image")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load the best model weights from the saved file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet().to(device)
    model_path = 'best_model_weights.pth'
    model = load_model(model, model_path)

    images = np.load('lung_images.npy')
    masks = np.load('lung_masks.npy')
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)  # Add channel dimension
    masks = torch.tensor(masks, dtype=torch.float32).permute(0, 3, 1, 2)   # Add channel dimension

    train_size = int(0.8 * len(images))
    images_val = images[train_size:]
    masks_val = masks[train_size:]

    val_dataset = CustomDataset(images_val, masks_val)

    # Create DataLoader for training and validation data
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Calculate the AUROC score
    # auroc = calculate_auroc(model, val_loader, device)
    # print("AUROC:", auroc)

    # Plot invalid images
    # plot_invalid_images(model, val_loader, device, num_samples=9)

    # Plot sample predictions
    plot_sample_predictions(model, val_loader, device, num_samples=9)