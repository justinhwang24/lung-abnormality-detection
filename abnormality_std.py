import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
import csv
from skimage.morphology import remove_small_objects
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.ndimage import zoom
from tqdm import tqdm

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

    if stats.shape[0] == 1:
        return False

    # Get the areas of all connected components
    areas = stats[:, 4]
    bg_area = areas[0]
    areas = areas[1:]

    largest = np.max(areas)

    similarity_count = 0
    for num in areas:
        if (largest - num) / largest < 0.6:
          similarity_count = similarity_count + 1

    print("Sim count:", similarity_count, "Largest Area:", largest)
    return similarity_count == 2 or (similarity_count == 1 and largest <= 20000)

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
    # print("Sorted Indices:", sorted_indices)

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

    cropped_image = image[rmin:rmax, cmin:cmax]
    cropped_image = cv2.resize(cropped_image.squeeze(), (roi_size[1], roi_size[0]), interpolation=cv2.INTER_LINEAR)

    return cropped_image, aligned_roi

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

# Load the best model weights from the saved file
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Custom dataset class to work with DataLoader
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images):
        self.images = images
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        return image

def lung_segmentation_model(model, image):
    with torch.no_grad():
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        output = model(image)
        predicted_mask = torch.sigmoid(output).cpu().numpy()
        print("PM:", predicted_mask)
        print("max:", np.max(predicted_mask))
        print("min:", np.min(predicted_mask))
        # predicted_mask_binary = (predicted_mask >= 0.7).astype(np.uint8)
        predicted_mask_binary = (predicted_mask >= 0.504).astype(np.uint8)

        plt.figure()
        plt.imshow(predicted_mask_binary.squeeze(), cmap='gray')  # Assuming grayscale images
        plt.title(f"Image {i + 1}")
        plt.show()

        cleaned_mask = clean_up_mask(predicted_mask_binary)
        predicted_mask_binary = get_top_two_largest_segments(predicted_mask_binary)
        if not check_valid_images(image, cleaned_mask):
            return None
    return predicted_mask_binary


if __name__ == "__main__":
    # Load the best model weights from the saved file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet().to(device)
    model_path = 'best_model_weights.pth'
    model = load_model(model, model_path)

    print("load")
    df = pd.read_csv("ids.csv")
    image_ids = df['image_id'].values
    passing_ids = []

    images = np.load('cxr.npy')
    images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    # images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)  # Add channel dimension
    dataset = CustomDataset(images)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    rois_array = []

    with torch.no_grad():
        for i, image in enumerate(tqdm(loader.dataset)):
            # Resize the image to 256x256
            image = image.squeeze().cpu().numpy()
            resized_image1 = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
            resized_image = torch.tensor(resized_image1, dtype=torch.float32)

            # Add batch and channel dimensions to the tensor
            resized_image = resized_image.unsqueeze(0).unsqueeze(0).to(device)

            # Get the lung segmentation mask
            output = model(resized_image)
            predicted_mask = torch.sigmoid(output).cpu().numpy()
            # print("PM:", predicted_mask)
            # print("max:", np.max(predicted_mask))
            # print("min:", np.min(predicted_mask))
            predicted_mask_binary = (predicted_mask >= 0.7).astype(np.uint8)
            # predicted_mask_binary = (predicted_mask >= 0.504).astype(np.uint8)

            # plt.figure()
            # plt.imshow(predicted_mask_binary.squeeze(), cmap='gray')  # Assuming grayscale images
            # plt.title(f"Image {i + 1}")
            # plt.show()

            cleaned_mask = clean_up_mask(predicted_mask_binary)
            cleaned_mask = get_top_two_largest_segments(cleaned_mask)
            is_valid = check_valid_images(resized_image, cleaned_mask)
            print("Valid:", is_valid)
            if not is_valid:
                print("Skipping image", i)
                continue
            passing_ids.append(image_ids[i])
            resized_mask = cv2.resize(cleaned_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

            # Get the region of interest by applying the mask
            roi, roi_mask = create_roi(image, resized_mask, roi_size=(512, 512), padding=50)
            resized_roi = cv2.resize(roi, (256, 256), interpolation=cv2.INTER_NEAREST)

            rois_array.append(resized_roi)

            # Visualize some of the ROIs
            if i < 15:
                plt.figure()
                plt.subplot(1, 3, 1)
                plt.imshow(resized_image.squeeze().cpu().numpy(), cmap='gray')
                plt.title("Original Image")

                plt.subplot(1, 3, 2)
                plt.imshow(roi_mask, cmap='gray')
                plt.title("ROI Mask")

                plt.subplot(1, 3, 3)
                plt.imshow(resized_roi, cmap='gray')
                plt.title("ROI")

        print("Saved", len(rois_array), "images out of", len(dataset))
        print(len(dataset) - len(rois_array), "images eliminated")

        rois_array = np.array(rois_array)
        np.save('rois.npy', rois_array)

        reject_path = 'pass.csv'
        # Write the list to a CSV file
        with open(reject_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for passing_id in passing_ids:
                csv_writer.writerow([passing_id])

        print(f'CSV file "{reject_path}" created successfully.')

        plt.show()