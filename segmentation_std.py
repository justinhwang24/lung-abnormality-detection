import numpy as np
from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from skimage import transform, measure

# Load and Preprocess Data
test_base_path = "/content/ChestXray/test/"
test_image_path = os.path.join(test_base_path, "image/")
test_mask_path = os.path.join(test_base_path, "mask/")
train_base_path = "/content/ChestXray/train/"
train_image_path = os.path.join(train_base_path, "image/")
train_mask_path = os.path.join(train_base_path, "mask/")
val_base_path = "/content/ChestXray/val/"
val_image_path = os.path.join(val_base_path, "image/")
val_mask_path = os.path.join(val_base_path, "mask/")

testing_image_files = sorted(os.listdir(test_image_path))
testing_mask_files = sorted(os.listdir(test_mask_path))
training_image_files = sorted(os.listdir(train_image_path))
training_mask_files = sorted(os.listdir(train_mask_path))
val_image_files = sorted(os.listdir(val_image_path))
val_mask_files = sorted(os.listdir(val_mask_path))

def contrast_normalization(image):
    cont_norm_image = cv2.equalizeHist(image)
    gaussian_blurred_image = cv2.GaussianBlur(cont_norm_image, (5, 5), 0)
    median_filtered_image = cv2.medianBlur(gaussian_blurred_image, 5)

    return median_filtered_image

def lung_standardization(image, mask):
    labeled_mask = measure.label(mask)
    regions = measure.regionprops(labeled_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    lung_regions = [regions[0].image, regions[1].image]

    # Calculate the center of each lung region using the coordinates of nonzero elements
    centers = [np.mean(np.where(region), axis=1) for region in lung_regions]

    # Calculate the angle of rotation needed to align each lung region
    angles = [np.arctan2(center[0] - center[1], center[1] - center[0]) for center in centers]

    aligned_lung_images = []
    aligned_lung_masks = []

    for lung_index in range(2):
        lung_region = lung_regions[lung_index]
        center_y, center_x = centers[lung_index]

        # Get the rotation angle from the AffineTransform object
        rotation_angle = -np.degrees(angles[lung_index])

        # Perform rotation and translation to align the lung region to the center
        aligned_lung_mask = transform.rotate(mask, rotation_angle, resize=False, center=(center_x, center_y), order=0)
        aligned_lung_mask = np.roll(aligned_lung_mask, (int(center_y - np.mean(np.where(lung_region)[0])),
                                                        int(center_x - np.mean(np.where(lung_region)[1]))),
                                     axis=(0, 1))

        # Warp the lung image using the same transformation
        aligned_lung_image = transform.rotate(image, rotation_angle, resize=False, center=(center_x, center_y), order=3)
        aligned_lung_image = np.roll(aligned_lung_image, (int(center_y - np.mean(np.where(lung_region)[0])),
                                                          int(center_x - np.mean(np.where(lung_region)[1]))),
                                       axis=(0, 1))

        aligned_lung_masks.append(aligned_lung_mask)
        aligned_lung_images.append(aligned_lung_image)

    return aligned_lung_images, aligned_lung_masks

def getData(X_shape, flag = "test"):
    im_array = []
    mask_array = []

    if flag == "test":
        print("test")
        length = len(testing_image_files) // 4
        for img_file, mask_file in tqdm(zip(testing_image_files[:length],
                                            testing_mask_files[:length])):
            if not img_file.endswith(".DS_Store") and not mask_file.endswith(".DS_Store"):
                im = cv2.resize(cv2.imread(os.path.join(test_image_path, img_file)), (X_shape, X_shape))[:, :, 0]
                mask = cv2.resize(cv2.imread(os.path.join(test_mask_path, mask_file)), (X_shape, X_shape))[:, :, 0]
                normalized_im = contrast_normalization(im)
                im_array.append(normalized_im)
                mask_array.append(mask)

    elif flag == "train":
        print("train")
        length = len(training_image_files) // 4
        for img_file, mask_file in tqdm(zip(training_image_files[:length],
                                            training_mask_files[:length])):
            if not img_file.endswith(".DS_Store") and not mask_file.endswith(".DS_Store"):
                im = cv2.resize(cv2.imread(os.path.join(train_image_path, img_file)), (X_shape, X_shape))[:, :, 0]
                mask = cv2.resize(cv2.imread(os.path.join(train_mask_path, mask_file)), (X_shape, X_shape))[:, :, 0]
                normalized_im = contrast_normalization(im)
                im_array.append(normalized_im)
                mask_array.append(mask)

    elif flag == "val":
        print("val")
        length = len(val_image_files) // 4
        for img_file, mask_file in tqdm(zip(val_image_files[:length],
                                            val_mask_files[:length])):
            if not img_file.endswith(".DS_Store") and not mask_file.endswith(".DS_Store"):
                im = cv2.resize(cv2.imread(os.path.join(val_image_path, img_file)), (X_shape, X_shape))[:, :, 0]
                mask = cv2.resize(cv2.imread(os.path.join(val_mask_path, mask_file)), (X_shape, X_shape))[:, :, 0]
                normalized_im = contrast_normalization(im)
                im_array.append(normalized_im)
                mask_array.append(mask)

    return im_array, mask_array

def plotMask(X, y):
    sample = []

    for i in range(6):
        left = X[i]
        right = y[i]
        combined = np.hstack((left, right))
        sample.append(combined)

    for i in range(0,6,3):

        plt.figure(figsize=(25,10))

        plt.subplot(2,3,1+i)
        plt.imshow(sample[i])

        plt.subplot(2,3,2+i)
        plt.imshow(sample[i+1])


        plt.subplot(2,3,3+i)
        plt.imshow(sample[i+2])

        plt.show()

dim = 256
X_train, y_train = getData(dim, flag="train")
X_test, y_test = getData(dim)
X_val, y_val = getData(dim, flag="val")

# print("training set")
# plotMask(X_train, y_train)
# print("testing set")
# plotMask(X_test, y_test)
# print("validation set")
# plotMask(X_test, y_test)

print("concat")
X_train = np.array(X_train).reshape(len(X_train),dim,dim,1)
y_train = np.array(y_train).reshape(len(y_train),dim,dim,1)
X_test = np.array(X_test).reshape(len(X_test),dim,dim,1)
y_test = np.array(y_test).reshape(len(y_test),dim,dim,1)
X_val = np.array(X_val).reshape(len(X_val),dim,dim,1)
y_val = np.array(y_val).reshape(len(y_val),dim,dim,1)
assert X_train.shape == y_train.shape
assert X_test.shape == y_test.shape
assert X_val.shape == y_val.shape
image = np.concatenate((X_train, X_test, X_val), axis=0)
mask = np.concatenate((y_train, y_test, y_val), axis=0)
image = image.astype('float32') / 255.0

print("save")
np.save('lung_images.npy', image)
np.save('lung_masks.npy', mask)