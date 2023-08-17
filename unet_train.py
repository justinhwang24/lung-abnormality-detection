import numpy as np
import os
import cv2
import tensorflow as tf
from tqdm import tqdm
from skimage import transform
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras import backend as K

def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(tf.keras.backend.cast(y_true, 'float32'))
    y_pred_f = tf.keras.backend.flatten(tf.keras.backend.cast(y_pred, 'float32'))
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def unet(input_size=(256,256,1)):
    inputs = Input(input_size)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])

print("load")
images = np.load('lung_images.npy')
masks = np.load('lung_masks.npy')

weight_path="{}_weights.best.hdf5".format('cxr_reg')
# model.load_weights(weight_path)  # Load the weights from the saved file

checkpoint = ModelCheckpoint(weight_path, monitor='loss', verbose=1, save_best_only=True)

early = EarlyStopping(monitor="loss",
                      patience=5, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

callbacks_list = [checkpoint, early]

X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

print("model")
model = unet(input_size=(256, 256, 1))
model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy',
                  metrics=[dice_coef, 'binary_accuracy'])
loss_history = model.fit(X_train, y_train, batch_size=4, epochs=20,
                         validation_data=(X_test, y_test), callbacks=callbacks_list)
predictions = model.predict(X_test)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
ax1.plot(loss_history.history['loss'], '-', label = 'Loss')
ax1.plot(loss_history.history['val_loss'], '-', label = 'Validation Loss')
ax1.legend()

ax2.plot(100*np.array(loss_history.history['binary_accuracy']), '-',
         label = 'Accuracy')
ax2.plot(100*np.array(loss_history.history['val_binary_accuracy']), '-',
         label = 'Validation Accuracy')
ax2.legend()

pred_candidates = np.random.randint(0, len(X_test), 9)
preds = model.predict(X_test)

plt.figure(figsize=(20,10))

# Compute AUROC
y_test_binary = (y_test > 0.5).astype(int)
preds_binary = (preds > 0.5).astype(int)
auroc = roc_auc_score(np.ravel(y_test_binary), np.ravel(preds_binary))
print("AUROC:", auroc)

for i in range(0, 9, 3):
    plt.subplot(3, 3, i + 1)
    plt.imshow(np.squeeze(X_test[pred_candidates[i]]))
    plt.xlabel("Base Image")

    plt.subplot(3, 3, i + 2)
    plt.imshow(np.squeeze(y_test[pred_candidates[i]]))
    plt.xlabel("Mask")

    plt.subplot(3, 3, i + 3)
    plt.imshow(np.squeeze(preds[pred_candidates[i]]))
    plt.xlabel("Prediction")

plt.show()
