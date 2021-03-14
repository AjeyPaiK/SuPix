import os
from dataloader import DataGenerator
from model import attention_unet_refined
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, MeanIoU
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard

tf.random.set_seed(100)
np.random.seed(100)

model_path = './supPix.h5'
log_path = './logs/supPix.csv'
weights_paths = None
train_path = "../Data/unlabelled/"
valid_path = "../Data/Valid/"
## Parameters
image_size = (640, 512) # Original = (2448, 1920)
batch_size = 4
lr = 1e-4
epochs = 300

def SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1))

def SSIM(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1))


model = attention_unet_refined(input_shape=image_size, mask_channels=3, out_channels=3, 
                               multiplier=5, freeze_segmentor=False, dropout_rate=0.0)

optimizer = Adam(learning_rate=lr)
model.compile(loss= SSIMLoss, optimizer=optimizer, metrics='mse')
model.summary()

# Resume from checkpoint
if weights_paths != None:
    for wp in weights_paths:
        model.load_weights(wp, by_name=True)
        print("Loaded!")


# Data generators
train_generator = DataGenerator(train_path, batch_size=batch_size, image_size =image_size, augment=True)
valid_generator = DataGenerator(valid_path, batch_size=batch_size, image_size = image_size, augment=False)
callbacks = [ModelCheckpoint(model_path, save_weights_only=True),
            ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, min_lr=1e-6),
            CSVLogger(log_path),
            # TensorBoard(),
            EarlyStopping(monitor='loss', patience=15, restore_best_weights=False)]

model.fit(x = train_generator, steps_per_epoch = train_generator.__len__(), 
	epochs = epochs, 
	verbose = 1, 
	callbacks = callbacks, 
	validation_data= valid_generator, 
	validation_steps= valid_generator.__len__(),
	validation_freq = 5,
	workers = 4,
	initial_epoch = 0)