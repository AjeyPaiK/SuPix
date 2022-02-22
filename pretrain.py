from datasets.pretraining_dataset import DataGenerator
from models.attention_unet import attention_unet_refined

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from utilities.utils import SSIMLoss

tf.random.set_seed(100)
np.random.seed(100)
SCRATCH = "/processing/a.karkala/supix"
model_path = './checkpoints/supPix_pretrain_elastic_continued_from41.h5'  # Saving location
log_path = './checkpoints/supPix_pretrain_elastic_continued_from41.csv'
weights_paths = ['./checkpoints/supPix_pretrain_elastic.h5']
images_folder = SCRATCH+"/Data/Images"
unlabelled_image_list = SCRATCH+"/Data/Lists/unlabelled.txt"
valid_image_list = SCRATCH+"/Data/Lists/valid40.txt"

## Parameters
image_size = (320, 256)  # Original = (2448, 1920)
batch_size = 6
lr = 1e-3
epochs = 500


model = attention_unet_refined(input_shape=image_size, mask_channels=3, out_channels=3,
                               multiplier=10, freeze_segmentor=False, dropout_rate=0.0)

optimizer = Adam(learning_rate=lr)
model.compile(loss=SSIMLoss, optimizer=optimizer, metrics='mse')
model.summary()

# Resume from checkpoint
if weights_paths != None:
    for wp in weights_paths:
        model.load_weights(wp, by_name=True)
        print("Loaded!")

# Data generators
train_generator = DataGenerator(images_folder, unlabelled_image_list, batch_size=batch_size, image_size=image_size, augment=True)
valid_generator = DataGenerator(images_folder, valid_image_list, batch_size=batch_size, image_size=image_size, augment=True)

callbacks = [ModelCheckpoint(model_path, save_weights_only=True),
            ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, min_lr=1e-6),
            CSVLogger(log_path),
            EarlyStopping(monitor='loss', patience=15, restore_best_weights=False)]


model.fit(x=train_generator, steps_per_epoch=train_generator.__len__(),
          epochs=epochs,
          verbose=1,
          callbacks=callbacks,
          validation_data=valid_generator,
          validation_steps=valid_generator.__len__(),
          validation_freq=5,
          workers=16,
          initial_epoch=41)
