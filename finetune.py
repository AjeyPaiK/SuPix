from datasets.finetuning_dataset import DataGenerator
from model import attention_unet_refined

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from loss.losses import uniclass_dice_coeff_0, uniclass_dice_coeff_1, uniclass_dice_coeff_2, multiclass_dice_loss

tf.random.set_seed(100)
np.random.seed(100)
model_path = '/content/drive/My Drive/SupPix/supPix_finetune.h5'  # Saving location
log_path = '/content/drive/My Drive/SupPix/supPix_finetune.csv'
weights_path = '/content/drive/My Drive/SupPix/supPix_pretrain.h5'
images_folder = "/content/drive/My Drive/Data/Images"
train_image_list = "/content/drive/MyDrive/Data/Lists/train50.txt"
valid_image_list = "/content/drive/MyDrive/Data/Lists/valid40.txt"
target_classes = ["Good Crypts", "Good Villi", "Epithelium"]
## Parameters
image_size = (320, 256)  # Original = (2448, 1920)
batch_size = 6
lr = 1e-3
epochs = 500

model = attention_unet_refined(input_shape=image_size, mask_channels=3, out_channels=3,
                               multiplier=10, freeze_segmentor=False, dropout_rate=0.0)
metrics = [uniclass_dice_coeff_0, uniclass_dice_coeff_1, uniclass_dice_coeff_2]
losses = multiclass_dice_loss(loss_scales=[1, 1, 1])
optimizer = Adam(learning_rate=lr)
model.compile(loss=losses, optimizer=optimizer, metrics=metrics)
model.summary()

# Resume from checkpoint
if weights_path is not None:
    model.load_weights(weights_path, by_name=True)
    print("loaded!")

train_generator = DataGenerator(images_folder, train_image_list, batch_size=batch_size, mode='seg',
                                target_classes=target_classes, image_size=image_size, augment=True)
valid_generator = DataGenerator(images_folder, valid_image_list, batch_size=batch_size, mode='seg',
                                target_classes=target_classes, image_size=image_size, augment=True)
callbacks = [ModelCheckpoint(model_path, save_weights_only=True),
             ReduceLROnPlateau(monitor='loss', factor=0.1, patience=20, verbose=1, min_lr=1e-6),
             CSVLogger(log_path),
             EarlyStopping(monitor='loss', patience=20, restore_best_weights=False)]

model.fit(x=train_generator, steps_per_epoch=train_generator.__len__(), epochs=epochs, verbose=1, callbacks=callbacks,
          validation_data=valid_generator, validation_steps=valid_generator.__len__()
          , workers=1)
