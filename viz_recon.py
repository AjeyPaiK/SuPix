"""Display outputs of SSL"""
from utils import *
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.losses import mean_squared_error
from datasets.pretraining_dataset import DataGenerator
from model import attention_unet_refined
import os

def display_recon():

    test_image_size = (640, 512)
    pred_model = attention_unet_refined(test_image_size
                                        , 3
                                        , 3
                                        , multiplier=10
                                        , freeze_segmentor=True
                                        , dropout_rate=0.0)
    test_folders = ["/content/drive/MyDrive/Data/Images/"]
    file_ids = "/content/drive/MyDrive/Data/Lists/valid40.txt"
    pred_model.load_weights('/content/drive/MyDrive/SupPix/supPix_pretrain.h5', by_name=True)
    predictor = pred_model.predict_on_batch

    for test_folder in test_folders:

        test_generator = DataGenerator(test_folder, file_ids, image_size=test_image_size, batch_size=1, augment=True)
        files = os.listdir(test_folder)
        file_ids = list(set([f.replace('.jpg', '').replace('.json', '').replace('.xml', '') for f in files]))
        print("Running Inpainting on {} images".format(len(file_ids)))
        l2_loss = []

        for i in tqdm(range(test_generator.__len__()), position=0, leave=True):
            """Fetching instance from the generator"""
            inputs, targets = test_generator.__getitem__(i)

            """Processing the input"""
            input_img = tensor2image(inputs[0])

            """Processing the ground truth data"""
            gt_img = tensor2image(targets[0])

            """Making predictions"""
            inputs = tf.convert_to_tensor(inputs)
            restored = predictor(inputs)
            restored_img = tensor2image(restored[0])

            """Calculate metric"""
            l2 = mean_squared_error(inputs, restored)
            l2_loss.append(l2)

            """Combined visualization"""
            divider = Image.new('RGB', (10, restored_img.height), (255, 255, 255))
            combined_img = Image.new('RGB', (restored_img.width * 3, restored_img.height))
            combined_img.paste(input_img, (0, 0))
            combined_img.paste(divider, (restored_img.width, 0))
            combined_img.paste(restored_img, (restored_img.width + 10, 0))
            combined_img.paste(divider, (restored_img.width * 2 + 10, 0))
            combined_img.paste(gt_img, (restored_img.width * 2 + 20, 0))
            display(combined_img)

        l2_loss = np.array(l2_loss)
        print_value("Average L2 loss", np.mean(l2_loss))
        print('\n')


display_recon()