from PIL import Image, ImageEnhance
from tensorflow.keras.utils import Sequence
import os
import random
import cv2
import numpy as np
from skimage.segmentation import slic


class DataGenerator(Sequence):
    def __init__(self, folder_path, file_ids, image_size, batch_size=4, augment=True):
        """
        target classes can be a list from Good Crypts / Good Villi / Interpretable Region / Epithelium / Muscularis Mucosa
        mode should be one of 'seg', 'loc' or 'full'
        """
        print("Initialising data generator")
        # Making the image ids list
        self.folder_path = folder_path
        self.image_size = image_size
        with open(file_ids) as f:
            self.image_ids = f.readlines()
            self.image_ids = [imgid.strip() for imgid in self.image_ids]
        self.batch_size = batch_size
        self.augment = augment
        print("Image count in {} path: {}".format(self.folder_path, len(self.image_ids)))
        self.on_epoch_end()

    def on_epoch_end(self):
        random.shuffle(self.image_ids)

    def __len__(self):
        """ Returns the number of batches per epoch """
        gen_len = len(self.image_ids) // self.batch_size
        if len(self.image_ids) % self.batch_size != 0:
            gen_len += 1
        return gen_len

    def load_image(self, index):
        """
        Load an image at the index.
        Returns PIL image
        """
        image_path = os.path.join(self.folder_path, self.image_ids[index] + '.jpg')
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        if w != self.image_size[0] and h != self.image_size[1]:
            img = img.resize((self.image_size[0], self.image_size[1]))
        return img

    def augment_instance(self, img, flip_hor=None, flip_ver=None, rotate_90=None, brightness_factor=None,
                         contrast_factor=None):
        """
        Args:
            PIL img
        Takes in an image and creates M+1 transformations of it and returns a query image along with its positive key images.
        """
        if flip_hor is None:
            flip_hor = np.random.randint(2)
            # flip_hor = 0
        if flip_ver is None:
            flip_ver = np.random.randint(2)
            # flip_ver = 0
        if rotate_90 is None:
            rotate_90 = np.random.randint(4)
            # rotate_90 = 1
        if brightness_factor is None:
            brightness_factor = 0.2 * random.random() + 0.9
        if contrast_factor is None:
            contrast_factor = 0.2 * random.random() + 0.9

        w, h = img.size

        # Flip left-right
        if flip_hor == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Flip top-bottom
        if flip_ver == 1:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # rotate 90 degrees anticlock
        if rotate_90 >= 1:
            img = img.rotate(90, expand=True)
            # Now image is in portrait shape, We need landscape window from it
            w_new, h_new = img.size
            w_crop = h
            h_crop = int(h * (w_crop / w))
            left = 0
            right = h
            upper = int(random.random() * (h_new - h))
            lower = upper + h_crop
            rotation_crop = (left, upper, right, lower)
            img = img.crop(rotation_crop)
            img = img.resize((w, h))

            # random brightness and contrast   
            brighten = ImageEnhance.Brightness(img)
            img = brighten.enhance(brightness_factor)
            contrast = ImageEnhance.Contrast(img)
            img = contrast.enhance(contrast_factor)

        return img

    def preprocess_instance(self, superPixel, image):
        """
        Args:
            PIL image
        """
        w, h = superPixel.size
        superPixel = np.array(superPixel)
        # Convert (H, W, C) to (W. H, C)
        superPixel = np.transpose(superPixel, (1, 0, 2))
        # img = np.clip(img - np.median(img)+127, 0, 255)
        superPixel = superPixel.astype(np.float32)
        # img = (img - np.min(img)) / (np.max(img) - np.min(img))
        superPixel = superPixel / 255.0

        w, h = image.size
        image = np.array(image)
        # Convert (H, W, C) to (W. H, C)
        image = np.transpose(image, (1, 0, 2))
        # img = np.clip(img - np.median(img)+127, 0, 255)
        image = image.astype(np.float32)
        # img = (img - np.min(img)) / (np.max(img) - np.min(img))
        image = image / 255.0
        return image, superPixel

    def get_instance(self, index):
        """
        index is the index of the sample in the main array of indices
        returns the PIL image, a dict of label: masks with bboxes of IRs in format (x, y, w, h) where x, y are top left coords
        """
        # start = time.time()
        # Load the source image and its annotations
        img = self.load_image(index)
        w, h = img.size

        # Perform random augmentations
        if self.augment:
            img = self.augment_instance(img)

        superPixel = np.array(img)
        real_image = np.array(img)
        enhancer = ImageEnhance.Contrast(img)
        pink_mass = enhancer.enhance(4.0)
        pink_mass = np.array(pink_mass)
        pink_mass = 255 - ((pink_mass[:, :, 0] > 150) * (pink_mass[:, :, 1] > 150) * (pink_mass[:, :, 2] > 150)) * 255
        pink_mass = pink_mass / 255
        img = np.array(img)
        img[:, :, 0] = img[:, :, 0] * pink_mass
        img[:, :, 1] = img[:, :, 1] * pink_mass
        img[:, :, 2] = img[:, :, 2] * pink_mass
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        img = cv2.erode(img, element, iterations=1)
        img = cv2.dilate(img, element, iterations=1)
        img = cv2.erode(img, element)

        # apply SLIC and extract (approximately) the supplied number of segments
        segments = slic(real_image, n_segments=500, sigma=5)

        # Randomly uniformly sample superpixel clusters
        for i in range(500):
            k = int(np.random.uniform(np.min(segments), np.max(segments)))
            if np.sum(img[segments == k]) > 2000:
                real_image[segments == k] = 0

        # Preprocess the image, masks and bboxes
        superPixel, real_image = self.preprocess_instance(Image.fromarray(superPixel), Image.fromarray(real_image))

        # print("Aug and preprocess time = {:.5f}s".format(time.time() - start))
        # start = time.time()

        return superPixel, real_image

    def getitem(self, index):
        """
        index is the index of batch here
        """
        # start = time.time()

        batch_indices = [i for i in range(index * self.batch_size, (index + 1) * self.batch_size)]
        batch_indices = [i % len(self.image_ids) for i in batch_indices]
        superPixel_imgs = []
        real_imgs = []
        for ind in batch_indices:
            # istart = time.time()
            superpixel, image = self.get_instance(ind)
            # print("Instance time = {:.5f}s".format(time.time() - istart))
            superPixel_imgs.append(superpixel)
            real_imgs.append(image)
        superPixel_imgs = np.array(superPixel_imgs)  # (B, w, h, 3)
        real_imgs = np.array(real_imgs)  # (B, M, w, h, 3)

        # print("Batch generation time = {:.5f}s".format(time.time() - start))
        return superPixel_imgs, real_imgs

    def __getitem__(self, index):
        return self.getitem(index)
