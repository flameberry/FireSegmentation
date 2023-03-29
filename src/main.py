import sys
import os

import numpy as np
import pathlib
import skimage

import pathlib

ROOT_DIR = pathlib.Path(__file__).parent.parent

sys.path.append(str(ROOT_DIR / 'vendor/Mask_RCNN-TF2'))

from mrcnn.config import Config
from mrcnn import model as modellib, utils

from PIL import Image


def split_train_test(data: np.array, test_ratio, seed: int = 42):
    shuffled = np.random.RandomState(seed=seed).permutation(len(data))
    test_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_size]
    train_indices = shuffled[test_size:]
    return data[train_indices], data[test_indices]


class FireConfig(Config):
    """
    Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "fire"

    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + fire

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


class FireDataset(utils.Dataset):
    def load_fire(self, dataset_dir, is_train: bool):
        """
        Load a subset of the Fire dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("fire", 1, "fire")

        dataset_dir = os.path.join(dataset_dir, 'Image/Fire')

        image_indices = np.arange(0, 27460, 1)
        train_img_indices, test_img_indices = split_train_test(image_indices, 0.3, seed=42)

        img_list = os.listdir(dataset_dir)

        for index in (train_img_indices if is_train else test_img_indices):
            image_name = f'{img_list[index]}'
            print(image_name)

            image_path = os.path.join(dataset_dir, image_name)
            im = Image.open(image_path)
            width, height = im.size

            self.add_image(
                "fire",
                image_id=image_name,  # use file name as a unique image id
                path=image_path,
                width=width,
                height=height,
                filename=image_name
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
        one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]

        # If not a fire dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "fire":
            return super(self.__class__, self).load_mask(image_id)

        # img = Image.open(f'{str(ROOT_DIR)}/dataset/Segmentation_Mask/Fire/{info["filename"]}')
        # mask = np.array(img)
        # mask.reshape(info["width"], info["height"], 1)

        image_path = f'{str(ROOT_DIR)}/dataset/Segmentation_Mask/Fire/{info["filename"]}'
        mask = skimage.io.imread(image_path, as_gray=True).astype(bool)
        print(image_path)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(bool), np.ones([mask.shape[-1]], dtype=np.int32)


if __name__ == '__main__':
    ROOT_DIR = pathlib.Path(__file__).parent.parent

    train_dataset = FireDataset()
    train_dataset.load_fire(dataset_dir=f'{str(ROOT_DIR)}/dataset', is_train=True)
    train_dataset.prepare()

    print('Finished preparing training dataset!')

    test_dataset = FireDataset()
    test_dataset.load_fire(dataset_dir=f'{str(ROOT_DIR)}/dataset', is_train=False)
    test_dataset.prepare()

    print('Finished preparing testing dataset!')

    fire_config = FireConfig()
    model = modellib.MaskRCNN(mode='training', model_dir='../log', config=fire_config)
    print(model.keras_model.summary())

    print("Training network heads")
    model.train(
        train_dataset, test_dataset,
        learning_rate=fire_config.LEARNING_RATE,
        epochs=30,
        layers='heads'
    )
