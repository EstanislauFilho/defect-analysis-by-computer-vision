import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from skimage import io
from libraries import etl
from settings import setup
from libraries import graphics
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Input, AveragePooling2D, Flatten, \
                                    Dropout, MaxPool2D, Conv2D, BatchNormalization, \
                                    Activation, Add, UpSampling2D, Concatenate


class Main:
    def __init__(self) -> None:
        pass

    def main(self):
        steel_plates_dataset = etl.load_dataset(setup.DATASET_PATH)

        defective_steel_plates_dataset = etl.load_dataset(setup.DATASET_DEFECT_TRAIN)

        # quantity_defective_steel_plates = \
        #     (defective_steel_plates_dataset.shape[0] / steel_plates_dataset.shape[0]) * 100
            
        defective_steel_plates_summary = etl.count_of_defect_types_per_image(defective_steel_plates_dataset)

        print(defective_steel_plates_summary)

        # graphics.relationship_between_defects_and_no_defects(steel_plates_dataset, display=False)
        # graphics.relationship_between_types_of_defects(defective_steel_plates_dataset, display=False)
        # graphics.count_of_defect_types_per_image(defective_steel_plates_summary, display=True)

        for i in range(2):
            img = io.imread(os.path.join("./images/train_images/", defective_steel_plates_dataset['ImageId'][i]))
            plt.figure()
            plt.title(defective_steel_plates_dataset['ClassId'][i])
            plt.imshow(img)
            plt.show()

if __name__ == "__main__":
    run = Main()
    run.main()


