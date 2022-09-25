import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from skimage import io
from libraries import etl
from settings import setup
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
        dataset = etl.load_dataset(setup.DATASET_PATH)

if __name__ == "__main__":
    run = Main()
    run.main()


