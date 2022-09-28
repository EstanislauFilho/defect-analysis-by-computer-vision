import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from libraries import etl
from settings import setup
from libraries import graphics
from libraries import image_functions
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications import ResNet50
from sklearn.model_selection import train_test_split
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
            
        defect_type_dataset = etl.count_of_defect_types_per_image(defective_steel_plates_dataset)

        graphics.relationship_between_defects_and_no_defects(steel_plates_dataset, display=False)
        graphics.relationship_between_types_of_defects(defective_steel_plates_dataset, display=False)
        graphics.count_of_defect_types_per_image(defect_type_dataset, display=False)

        # print(defective_steel_plates_dataset['EncodedPixels'])

        # função que cria uma macara com o mesmo tamanho da altura e largura da imagem
        # RUN LENGTH ENCODING (RLE) é uma técnica para compressão de dados que armazena 
        # sequências que contém dados consecutivos (combinação em um só valor)
        # Considerando que temos uma imagem com texto preto em fundo branco
        # Em uma image 800x600 tem se 480.000 valores e com o método RLE tem uma sequencia
        # bem reduzida o que ocupa menos espaço para armazenamento. 
        img = image_functions.load_image(dataset=defective_steel_plates_dataset, img_number=20)
        mask = image_functions.rle2mask(defective_steel_plates_dataset['EncodedPixels'][20], img.shape[0], img.shape[1])       
        plt.figure()
        plt.title(defective_steel_plates_dataset['ClassId'][20])
        plt.imshow(mask)
        # plt.show()

        # Converte a imagem para escala de cores RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Nos pixels que foram identificados os defeitos, faz o preenchimento na cor verde
        img[mask == 1,1] = 255
        plt.figure()
        plt.title(defective_steel_plates_dataset['ClassId'][20])
        plt.imshow(img)
        # plt.show()

        train, test = train_test_split(steel_plates_dataset, test_size = 0.15)

if __name__ == "__main__":
    run = Main()
    run.main()


