

from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications import ResNet50
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Input, AveragePooling2D, Flatten, \
                                    Dropout, MaxPool2D, Conv2D, BatchNormalization, \
                                    Activation, Add, UpSampling2D, Concatenate



def dataset_separation(dataset):
    train_dataset, test_dataset = train_test_split(dataset, test_size = 0.15)
    return train_dataset, test_dataset

def data_generation_for_training(dataset):
    # treinamento será usado para rede neural aprender, ajustando os pesos da ann
    # Faz a leitura das imagens que estão no disco
    # Faz a normalização das imagens do dataset para padronização e otimização da execução
    # Faz a separação da base de imagens para treinamento, validação e teste.
    # O shuffle é ativado somente nesta etapa para evitar que a rede neural aprenda
    # a partir da ordem das imagens
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)

    # Faz o carregamento dos dados do dataset , considerando tamanho do lote igual a 16,
    # a imagens serão misturadas (shuffle = True), problema de multiplas classes 
    # (class_mode = other), e tamanho das imagens target_size = (256,256)
    # A arquitetura ResNet exige que as imagens estejam no formato 256x256
    train_generator = datagen.flow_from_dataframe(dataframe=dataset, 
                                                  directory="./images/train_images/",
                                                  x_col = 'ImageID', y_col = 'label',
                                                  subset = 'training', batch_size = 16,
                                                  shuffle = True, class_mode = 'other',
                                                  target_size = (256,256))
    return train_generator

def data_generation_for_validation(dataset):
    # validação será utilizada para testar a eficacia do treinamento em cada época
    # Faz a leitura das imagens que estão no disco
    # Faz a normalização das imagens do dataset para padronização e otimização da execução
    # Faz a separação da base de imagens para treinamento, validação e teste.
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)

    # Faz o carregamento dos dados do dataset , considerando tamanho do lote igual a 16,
    # a imagens serão misturadas (shuffle = True), problema de multiplas classes 
    # (class_mode = other), e tamanho das imagens target_size = (256,256)
    # A arquitetura ResNet exige que as imagens estejam no formato 256x256
    validation_generator = datagen.flow_from_dataframe(dataframe=dataset, 
                                                  directory="./images/train_images/",
                                                  x_col = 'ImageID', y_col = 'label',
                                                  subset = 'validation', batch_size = 16,
                                                  shuffle = True, class_mode = 'other',
                                                  target_size = (256,256))
    return validation_generator


def data_generation_for_test(dataset):
    # validação será utilizada para testar a eficacia do treinamento em cada época
    # Faz a leitura das imagens que estão no disco
    # Faz a normalização das imagens do dataset para padronização e otimização da execução
    # Faz a separação da base de imagens para treinamento, validação e teste.
    datagen = ImageDataGenerator(rescale=1./255.)

    # Faz o carregamento dos dados do dataset , considerando tamanho do lote igual a 16,
    # a imagens serão misturadas (shuffle = True), problema de multiplas classes 
    # (class_mode = None) pois é somente o teste, e tamanho das imagens target_size = (256,256)
    # A arquitetura ResNet exige que as imagens estejam no formato 256x256
    test_generator = datagen.flow_from_dataframe(dataframe=dataset, 
                                                  directory="./images/train_images/",
                                                  x_col = 'ImageID', y_col = None,
                                                  batch_size = 16, shuffle = False,
                                                  class_mode = None,
                                                  target_size = (256,256))
    return test_generator