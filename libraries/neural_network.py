
import tensorflow as tf
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

    validation_generator = datagen.flow_from_dataframe(dataframe=dataset, 
                                                  directory="./images/train_images/",
                                                  x_col = 'ImageID', y_col = 'label',
                                                  subset = 'validation', batch_size = 16,
                                                  shuffle = True, class_mode = 'other',
                                                  target_size = (256,256))
    return train_generator, validation_generator


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

def training(train_dataset, validation_dataset):
    # Utiliza-se  a resnet para fazer o treinamento utilizando a técnicas de 
    # transferência de aprendizado, ou seja, não precisa retreinar a rede neural
    # basemodel = \
        # ResNet50(weights='imagenet', include_top=False, input_tensor= Input(shape=(256, 256, 3)))

    basemodel = None
    # Cria-se a camada de saída personalizada
    headmodel = basemodel.output
    #pool_size = tamanho da matriz para fazer a redução de dimensionalidade
    headmodel = AveragePooling2D(pool_size=(4,4))(headmodel)
    # Transforma a matriz em vetor, o (headmodel) indica que uma camada esta
    # sendo ligada em outra
    headmodel = Flatten()(headmodel)
    # Define-se uma única camada densa
    headmodel = Dense(256, activation='relu')(headmodel)
    # Na camada de dropout zera-se 30% dos neurônios durante o 
    # treinamento para se evitar o overfitting
    headmodel = Dropout(0.3)(headmodel)
    # defini-se um único na camada de saída pois temos
    # um problema de classificação binária
    headmodel = Dense(1, activation='sigmoid')(headmodel)
    # Criação da arquitetura da rede
    model = Model(inputs = basemodel.input, outputs = headmodel)
    # Compila-se o modelo de ann
    model.compile(loss = 'binary_crossentropy', optimizer = 'Nadam', metrics = ['accuracy'])

    # Faz uma parada antecipada do treinamento caso a patir de 20 épocas não haja 
    # redução no valor do val_loss
    earlystopping = EarlyStopping(monitor='val_loss', mode='min', patience=20)

    # Salva os pesos se a ann conseguir evoluir no seu treinamento e desempenho
    # da rede
    checkpointer = ModelCheckpoint(filepath='weights.hdf5', save_best_only=True)

    # Instrução que irá efetivar o treinamento
    history = model.fit_generator(train_dataset, epochs=40, 
                              validation_data=validation_dataset, 
                              callbacks=[checkpointer, earlystopping])

    # Estrutura para fazer o salvamento da arquitetura da ann
    model_json = model.to_json()
    with open("resnet-classifier-model.json","w") as json_file:
        json_file.write(model_json)

def load_model(model_path, weights_path):
    with open(model_path, "r") as json_file:
        json_model = json_file.read()
    model = tf.keras.models.model_from_json(json_model)
    model.load_weights(weights_path)
    model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    return model