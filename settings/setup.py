""" Script para configurações da aplicação
"""
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

DATASET_PATH = "./dataset/defect_and_no_defect.csv"
DATASET_DEFECT_TRAIN = "./dataset/train.csv"
IMAGES_PATH = "./images/"

RESNET_MODEL_PATH = "./models/resnet-classifier-model.json"
RESNET_WEIGHTS_PATH = "./models/weights.hdf5"