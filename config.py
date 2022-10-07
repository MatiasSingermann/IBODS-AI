import os

#Agarro el directorio actual

cwd = os.getcwd()

if not os.path.exists('logs'):
    os.mkdir('logs')

if not os.path.exists(f'{cwd}/result'):
    os.mkdir(f'{cwd}/result')

TRAIN_DATASET_PATH = f'{cwd}/IBODS_LBL/train'
VALID_DATASET_PATH = f'{cwd}/IBODS_LBL/valid'
TEST_DATASET_PATH = f'{cwd}/IBODS_LBL/test'
MODEL_PATH = f'{cwd}/logs'

MODEL = 'efficientdet_lite1'
MODEL_NAME = 'model1.tflite'
CLASSES = ['cordon','autos','personas','cruces','pozos','parar','cruzar','bicicleta','moto','escalones']
EPOCHS = 100
BATCH_SIZE = 8