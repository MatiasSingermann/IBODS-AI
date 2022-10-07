from tflite_support.task import vision
import config
MODEL_PATH = config.MODEL_PATH
MODEL_NAME = config.MODEL_NAME
model_path = f'{MODEL_PATH}/{MODEL_NAME}'

detector = vision.ObjectDetector.create_from_file(model_path)
tensor_image = vision.TensorImage.create_from_file('IBODS_LBL-1/test/Audi_A7_2012_59_18_310_30_6_75_55_195_18_AWD_4_4_4dr_aqB_jpg.rf.2cdf87614d0d1c937d1e97bbdd338b2c.jpg')
detector.detect(tensor_image)