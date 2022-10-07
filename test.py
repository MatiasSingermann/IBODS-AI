import os

import tensorflow as tf
from PIL import Image

from helper_functions import run_odt_and_draw_results
import config

cwd = os.getcwd()

MODEL_PATH = config.MODEL_PATH
MODEL_NAME = config.MODEL_NAME

DETECTION_THRESHOLD = 0.3

# Change the test file path to your test image
INPUT_IMAGE_PATH = 'IBODS_LBL-1/test/Audi_A7_2012_59_18_310_30_6_75_55_195_18_AWD_4_4_4dr_aqB_jpg.rf.2cdf87614d0d1c937d1e97bbdd338b2c.jpg'

im = Image.open(INPUT_IMAGE_PATH)
im.thumbnail((512, 512), Image.ANTIALIAS)
im.save(f'{cwd}/result/input.png', 'PNG')

# Load the TFLite model
model_path = f'{MODEL_PATH}/{MODEL_NAME}'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Run inference and draw detection result on the local copy of the original file
detection_result_image = run_odt_and_draw_results(
    f'{cwd}/result/input.png',
    interpreter,
    threshold=DETECTION_THRESHOLD
)

# Show the detection result
img = Image.fromarray(detection_result_image)