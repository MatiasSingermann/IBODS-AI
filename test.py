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
INPUT_IMAGE_PATH = 'IBODS_LBL.voc/test/heon_IMG_0685_JPG_jpg.rf.cd500306853d7810fc4473f8e74bed47.jpg' 

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
img.save(f'{cwd}/result/ouput.png')