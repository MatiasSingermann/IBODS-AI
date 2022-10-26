import cv2
import numpy as np
import tensorflow as tf
import os
from PIL import Image
from tflite_support.task import vision
from tflite_support.task.vision import ObjectDetector
from tflite_support.task.vision import ObjectDetectorOptions
from tflite_support.task.processor import DetectionOptions
from tflite_support.task.core import BaseOptions
from tflite_support import task

import config

CLASSES = config.CLASSES

# Define a list of colors for visualization
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype=np.uint8)

def preprocess_image(image_path, input_size):
  ''' Preprocess the input image to feed to the TFLite model
  '''
  img = tf.io.read_file(image_path)
  img = tf.io.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.uint8)
  original_image = img
  resized_img = tf.image.resize(img, input_size)
  resized_img = resized_img[tf.newaxis, :]
  resized_img = tf.cast(resized_img, dtype=tf.uint8)
  return resized_img, original_image


def detect_objects(interpreter, image, threshold):
  ''' Returns a list of detection results, each a dictionary of object info.
  '''

  signature_fn = interpreter.get_signature_runner()

  # Feed the input image to the model
  output = signature_fn(images=image)

  # Get all outputs from the model
  count = int(np.squeeze(output['output_0']))
  scores = np.squeeze(output['output_1'])
  classes = np.squeeze(output['output_2'])
  boxes = np.squeeze(output['output_3'])

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
      }
      results.append(result)
  return results


def run_odt_and_draw_results(image_path, interpreter, threshold=0.5):
  ''' Run object detection on the input image and draw the detection results
  '''
  
  # Load the input shape required by the model
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  # Load the input image and preprocess it
  preprocessed_image, original_image = preprocess_image(
      image_path,
      (input_height, input_width)
    )

  # Run object detection on the input image
  results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

  # Plot the detection results on the input image
  original_image_np = original_image.numpy().astype(np.uint8)
  for obj in results:
    # Convert the object bounding box from relative coordinates to absolute
    # coordinates based on the original image resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * original_image_np.shape[1])
    xmax = int(xmax * original_image_np.shape[1])
    ymin = int(ymin * original_image_np.shape[0])
    ymax = int(ymax * original_image_np.shape[0])

    # Find the class index of the current object
    class_id = int(obj['class_id'])

    # Draw the bounding box and label on the image
    color = [int(c) for c in COLORS[class_id]]
    cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), color, 2)
    # Make adjustments to make the label visible for all objects
    y = ymin - 15 if ymin - 15 > 15 else ymin + 15
    label = "{}: {:.0f}%".format(CLASSES[class_id], obj['score'] * 100)
    cv2.putText(original_image_np, label, (xmin, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  # Return the final image
  original_uint8 = original_image_np.astype(np.uint8)
  return original_uint8

def drawBoundingBox(image_path, model_path, threshold, classes):

  # Agarro el directorio (carpeta) actual

  cwd = os.getcwd()

  # Le asigno los tres parametros:
  # - La ruta de la imagen a detectar
  # - La ruta del modelo de IA
  # - El umbral de la IA

  image_path = image_path
  model_path = model_path
  threshold = threshold

  # Leo la imagen como un tensor
  tensor_image = task.vision.TensorImage.create_from_file(image_path)

  # Abro y leo la imagen como un array
  image = Image.open(image_path)
  image = np.asarray(image)

  # Divido la imagen en 3 (Izquierda, derecha y centro) utilizando el ancho de la imagen

  wid = image.shape[1]
  left = wid/3
  center = left * 2
  right = wid 

  # Asigno las opciones de configuracion del modelo de deteccion (Que modelo, el umbral, etc)
  base_opt = BaseOptions(model_path)
  detection_opt = DetectionOptions(score_threshold = threshold)
  options = ObjectDetectorOptions(base_opt, detection_opt)

  # Creo el modelo de deteccion
  detector = ObjectDetector.create_from_options(options)

  # Agarro los resultados de la deteccion
  results = detector.detect(tensor_image)
  resultsdetect = results.detections

  # Creo un randomizador de colores por clase
  CLASSES = classes
  COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype=np.uint8)

  # Recorro cada objetos de los resultdos de deteccion

  for obj in resultsdetect:

    # Selecciono todos los ids, nombres, porcentajes de los objetos detectados

    categories = obj.categories 
    for category in categories: 
      names = category.category_name 
      scores = category.score * 100 
      class_id = category.index 

    # Creo a porcentajes en una lista

    percentages = [float(scores)]

    # Por cada id (identificador unico del objeto), le asigno un color

    color = [int(c) for c in COLORS[class_id]]

    # Usando las coordenadas del punto de abajo a la izquierda, creo el punto de arriba de la derecha mediante el width y el height, y creo la bounding box, uniendo ambos puntos

    start_point = obj.bounding_box.origin_x, obj.bounding_box.origin_y 
    end_point = obj.bounding_box.origin_x + obj.bounding_box.width, obj.bounding_box.origin_y + obj.bounding_box.height
    detect_img = cv2.rectangle(image, start_point, end_point, color, 2)

    # Creo las coordenadas de los puntos en x, los de abajo

    x1 = obj.bounding_box.origin_x
    x2 = obj.bounding_box.origin_x + obj.bounding_box.width

    # Calculo para saber la posicion del objeto
    middle = obj.bounding_box.width / 2
    Center = x2 - middle

    x = (x1, x2)

    # Comprobacion de la posicion del objeto
    if Center < left:
      position = "izquierda"
    elif Center < center:
      position = "centro"
    else:
      position = "derecha"
    
    # Creo el texto a poner y lo pongo arriba de las bounding boxes

    for percentage in percentages:
      confidence = str(int(percentage)) + '%'
      label = names + ", " + confidence
      cv2.putText(image, label, (obj.bounding_box.origin_x,obj.bounding_box.origin_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
      cv2.putText(image, position, (obj.bounding_box.origin_x,obj.bounding_box.origin_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  # Leo devuelta la imagen, para pasarla de array a imagen y luego la devuelvo 
  img = Image.fromarray(detect_img, 'RGB')
  return img