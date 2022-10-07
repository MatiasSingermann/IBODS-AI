from tflite_support.task import vision
import config
MODEL_PATH = config.MODEL_PATH
MODEL_NAME = config.MODEL_NAME
model_path = f'{MODEL_PATH}/{MODEL_NAME}'


detector = vision.ObjectDetector.create_from_file(model_path)

import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se puede abrir la camara")
    exit()

while True:
  ret,frame = cap.read() # Voy agarrando los frames

  if ret == True: # Si leyo correctamente los frames, signfica que el retun (ret) es true, por lo que ejecuto el codigo en ese caso
    cv2.cvtColor(frame, cv2.COLOR_BGRRGB) # Convierto todos los frames al espectro RGB
    tensor_image = vision.TensorImage.create_from_array(frame) # Convierto todos los frames a tensores
    detector.detect(tensor_image)
    key = cv2.waitKey(1)
    if key == ord('q'):
      break
  if not ret:
        print("No puedo recibir imagenes (finalizaste la trasmision?). Saliendo ...")
        break
cap.release()
cv2.destroyAllWindows()