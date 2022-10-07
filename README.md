# IBODS-AI

## Funcionamiento

El funcionamiento de este modelo de inteligencia artificial se basa en el modelo previamente creado `yolov5`. Este modelo cuenta con gran una gran rapidez y eficiencia en el procesado de datos, al igual que una destacable versatilidad para poder detectar distintos tipos de objetos. A la par de esto, estamos por nuestra propia cuenta, desarrollando una IA desde 0, usando nada más que OpenCV.

Usando este modelo, tendriamos la posibilidad de detectar diversos objetos, con el objetivo de poder avisarle al usuario lo que deberia hacer. Para lograr esto, la I.A deberia ser capaz de detectar objetos tipicos de la calle, los cuales seleccionamos entre los siguientes:

- Arboles
- Postes de luz
- Semaforos peatonales
- Cruces de calle
- Pozos
- Personas
- Vehiculos
- Escalones

Para ilustrar a lo que me refiero, les presento la siguiente imagen:

![Imagen de referencia](https://learn.alwaysai.co/hs-fs/hubfs/object-dectection-4-2.jpg?width=900&height=569&name=object-dectection-4-2.jpg)

Tal como se ve en la imagen, deberiamos ser capaz de poder detectar los objetos que tenemos pensado, con el fin de poder dar las ordenes necesarias. 

## Instrucciones

Ejecutar en la terminal de la raspi el acrhivo `test.py`

## Fuentes

Para aquellos que quieran obtener mas informacion y un informe mas detallado del funcionamiento de yolov5, les dejo el siguiente material:

[Official yolov5 Git](https://github.com/ultralytics/yolov5)

[yolov5 Data](https://www.v7labs.com/blog/yolo-object-detection)

[yolov5 Data](https://www.section.io/engineering-education/introduction-to-yolo-algorithm-for-object-detection/)

[yolov5 Data](https://medium.com/analytics-vidhya/yolo-explained-5b6f4564f31)




