# IBODS-AI

## Funcionamiento

El funcionamiento de este modelo de inteligencia artificial se basa en el modelo de `EfficientDet0`, familia de los modelos `EfficientNet`. 

Usando este modelo, tendriamos la posibilidad de detectar diversos objetos, con el objetivo de poder avisarle al usuario lo que deberia hacer. Para lograr esto, la I.A deberia ser capaz de detectar objetos tipicos de la calle, los cuales seleccionamos entre los siguientes:

- Escalones
- Personas
- Vehículos (Autos, bicicletas y motos)
- Cruces de calle
- Pozos
- Semaforos peatonales
- Cordones de calle

Para ilustrar a lo que me refiero, esta es un ejemplo de imagen que reciba la inteligencia artificial y que necesite detectar los objetos pertinentes.

**INPUT IMAGE**

![Input Image](https://user-images.githubusercontent.com/101400526/195096876-939e9341-dd66-4a15-a491-e04f52f2259c.png)

Aca estarian los resultados de como detectaria la IA los objetos, con sus respectivos porcentajes sobre la seguridad que tiene sobre la presencia de estos objetos.

**OUTPUT IMAGE**

![Ouput Image](https://user-images.githubusercontent.com/101400526/195097920-12c099f8-63a3-47dc-b48c-3eacf4b9495b.png)

Tal como se ve en la imagen, deberiamos ser capaz de poder detectar los objetos que tenemos pensado, con el fin de poder dar las ordenes necesarias. 

## Instrucciones

Ejecutar en la terminal de la raspi el acrhivo `test.py`

## Fuentes

Para aquellos que quieran obtener mas informacion y un informe mas detallado del funcionamiento de yolov5, les dejo el siguiente material:

[Official yolov5 Git](https://github.com/ultralytics/yolov5)

[yolov5 Data](https://www.v7labs.com/blog/yolo-object-detection)

[yolov5 Data](https://www.section.io/engineering-education/introduction-to-yolo-algorithm-for-object-detection/)

[yolov5 Data](https://medium.com/analytics-vidhya/yolo-explained-5b6f4564f31)




