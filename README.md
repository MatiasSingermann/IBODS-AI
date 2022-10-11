# IBODS-AI

## Funcionamiento

El funcionamiento de este modelo de inteligencia artificial se basa en el modelo de `EfficientDet0`, familia de los modelos `EfficientNet`. 

Estos son los distintos modelos de EfficientDet, notamas que mientras mayor es la precision, mas pesado y lento es el modelo.

| Model architecture | Size(MB)* | Latency(ms)** | Average Precision*** |
| ------------------ | --------- | ------------------ | --------- |
| EfficientDet-Lite0  | 4.4 | 37 | 25.69% |
| EfficientDet-Lite1  | 5.8 | 49 | 30.55% |
| EfficientDet-Lite2  | 7.2 | 69 | 33.97% |
| EfficientDet-Lite3  | 11.4 | 116 | 37.70% |
| EfficientDet-Lite4  | 19.9 | 260 | 41.96% |

(*) *Size of the integer quantized models.*

(**) *Latency measured on Pixel 4 using 4 threads on CPU.*

(***) *Average Precision is the mAP (mean Average Precision) on the COCO 2017 validation dataset.*

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

Para aquellos que quieran obtener mas informacion y un informe mas detallado del funcionamiento de todo el proyecto, les dejo el siguiente material

https://docs.google.com/document/d/1mwpc2iAVPkO-ZTgS3Kg5jJp_Yd80aTJYRnWdDrSZ-cQ/edit?usp=sharing



