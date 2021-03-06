# Preparación
Se deben agregar las carpetas:
* *media*: contiene las imágenes de entrada y salida.
    * *input*: donde se deben localizar las imágenes a analizar.s
    * *output*: donde se guardarán las máscaras generadas.
* *model*: contiene el modelo de la red neuronal utilizado para realizar la deteccion de punto de la cara.

## Agregar librerias de python
```python
# Update pip
python -m pip install -U pip
# Install scikit-image
python -m pip install -U scikit-image`
```
# Scripts
## cutImageForCasscade.py
Este script recibe el nombre de una imagen jpg dentro de la carpeta *media/input/* y genera un equivalente compatible en blanco y negro para entrenar a la NN. La salida es en *media/output/forTraining* (se debe crear la carpeta antes de usar).
## maskGenerator.py
Este script recibe el nombre de una imagen jpg dentro de la carpeta *media/input/* y genera una máscara de la cara del gato (si es que fue detectada correctamente). La salida es en *media/output/*.
## maskComparator.py
Este script recibe el nombre de 2 máscaras en la carpeta *media/output/* y compara si corresponden al mismo gato.
## picturesCompatibleWithCasscade.py
Este script inicializa la cámara web y si detecta gatos (según el har-casscade) las almacena dentro de la carpeta *media/webcamCascade/* (debe ser creada la carpeta antes de usar).

# Ejecución
El archivo *principal.py* es el script que recibe una imagen de entrada y a partir de la misma genera la máscara (en caso que se detecte una cara). La máscara es almacenada en la carpeta *media/output*. En caso que no se detecte una cara, se informará con un mensaje de error y se cerrará el script.

# Links utiles

Firebase
https://morioh.com/p/4dca3ded4cea

De donde saque el tuto para el calibrador:
    + https://pysource.com/2019/02/15/detecting-colors-hsv-color-space-opencv-with-python/
    + https://www.youtube.com/watch?v=SJCu1d4xakQ
    + https://pysource.com/2018/01/31/object-detection-using-hsv-color-space-opencv-3-4-with-python-3-tutorial-9/

De donde saque el tuto para el reconocedor de colores (gameboy colors):
    + https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/

Posible manera de comparar 2 imagenes:
    + https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/

Escalas de colores en varios formatos:
    + http://colorizer.org/

Lo que quiero tratar de lograr:
    + http://labs.tineye.com/color/d7baceba95b7660cd10f83afa61040df7e1a35ab?ignore_background=True&ignore_interior_background=True&width=250&height=224&scroll_offset=528

Pizarron de Trello del proyecto:
    + https://trello.com/b/OLuZzmaW/proyecto-final

Ejemplo del que me base para obtener los contornos de los colores que fui obteniendo
    + https://stackoverflow.com/questions/57282935/how-to-detect-area-of-pixels-with-the-same-color-using-opencv