#Crear las carpetas para subir las imagenes
!mkdir ccotorra_argentina
!mkdir cescribano_palustre
!mkdir cmina_comun
!mkdir ctarro_canelo
!mkdir cmilano_real
'''Es importante descomprimir los archivos, uno por uno, y no cargar las imágenes de golpe en una carpeta, puesto que podemos sobrecargar el entorno, interrumpiendo el 
proceso de subida y corrompiendo imágenes posteriormente'''
%cd ccotorra_argentina
!unzip cotorra_argentina.zip.zip
%cd ..

%cd cescribano_palustre
!unzip escribano_palustre.zip.zip
%cd ..

%cd cmilano_real
!unzip milano_real.zip.zip
%cd ..

%cd cmina_comun
!unzip mina_comun.zip.zip
%cd ..

%cd ctarro_canelo
!unzip tarro_canelo.zip.zip
%cd ..

#Borrar los archivo ZIP
!rm -rf /content/cotorra_argentina.zip.zip
!rm -rf /content/cuchillos/escribano_palustre.zip.zip
!rm -rf /content/tenedores/milano_real.zip.zip
!rm -rf /content/tenedores/mina_comun.zip.zip
!rm -rf /content/tenedores/tarro_canelo.zip.zip

#Mostrar cuantas imagenes tengo de cada categoria
!ls /content/ccotorra_argentina/ | wc -l
!ls /content/cescribano_palustre/ | wc -l
!ls /content/cmilano_real/ | wc -l
!ls /content/cmina_comun/ | wc -l
!ls /content/ctarro_canelo/ | wc -l
import os

# Directorio base donde se encuentran las carpetas
base_dir = '/content/'

# Lista de carpetas
carpetas = ['ccotorra_argentina', 'cescribano_palustre', 'cmilano_real', 'cmina_comun', 'ctarro_canelo']

# Función para contar elementos en una carpeta
def contar_elementos_carpeta(carpeta):
    ruta_carpeta = os.path.join(base_dir, carpeta)
    return len(os.listdir(ruta_carpeta))

# Encontrar la carpeta con el menor número de elementos
min_carpetas = min(carpetas, key=contar_elementos_carpeta)

print(f"La carpeta con el menor número de elementos es {min_carpetas}, con {contar_elementos_carpeta(min_carpetas)} elementos.")
'''Debemos fijarnos en el número de elementos que tiene la carpeta con menos imágenes, puesto que el modelo debe ser entrenado con el mismo número de imágenes
para todas las categorías'''
plt.figure(figsize=(15,15))

carpeta = '/content/ccotorra_argentina'
imagenes = os.listdir(carpeta)
# Es recomendable visualizar un par de imágenes de nuestras carpetas para comprobar que tienen tamaños aceptables y, sobretodo, que se visualizan correctamente

for i, nombreimg in enumerate(imagenes[:25]):
  plt.subplot(5,5,i+1)
  imagen = mpimg.imread(carpeta + '/' + nombreimg)
  plt.imshow(imagen)
  #Crear carpetas para hacer el set de datos con el que entrenaremos nuestro modelo

!mkdir dataset
!mkdir dataset/ccotorra_argentina
!mkdir dataset/cmina_comun
!mkdir dataset/cescribano_palustre
!mkdir dataset/cmilano_real
!mkdir dataset/ctarro_canelo
# Esta función nos facilita que todos los datasets tengan el mismo número de fotos (recuerda cambiar el número cuando utilices un repositorio distinto al mío)
carpeta_fuente = '/content/ccotorra_argentina'
carpeta_destino = '/content/dataset/ccotorra_argentina'

imagenes = os.listdir(carpeta_fuente)

for i, nombreimg in enumerate(imagenes):
  if i < 1102:
    #Copia de la carpeta fuente a la destino
    shutil.copy(carpeta_fuente + '/' + nombreimg, carpeta_destino + '/' + nombreimg)
    carpeta_fuente = '/content/cescribano_palustre'
carpeta_destino = '/content/dataset/cescribano_palustre'

imagenes = os.listdir(carpeta_fuente)

for i, nombreimg in enumerate(imagenes):
  if i < 1102:
    #Copia de la carpeta fuente a la destino
    shutil.copy(carpeta_fuente + '/' + nombreimg, carpeta_destino + '/' + nombreimg)

carpeta_fuente = '/content/cmilano_real'
carpeta_destino = '/content/dataset/cmilano_real'

imagenes = os.listdir(carpeta_fuente)

for i, nombreimg in enumerate(imagenes):
  if i < 1102:
    #Copia de la carpeta fuente a la destino
    shutil.copy(carpeta_fuente + '/' + nombreimg, carpeta_destino + '/' + nombreimg)

carpeta_fuente = '/content/ctarro_canelo'
carpeta_destino = '/content/dataset/ctarro_canelo'

imagenes = os.listdir(carpeta_fuente)

for i, nombreimg in enumerate(imagenes):
  if i < 1102:
    #Copia de la carpeta fuente a la destino
    shutil.copy(carpeta_fuente + '/' + nombreimg, carpeta_destino + '/' + nombreimg)

    carpeta_fuente = '/content/cmina_comun'
carpeta_destino = '/content/dataset/cmina_comun'

imagenes = os.listdir(carpeta_fuente)

for i, nombreimg in enumerate(imagenes):
  if i < 1102:
    #Copia de la carpeta fuente a la destino
    shutil.copy(carpeta_fuente + '/' + nombreimg, carpeta_destino + '/' + nombreimg)

#Recomendable observar el número total de valores en cada dataset y comprobar que está todo bien
!ls /content/dataset/ccotorra_argentina | wc -l
!ls /content/dataset/cmina_comun | wc -l
!ls /content/dataset/cmilano_real | wc -l
!ls /content/dataset/ctarro_canelo | wc -l
!ls /content/dataset/cescribano_palustre | wc -l
'''Con esto ya hemos terminado de definir nuestros repositorios, ahora toca empezar el preprocesado para poder entrenar posteriormente nuestro modelo'''
#Aumento de datos con ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

#Crear el dataset generador
datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range = 30,
    width_shift_range = 0.25,
    height_shift_range = 0.25,
    shear_range = 15,
    zoom_range = [0.5, 1.5],
    validation_split=0.2 #20% para pruebas
)

#Generadores para sets de entrenamiento y pruebas
data_gen_entrenamiento = datagen.flow_from_directory('/content/dataset', target_size=(224,224),
                                                     batch_size=32, shuffle=True, subset='training')
data_gen_pruebas = datagen.flow_from_directory('/content/dataset', target_size=(224,224),
                                                     batch_size=32, shuffle=True, subset='validation')

#Imprimir 10 imagenes del generador de entrenamiento
for imagen, etiqueta in data_gen_entrenamiento:
  for i in range(10):
    plt.subplot(2,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(imagen[i])
  break
plt.show()
'''En nuestro caso, hemos decidido importar mobilenet_v2. Feature vector hace mención a que es el modelo sin la capa de salida, para que podamos ajustar las salidas
que deseemos. Recordar que mobilenetv2 es extremadamente preciso y eficaz, lo que nos permite obtener unos resultados impecables con muy pocos datos. Sin embargo, no
es recomendable a la hora de productivizar un modelo, puesto que pesa mucho y es difícilmente exportable a html. Se recomienda en ese caso utilizar modelos algo menos eficaces pero 
más optimizados'''
import tensorflow as tf
import tensorflow_hub as hub

url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
mobilenetv2 = hub.KerasLayer(url, input_shape=(224,224,3))
#Congelar el modelo descargado
mobilenetv2.trainable = False

'''Ajustamos el modelo a las características predeterminadas que se encuentran en tensorhub, puesto que estas maximizan sus buenos resultados'''
modelo = tf.keras.Sequential([
    mobilenetv2,
    tf.keras.layers.Dense(5, activation='softmax')
])

#Compilar como siempre
modelo.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
) 
'''Finalmente, nuestro modelo está listo para ser entrenado. Aunque los resultados empiezan a estancarse a partir de las 20 vueltas, recomiendo dar por lo menos 40,
especialmente debido a que con la gpu de collab el proceso se acorta mucho  e incrementar el número de vueltas no nos supone una dilación excesiva'''

EPOCAS = 50

try:
    historial = modelo.fit(
        data_gen_entrenamiento, epochs=EPOCAS, batch_size=32,
        validation_data=data_gen_pruebas,
        
    )
except Exception as e:
    print(f"Error durante el entrenamiento debido a una imagen dañada: {str(e)}")
    print(f"Nombre de la imagen dañada: {data_gen_entrenamiento.filenames[data_gen_entrenamiento.batch_index]}")
    print(f"Carpeta de la imagen dañada: {data_gen_entrenamiento.directory}")

'''Podemos utilizar estas gráficas para realizar un seguimiento básico del modelo, sobretodo a modo de comprobación. Podemos en todo caso utilizar la función callbacks
y utilizar tensorboard si queremos un seguimiento aún más detallado'''

#Graficas de precisión
acc = historial.history['accuracy']
val_acc = historial.history['val_accuracy']

loss = historial.history['loss']
val_loss = historial.history['val_loss']

rango_epocas = range(50)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(rango_epocas, acc, label='Precisión Entrenamiento')
plt.plot(rango_epocas, val_acc, label='Precisión Pruebas')
plt.legend(loc='lower right')
plt.title('Precisión de entrenamiento y pruebas')

plt.subplot(1,2,2)
plt.plot(rango_epocas, loss, label='Pérdida de entrenamiento')
plt.plot(rango_epocas, val_loss, label='Pérdida de pruebas')
plt.legend(loc='upper right')
plt.title('Pérdida de entrenamiento y pruebas')
plt.show()

from google.colab import files
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

def categorizar(entrada):
    if entrada.startswith('http://') or entrada.startswith('https://'):
        # Si es una URL, descarga la imagen y guárdala localmente
        respuesta = requests.get(entrada)
        if respuesta.status_code == 200:
            img = Image.open(BytesIO(respuesta.content))
        else:
            print("No se pudo descargar la imagen desde la URL.")
            return None
    else:
        # Si no es una URL, intenta cargarla desde el entorno de Colab
        try:
            uploaded_files = files.upload()
            # Suponemos que el nombre del archivo cargado es el mismo que proporcionaste en 'entrada'
            file_name = list(uploaded_files.keys())[0]
            img = Image.open(BytesIO(uploaded_files[file_name]))
        except Exception as e:
            print("No se pudo cargar la imagen desde Colab:", e)
            return None

    img = np.array(img).astype(float) / 255
    img = cv2.resize(img, (224, 224))
    prediccion = modelo.predict(img.reshape(-1, 224, 224, 3))
    return np.argmax(prediccion[0], axis=-1)

# 0 cotorra_argentina 1 escribano palustre 2 milano real 3 mina comun 4 tarro canelo
url = 'https://th.bing.com/th/id/OIP.13RCltj9YajYsQAmt9-QEQHaFA?pid=ImgDet&rs=1' #debe ser 2
prediccion = categorizar (url)
print(prediccion)
  
'''Tras probarlo un poco y verificar que funciona correctamente, podemos guardarlo utilizando esta función:'''
from google.colab import files
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
#Crear la carpeta para exportarla a TF Serving
!mkdir -p carpeta_salida/modelo_pajaros/1
#Guardar el modelo en formato SavedModel
modelo.save('carpeta_salida/modelo_pajaros.h5')

'''Finalmente, ya estamos listos para utilizar el modelo cuando lo deseemos'''
# Primero, lo descargamos cuando lo necesitemos
from google.colab import files

# Descarga el modelo desde la carpeta "carpeta_salida" en tu sistema local
files.download('modelo_pajaros.h5')

'''Y podemos utilizar esto como una versión más sofisticada para utilizarlo. Por supuesto cualquier salida es libremente modificable'''
import cv2
import numpy as np
from google.colab import files
from IPython.display import display
from ipywidgets import FileUpload, Output
import tensorflow as tf

# Cargar el modelo desde el archivo .h5
model = tf.keras.models.load_model('modelo_pajaros.h5')
print("Bienvenido al servicio de protección de flora y fauna. Por favor, introduzca un archivo de la especie en cuestión.")


def cargar_imagen(change):
    # Obtener el archivo cargado
    uploaded_file = list(change.new.values())[0]
    
    # Leer la imagen y realizar la predicción
    image = cv2.imdecode(np.frombuffer(uploaded_file['content'], dtype=np.uint8), -1)
    if image is not None:
        # Redimensionar la imagen al tamaño esperado por el modelo (224x224)
        input_image = cv2.resize(image, (224, 224))

        # Convertir la imagen en un tensor y normalizarla si es necesario
        input_image = np.array(input_image).astype(float) / 255.0

        # Realizar la predicción
        predictions = model.predict(np.expand_dims(input_image, axis=0))

        # Obtener la clase con la probabilidad más alta
        predicted_class = np.argmax(predictions[0])

        # Mostrar el resultado según la clase predicha
        with output:
            if predicted_class == 0:
                print("La especie ha sido identificada como Myiopsitta monachus, coloquialmente conocida como 'cotorra argentina'. Este ave es una especie invasora en la zona, se recomienda ponerse en contacto con el personal especializado en fauna invasora con la mayor brevedad posible, debido a su alto potencial reproductivo.")
            elif predicted_class == 1:
                print("La especie ha sido identificada como Emberiza schoeniclus, coloquialmente conocida como 'escribano' o 'escribano palustre'. Esta especie está amenazada, principalmente debido a los riesgos derivados del calor generado en verano, que induce a estos animales a la deshidratación. Se recomienda facilitar una fuente de agua sin realizar contacto con el animal. Finalmente, si el animal está herido, contactar con el número de teléfono 6...5....2....3....1.")
            elif predicted_class == 2:
                print("La especie ha sido identificada como Milvus milvus, coloquialmente conocida como 'milano', 'milana' o 'milano real'. Esta especie está amenazada, principalmente debido a la escasez de presas en multitud de entornos. Se debe extremar la precaución al acercarse a ellos, ya que podrían llegar a ser agresivos si se sienten acorralados. Finalmente, si el animal está herido, contactar con el número de teléfono 6...5....2....3....1.")
            elif predicted_class == 3:
                print("La especie ha sido identificada como Acridotheres tristis, coloquialmente conocida como 'miná' o 'miná común'. Es una especie invasora procedente del Medio Oriente, concretamente de la región de la península arábica. Es muy importante ponerse en contacto con las autoridades especializadas en especies invasoras. Es importante no abandonar la zona hasta la posterior intervención de las autoridades especializadas, ya que estas aves son muy territoriales y extremadamente agresivas con las aves locales de tamaño similar, pudiendo exterminar grupos en horas.")
            elif predicted_class == 4:
                print("La especie ha sido identificada como Tadorna ferruginea, coloquialmente conocida como 'tarro canelo'. Es una especie invasora procedente de América y las Islas Canarias. Aunque es un potencial riesgo para el ecosistema, no se ha registrado una gran influencia por su parte. Sin embargo, suele representar un problema para los agricultores, que suelen formular multitud de reportes de que estas aves se comen sus semillas. Se recomienda llamar a la comitiva agraria estatal que le asignará el número de teléfono 9..3...4.2...5.")
    else:
        with output:
            print("No se pudo cargar la imagen.")

# Crear un widget para cargar archivos
uploader = FileUpload()
output = Output()

# Asociar la función de carga con el evento de cambio de archivo
uploader.observe(cargar_imagen, names='value')

# Mostrar el widget
display(uploader, output)