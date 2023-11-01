import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
import cv2
import numpy as np
from PIL import Image as img
import numpy as np
from matplotlib import pyplot as plt
os.path.join("Ofídios-Peconhentos")
imgs = []

def adicionar_especie(especie,classificacao,path,imgs):
    for img_name in os.listdir(path):
        if img_name.endswith('.jpeg'):
            imgs.append(img.open(os.path.join(path,img_name)))
        elif img_name.endswith('.png'):
            imgs.append(img.open(os.path.join(path,img_name)))
        elif img_name.endswith('.jpg'):
            imgs.append(img.open(os.path.join(path,img_name)))
        else: print("error")

def main():
    adicionar_especie('urutu','peconhenta', "database/peconhentas/bothrups-alternatus",imgs)
    adicionar_especie('cascavel','peconhenta',"database/peconhentas/Cascavel (Crotalus durissus)",imgs)
    adicionar_especie('jararaca','peconhenta',"database/peconhentas/Jararaca (Bothrops jararaca)_Venomous",imgs)
    adicionar_especie('coral-verdadeira','peconhenta', "database/peconhentas/Jararacuçu (Bothrops jararacussu)",imgs)
    adicionar_especie('cobra-verde','peconhenta', "database/peconhentas/micrurus-frontalis",imgs)
    adicionar_especie('cobra-dagua','peconhenta',"database/peconhentas/Surucucu (Lachesis muta)",imgs)
    adicionar_especie('jiboia','peconhenta',"database/nao-peconhentas/Jiboia(Boa constrictor)",imgs)
    adicionar_especie('cobra-cega','peconhenta',"database/nao-peconhentas/typhlops-brongersmianus",imgs)
    imgs[0].show()

main()
pic1 = imgs[0]
pic1_array = np.array(pic1)
pic1_array = cv2.resize(pic1_array, (150, 150))
pic1 = img.fromarray(pic1_array)
pic1.show()




def load_and_preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.resize(image, target_size)
        image = image / 255.0  # Normalize the pixel values
        return image
    else:
        return None
image_path = "path/to/your/image.jpg"
target_size = (150, 150)
processed_image = load_and_preprocess_image(image_path, target_size)

if processed_image is not None:
    cv2.imshow("Preprocessed Image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Image not found or unable to load.")

