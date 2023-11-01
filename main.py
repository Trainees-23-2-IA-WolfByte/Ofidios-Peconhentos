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


imgs = []

def adicionar_especie(especie,classificacao,path,imgs):
    valid_images = ['.jpg','.png','.jpeg']
    for img_name in os.listdir(path):
        if img_name.endswith(valid_images):
            imgs.append(img.open(os.path.join(path,img_name)))


def main():
    adicionar_especie('urutu','peconhenta','/database/peconhentas/bothrups-alternatus',imgs)
    adicionar_especie('cascavel','peconhenta','/database/peconhentas/Cascavel (Crotalus durissus)',imgs)
    adicionar_especie('jararaca','peconhenta','/database/peconhentas/Jararaca (Bothrops jararaca)_Venomous',imgs)
    adicionar_especie('coral-verdadeira','peconhenta','/database/database/peconhentas/micrurus-frontalis',imgs)
    adicionar_especie('cobra-verde','peconhenta','/database/nao-peconhentas/Cobra Verde(Philodryas ofersii)',imgs)
    adicionar_especie('cobra-dagua','peconhenta','/database/nao-peconhentas/helicops-danieli',imgs)
    adicionar_especie('jiboia','peconhenta','/database/nao-peconhentas/Jiboia(Boa constrictor)',imgs)
    adicionar_especie('cobra-cega','peconhenta','/database/nao-peconhentas/typhlops-brongersmianus',imgs)
    imgs[0].show()
main()


imgs = [cv2.resize(img, (150, 150)) for img in imgs]
imgs = np.array(imgs)

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

