import os
import numpy as np
import cv2
from PIL import Image as im
img_dict = {}

def adicionar_especie(especie,classificacao,path,img_dict):
# pegar a imagem e transformar pra GREYSCALE
    for img_path in os.listdir(path):
        if img_path.endswith('.jpeg') or img_path.endswith('.png') or img_path.endswith('.jpg'):
            image = im.open(img_path)
            image = image.convert('L')
            image = np.array(image)
            image = cv2.resize(image, (150,150))
            image = map(lambda pixel: pixel/255,image)
            img_dict[classificacao][especie] = image

def main():
    adicionar_especie('urutu','peconhenta', "database/peconhentas/bothrups-alternatus",img_dict)
    adicionar_especie('cascavel','peconhenta',"database/peconhentas/Cascavel (Crotalus durissus)",img_dict)
    adicionar_especie('jararaca','peconhenta',"database/peconhentas/Jararaca (Bothrops jararaca)_Venomous",img_dict)
    adicionar_especie('coral-verdadeira','peconhenta', "database/peconhentas/Jararacuçu (Bothrops jararacussu)",img_dict)
    adicionar_especie('cobra-verde','peconhenta', "database/peconhentas/micrurus-frontalis",img_dict)
    adicionar_especie('cobra-dagua','peconhenta',"database/peconhentas/Surucucu (Lachesis muta)",img_dict)
    adicionar_especie('jiboia','peconhenta',"database/nao-peconhentas/Jiboia(Boa constrictor)",img_dict)
    adicionar_especie('cobra-cega','peconhenta',"database/nao-peconhentas/typhlops-brongersmianus",img_dict)

main()