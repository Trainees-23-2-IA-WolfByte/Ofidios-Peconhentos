import os
import numpy as np
import cv2
from PIL import Image as im

def adicionar_especie(especie,classificacao,path,img_dict):
    for img_path in os.listdir(path):
        if img_path.endswith('.jpeg') or img_path.endswith('.png') or img_path.endswith('.jpg'):
            image = im.open(img_path)  # abre a imagem
            image = image.convert('L') # transforma greyscale
            image = np.array(image) # transforma em array de pixels ( 0 a 255 )
            image = cv2.resize(image, (150,150))  # redimensiona a imagem ( 150 x 150 ) 
            image = map(lambda pixel: pixel/255,image) # transforma os pixels em um numero de 0 a 1
            img_dict[classificacao][especie] = image # labels

def main():
    img_dict = {}
    adicionar_especie('surucucu','peconhenta', "database/peconhentas/Surucucu (Lachesis muta)",img_dict)
    adicionar_especie('jararacucu','peconhenta', "database/peconhentas/Jararacuçu (Bothrops jararacussu)",img_dict)
    adicionar_especie('urutu','peconhenta', "database/peconhentas/bothrups-alternatus",img_dict)
    adicionar_especie('cascavel','peconhenta',"database/peconhentas/Cascavel (Crotalus durissus)",img_dict)
    adicionar_especie('jararaca','peconhenta',"database/peconhentas/Jararaca (Bothrops jararaca)_Venomous",img_dict)
    adicionar_especie('coral-verdadeira','peconhenta', "database/peconhentas/micrurus-frontalis",img_dict)
    adicionar_especie('cobra-verde','nao_peconhenta', "database/nao_peconhentas/Cobra Verde(Philodryas ofersii)",img_dict)
    adicionar_especie('cobra-dagua','nao_peconhenta',"database/nao_peconhentas/helicops-danieli",img_dict)
    adicionar_especie('jiboia','nao_peconhenta',"database/nao-peconhentas/Jiboia(Boa constrictor)",img_dict)
    adicionar_especie('cobra-cega','nao_peconhenta',"database/nao-peconhentas/typhlops-brongersmianus",img_dict)
    adicionar_especie('cobra-cipo','nao_peconhenta',"database/nao-peconhentas/Cobra-Cipó",img_dict)
    adicionar_especie('cobra-de-vidro','nao_peconhenta',"database/nao-peconhentas/Cobra-de-Vidro",img_dict)

main()
