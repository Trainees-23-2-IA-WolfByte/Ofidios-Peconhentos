import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras

# IGNORA codigo pra evitar erros de gpu no tensorflow
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)


def main():
    dataset = criar_dataset() # LISTA GIGANTE COM TODAS IMAGENS E LABELS
    dataset = pre_processamento(dataset)

    dataset_np = dataset.as_numpy_iterator() # divide o dataset em 32 imagens por grupo(pipeline), cada imagem um array, e um label.
    data = dataset_np.next() # vai pro prox grupo de 32 imagens do pipeline.

    # data eh uma lista, data[0] - lista com as imagens do pipeline atual, data[1] - lista com os labels do pipeline atual
    # por exemplo: data[0][0] é a primeira imagem. data[1][0] é o label da primeira imagem.
    test_plot(data,3)
    # relaxa a cor ta certa, o matplot que é meio doido.
    dataset_treino, dataset_val, dataset_test = separar_dados(dataset)


def criar_dataset():
    # sim o keras faz tudo
    dataset = tf.keras.utils.image_dataset_from_directory(
        "database", # pasta
        labels = 'inferred', # pastas em ordem alfabetica
        label_mode = 'int', # 0 = nao peconhenta, 1 = peconhenta
                #label_mode = 'categorical', futuramente com as especies.
        color_mode = 'grayscale',
        image_size = (150,150), # redimensiona
        shuffle = True, # embaralha
        batch_size = 32, # 32 imagens por grupo de treino!
        #seed = 123, # so pra salvar a ordem do embaralhamento
    )

    return dataset

def pre_processamento(dataset):
    dataset = dataset.map(lambda x,y: (x/255,y)) # dividindo o pixel por 255, pra ficar entre 0 e 1. e mantendo o y(label) intacto.
    # falta entender a parte la de convolucao e max pooling
    return dataset


def test_plot(data,n): # funcao so pra ver se ta funcionando
    # as cores vao estar erradas ,mas no dataset ta certo.
    img, axis = plt.subplots(ncols = n, figsize = (10,10))
    for i,img in enumerate(data[0][:n]):
        label = data[1][i]
        axis[i].imshow(img)
        axis[i].title.set_text(label)
    plt.show()
    #0 NAO PECONHENTA , 1 PECONHENTA


def separar_dados(dataset):
    train_size = int(len(dataset)*.8)
    validation_size = int(len(dataset)*.15)
    test_size = int(len(dataset)*.05)
    #test é o unico PÓS treino.
    #train_size + val_size + test_size = dataset size

    train = dataset.take(train_size) # QUANTOS GRUPOS DE IMAGENS (PARTES DO PIPELINE) vao para o treino
    validation = dataset.skip(train_size).take(validation_size) # ignorar as que foram usadas no treino, e pegar as pra validacao cruzada.
    test = dataset.skip(train_size+validation_size).take(test_size)
    
    return train,validation,test



main()
