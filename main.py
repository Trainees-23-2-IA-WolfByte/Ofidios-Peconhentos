import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Configura a alocação de memória para as GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Função para criar um conjunto de dados de imagens a partir de um diretório
def criar_dataset():
    dataset = tf.keras.utils.image_dataset_from_directory(
        "database",  # diretório contendo as imagens
        labels='inferred',  # inferência dos rótulos a partir das pastas
        label_mode='int',  # rótulos representados como inteiros
        color_mode='grayscale',  # imagens em escala de cinza
        image_size=(150, 150),  # redimensionamento das imagens
        shuffle=True,  # embaralha as imagens
        batch_size=32,  # tamanho do lote de imagens
    )
    return dataset

# Função para pré-processar o conjunto de dados
def pre_processamento(dataset):
    dataset = dataset.map(lambda x, y: (x / 255, y))  # normaliza as imagens
    return dataset

# Função para plotar imagens do conjunto de dados
def test_plot(data, n):
    img, axis = plt.subplots(ncols=n, figsize=(10, 10))
    for i, img in enumerate(data[0][:n]):
        label = data[1][i]
        axis[i].imshow(img)
        axis[i].title.set_text(label)
    plt.show()

# Função para separar os conjuntos de dados em treino, validação e teste
def separar_dados(dataset):
    dataset_size = len(list(dataset))
    train_size = int(dataset_size * 0.8)
    validation_size = int(dataset_size * 0.15)
    test_size = dataset_size - train_size - validation_size

    train = dataset.take(train_size)  # conjunto de treinamento
    validation = dataset.skip(train_size).take(validation_size)  # conjunto de validação
    test = dataset.skip(train_size + validation_size).take(test_size)  # conjunto de teste

    return train, validation, test

# Função principal que controla o fluxo do programa
def main():
    dataset = criar_dataset()  # cria o conjunto de dados
    dataset = pre_processamento(dataset)  # pré-processa o conjunto de dados

    dataset_np = dataset.as_numpy_iterator()
    data = dataset_np.next()

    test_plot(data, 3)  # exibe algumas imagens do conjunto de dados
    dataset_treino, dataset_val, dataset_test = separar_dados(dataset)  # separa o conjunto de dados

main()
