import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image

class Dados():
    def __init__(self,directory_path):
        # criar um conjunto de dados de imagens a partir de um diretório
        self.dataset = tf.keras.utils.image_dataset_from_directory(
            directory_path,  # diretório contendo as imagens
            labels='inferred',  # inferência dos rótulos a partir das pastas
            label_mode='int',  # rótulos representados como inteiros
            color_mode='rgb',  # imagens em escala de cinza
            image_size=(150, 150),  # redimensionamento das imagens
            shuffle=True, # embaralha as imagens
            seed=15245, # criar sempre com a mesma config
            batch_size=32,  # tamanho do lote de imagens
        )
        self.escalonar()
        self.dataset_treino, self.dataset_validacao, self.dataset_teste = self.separar_dados()

    # função para pré-processar o conjunto de dados
    def escalonar(self):
        self.dataset = self.dataset.map(lambda x, y: (x / 255, y))  # normaliza as imagens, pixels entre 0 e 1

    # função para separar os conjuntos de dados em treino, validação e teste
    def separar_dados(self):
        dataset_size = len(list(self.dataset))
        treino_size = int(dataset_size * 0.8)
        validacao_size = int(dataset_size * 0.15)
        teste_size = int(dataset_size - treino_size - validacao_size)

        treino = self.dataset.take(treino_size)
        validacao = self.dataset.skip(treino_size).take(validacao_size)
        test = self.dataset.skip(treino_size + validacao_size).take(teste_size)

        return treino, validacao, test

class Novo_Modelo():
    def __init__(self,dataset_treino,dataset_validacao,dataset_teste):
        self.dataset_treino = dataset_treino
        self.dataset_teste = dataset_teste
        self.dataset_validacao = dataset_validacao
        # criação do modelo Sequential para classificação de serpentes peçonhentas
        self.model = Sequential([
            Conv2D(32, (3, 3),1, activation='relu', input_shape=(150,150,3)),
            MaxPooling2D(),
            Conv2D(64, (3, 3),1, activation='relu'),
            MaxPooling2D(),
            Conv2D(32, (3, 3),1, activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(150, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.setup_model()

    def setup_model(self):
        # compilação do modelo com parâmetros adequados para treinamento
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # treinamento do modelo com os conjuntos de dados
        self.model.fit(self.dataset_treino, validation_data=self.dataset_validacao, epochs=12)
        self.model.save(os.path.join('Modelos','modelo_teste.h5'))
