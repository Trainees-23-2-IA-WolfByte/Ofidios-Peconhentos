import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model

# configura a alocação de memória para as GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# função para criar um conjunto de dados de imagens a partir de um diretório
def criar_dataset():
    dataset = tf.keras.utils.image_dataset_from_directory(
        "database",  # diretório contendo as imagens
        labels='inferred',  # inferência dos rótulos a partir das pastas
        label_mode='int',  # rótulos representados como inteiros
        color_mode='rgb',  # imagens em escala de cinza
        image_size=(150, 150),  # redimensionamento das imagens
        shuffle=True,  # embaralha as imagens
        batch_size=32,  # tamanho do lote de imagens
    )
    return dataset

# função para pré-processar o conjunto de dados
def pre_processamento(dataset):
    dataset = dataset.map(lambda x, y: (x / 255, y))  # normaliza as imagens
    return dataset

# função para plotar imagens do conjunto de dados
def test_plot(data, n):
    img, axis = plt.subplots(ncols=n, figsize=(10, 10))
    for i, img in enumerate(data[0][:n]):
        label = data[1][i]
        axis[i].imshow(img)
        axis[i].title.set_text(label)

# função para separar os conjuntos de dados em treino, validação e teste
def separar_dados(dataset):
    dataset_size = len(list(dataset))
    train_size = int(dataset_size * 0.8)
    validation_size = int(dataset_size * 0.15)
    test_size = dataset_size - train_size - validation_size

    train = dataset.take(train_size)
    validation = dataset.skip(train_size).take(validation_size)
    test = dataset.skip(train_size + validation_size).take(test_size)

    return train, validation, test

# função principal que controla o fluxo do programa
def main():
    dataset = criar_dataset()  # cria o conjunto de dados
    dataset = pre_processamento(dataset)  # pré-processa o conjunto de dados

    dataset_np = dataset.as_numpy_iterator()
    data = dataset_np.next()

    test_plot(data, 3)  # exibe algumas imagens do conjunto de dados
    dataset_treino, dataset_val, dataset_test = separar_dados(dataset)  # separa o conjunto de dados

    # criação do modelo Sequential para classificação de serpentes peçonhentas
    
    model = Sequential([
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
    

    # compilação do modelo com parâmetros adequados para treinamento
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # treinamento do modelo com os conjuntos de dados
    model.fit(dataset_treino, validation_data=dataset_val, epochs=12)

    print(model.summary())

    model.save(os.path.join('modelos')('modelo1.h5'))

    precision = Precision()
    recall = Recall()
    accuracy = BinaryAccuracy()
    for test_batch in dataset_test.as_numpy_iterator():
        x,y = test_batch
        predictions = model.predict(x)
        precision.update_state(y,predictions)
        recall.update_state(y,predictions)
        accuracy.update_state(y,predictions)

    print(f'''
        Precisão: {precision.result().numpy()},
        Acurácia: {accuracy.result().numpy()}
        '''
    )
    
    img_path = 'teste.png'
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    print(img_array.shape)
    img_array = img_array / 255
    prediction = model.predict(img_array)
    print(prediction)
    if prediction < 0.5:
        print("Não é uma serpente peçonhenta.")
    else:
        print("É uma serpente peçonhenta.")


main()