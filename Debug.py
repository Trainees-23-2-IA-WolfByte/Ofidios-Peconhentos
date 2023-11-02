from tensorflow.keras.metrics import Precision, BinaryAccuracy
from tensorflow.keras.preprocessing import image
import numpy as np
#from matplotlib import pyplot as plt

# funçao para testar a batch reservada pelo dataset.
def batch_test(modelo,dataset_teste):
    precision = Precision()
    accuracy = BinaryAccuracy()
    for batch_teste in dataset_teste.as_numpy_iterator():
        x,y = batch_teste
        predictions = modelo.predict(x)
        precision.update_state(y,predictions)
        accuracy.update_state(y,predictions)
        
    print(f'''
    Precisão: {precision.result().numpy()},
    Acurácia: {accuracy.result().numpy()}''')

# funcao para testar uma nova imagem nunca vista antes pelo modelo.
def image_test(modelo,image_path,label):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255
    prediction = modelo.predict(img_array)
    print(f'A previsão foi: {prediction}\n')
    if prediction < 0.5:
        print("Não é uma serpente peçonhenta.\n")
        resposta = 'peconhenta' 
        if resposta == label:
            print("A i.a acertou!\n")
        else:
            print("A i.a errou!\n")
    else:
        print("É uma serpente peçonhenta.\n")
        resposta = 'nao-peconhenta'
        if resposta == label:
            print("A i.a acertou!\n")
        else:
            print("A i.a errou!\n")
    

        
'''
# função para plotar imagens do conjunto de dados
def test_plot(data, n):
    img, axis = plt.subplots(ncols=n, figsize=(10, 10))
    for i, img in enumerate(data[0][:n]):
        label = data[1][i]
        axis[i].imshow(img)
        axis[i].title.set_text(label)
'''