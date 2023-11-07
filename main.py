import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from Classes import Dados
from Classes import Novo_Modelo
import Debug

# configura a alocação de memória para as GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# função principal que controla o fluxo do programa
def main():
    DADOS = Dados("database") # importar dados

    #criar novo modelo, treiná-lo e salvar:
    #Novo_Modelo(DADOS.dataset_treino,DADOS.dataset_validacao,DADOS.dataset_teste)
    
    MODELO = load_model(os.path.join('Modelos','modelo_teste.h5')) #importar o modelo treinado
    Debug.batch_test(MODELO, DADOS.dataset_teste) # USANDO OS BATCHS DE TESTE
    Debug.image_test(MODELO, "TESTE/dormedorme1.jpg",label='peconhenta') # USANDO IMAGEM DE TESTE
    Debug.image_test(MODELO, "TESTE/dormedorme2.jpg",label='peconhenta')
    Debug.image_test(MODELO, "TESTE/dormedorme3.jpeg",label='peconhenta')
    Debug.image_test(MODELO, "TESTE/dormedorme4.jpg",label='peconhenta')
    Debug.image_test(MODELO, "TESTE/cobrareal.jpg",label='peconhenta')
    Debug.image_test(MODELO, "TESTE/cobrareal2.jpg",label='peconhenta')
    Debug.image_test(MODELO, "TESTE/king.jpg",label='peconhenta')
    Debug.image_test(MODELO, "TESTE/teste.png",label='peconhenta')


main()