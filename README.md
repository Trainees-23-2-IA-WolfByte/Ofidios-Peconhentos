# Reconhecimento de Ofídio Peçonhento (Visão Computacional)
Este é um projeto desenvolvido por candidatos do processo seletivo 23.2 para a Equipe WolfByte-IA do Ramo Estudantil IEEE CEFET-RJ.

Usando técnicas de visão computacional, o objetivo principal é desenvolver e treinar um modelo de inteligência artificial capaz de determinar se uma serpente é ou não peçonhenta com base em uma imagem.

## Funcionalidades

- [X] Desenvolvimento de um conjunto de dados com 4 a 10 diferentes espécies de cobras, sendo metade peçonhenta e metade não peçonhenta.
- [ ] Treinamento de um modelo de inteligência artificial para classificação das serpentes com uma precisão mínima de 75%.
- [ ] Implementação de uma função que recebe uma imagem de um ofídio e retorna se ele é peçonhento ou não.

## Bônus (Opcional)

- [ ] Generalização do modelo para ser capaz de classificar qualquer cobra como peçonhenta ou não com base em características fundamentais.

## Exemplo de Uso

```python
# Carregar o modelo treinado
classificador = ofidio_peconhento("modelo_final")

# Classificar uma imagem
resultado = classificador.classificar("serpente.jpg")
print(resultado)
# Peçonhenta! / Não peçonhenta!
```

## Técnico

- Python, Jupyter Notebook
- Google Colab
- Bibliotecas (numpy, tensorflow, keras, scikit-learn)

  ### Boas Práticas de Programação
  Todo código deve seguir os padrões de boas práticas, como os ensinados no livro "Clean Code" - Robert C. Martin. Isso inclui:
  Nomes de variáveis, classes e funções devem ser descritivos e auto-explicativos.
  Funções devem ter um propósito claro e realizar uma única tarefa.
  Evitar códigos duplicados através da criação de funções e classes reutilizáveis.

## Equipe Desenvolvedora Trainee

- [Daniel Lanzillotta Serodio](https://github.com/dlseodio)
- [Caio Passos Torkst Ferreira](https://github.com/stepsbtw)
- Guilherme Soares Vieira

### Membros envolvidos no treinamento

- **Dannylo** : *Líder da Equipe I.A - WolfByte* : Assessor Técnico do Ramo
- **Maria Luisa** - *Líder do Processo Seletivo IEEE* : Assessor de Gestão do Ramo

---

Para mais informações, consulte : https://github.com/dlserodio/Ofidios-Peconhentos.
