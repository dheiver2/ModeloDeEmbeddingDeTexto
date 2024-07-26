# ModeloDeEmbeddingDeTexto

## Descrição

O `ModeloDeEmbeddingDeTexto` é uma classe para processar, treinar e manipular textos usando técnicas de TF-IDF e Word2Vec. Esta classe permite transformar textos em vetores, treinar um modelo de Word2Vec com esses textos, salvar e carregar o modelo treinado, obter vetores de palavras específicas e gerar novos textos a partir de uma palavra inicial.

## Funcionalidades

- **Preprocessamento de texto**: Converte texto para minúsculas e tokeniza.
- **TF-IDF**:
  - Ajuste e transformação de textos usando TF-IDF.
- **Word2Vec**:
  - Treinamento do modelo Word2Vec.
  - Salvamento e carregamento do modelo Word2Vec.
  - Obtenção de vetores de palavras específicas.
  - Geração de texto a partir de uma palavra inicial.
- **Correção ortográfica**: Correção de palavras usando um corretor ortográfico.
- **Contagem de parâmetros**: Contagem do número total de parâmetros do modelo Word2Vec.

## Requisitos

- Python 3.x
- Bibliotecas: `numpy`, `spellchecker`, `sklearn`, `gensim`, `pandas`

## Instalação

1. Clone o repositório ou copie o código para o seu ambiente local.
2. Instale as bibliotecas necessárias:

```bash
pip install numpy spellchecker sklearn gensim pandas
```

## Exemplo de Uso

```python
import pandas as pd

# Dados de exemplo
dados = {
    'id': [0, 1, 2, 3, 4],
    'translation_text': [
        'No princípio criou Deus o céu e a terra.',
        'E disse Deus: Haja luz; e houve luz.',
        'E viu Deus que era boa a luz; e fez Deus a separação entre a luz e as trevas.',
        'E Deus chamou à luz Dia; e às trevas chamou Noite.',
        'E disse Deus: Haja uma expansão no meio das águas; e haja separação entre águas e águas.'
    ]
}

df = pd.DataFrame(dados)
textos = df['translation_text']

# Inicializa o modelo
modelo = ModeloDeEmbeddingDeTexto()

# Ajusta o TF-IDF
modelo.ajustar_tfidf(textos)
matriz_tfidf = modelo.transformar_tfidf(textos)
print("Formato da Matriz TF-IDF:", matriz_tfidf.shape)

# Treina o Word2Vec
modelo.treinar_word2vec(textos)
modelo.salvar_modelo_word2vec("word2vec.model")

# Carrega o modelo Word2Vec e gera texto
modelo.carregar_modelo_word2vec("word2vec.model")
texto_gerado = modelo.gerar_texto('deus', comprimento=5)
print("Texto gerado:", texto_gerado)

# Conta e exibe a quantidade de parâmetros
total_parametros = modelo.contar_parametros()
print("Quantidade total de parâmetros:", total_parametros)
```

## Métodos

### `__init__(self, tamanho_do_vetor=100, janela=5, contagem_minima=1)`
Inicializa a classe com parâmetros para o modelo Word2Vec.

### `preprocessar_texto(self, texto)`
Preprocessa o texto para tokenização.

### `ajustar_tfidf(self, textos)`
Ajusta o modelo TF-IDF aos textos fornecidos.

### `transformar_tfidf(self, textos)`
Transforma os textos usando o modelo TF-IDF ajustado.

### `treinar_word2vec(self, textos)`
Treina o modelo Word2Vec com os textos fornecidos.

### `salvar_modelo_word2vec(self, caminho)`
Salva o modelo Word2Vec em um arquivo especificado.

### `carregar_modelo_word2vec(self, caminho)`
Carrega o modelo Word2Vec a partir de um arquivo especificado.

### `obter_vetor_da_palavra(self, palavra)`
Obtém o vetor de uma palavra específica do modelo Word2Vec.

### `gerar_texto(self, palavra_inicial, comprimento=10)`
Gera um texto a partir de uma palavra inicial usando o modelo Word2Vec.

### `corrigir_ortografia(self, palavras)`
Corrige a ortografia das palavras usando o corretor ortográfico.

### `contar_parametros(self)`
Conta o número total de parâmetros do modelo Word2Vec.

## Contribuição

Sinta-se à vontade para contribuir com o projeto enviando issues e pull requests. Toda contribuição é bem-vinda!

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

---

Para dúvidas ou sugestões, entre em contato com [seu email].
