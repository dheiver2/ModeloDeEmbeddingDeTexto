
## Tutorial: Criando e Usando um Modelo de Embedding de Texto com TF-IDF e Word2Vec

### 1. Introdução

Neste tutorial, vamos explorar um modelo de embedding de texto que combina TF-IDF e Word2Vec para processamento e análise de texto. Você aprenderá como:

- Preprocessar textos
- Ajustar e transformar textos com TF-IDF
- Treinar e salvar um modelo Word2Vec
- Gerar texto a partir de um modelo Word2Vec
- Corrigir a ortografia das palavras geradas
- Contar o número de parâmetros do modelo Word2Vec

### 2. Preparação do Ambiente

Certifique-se de que você tem as seguintes bibliotecas instaladas:
- `numpy`
- `spellchecker`
- `sklearn`
- `gensim`
- `pandas`

Você pode instalar essas bibliotecas usando o comando pip:

```bash
pip install numpy spellchecker scikit-learn gensim pandas
```

### 3. Código do Modelo

Vamos usar a classe `ModeloDeEmbeddingDeTexto` que combina TF-IDF e Word2Vec. Aqui está o código:

```python
import numpy as np
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import pandas as pd

class ModeloDeEmbeddingDeTexto:
    def __init__(self, tamanho_do_vetor=100, janela=5, contagem_minima=1):
        self.tamanho_do_vetor = tamanho_do_vetor
        self.janela = janela
        self.contagem_minima = contagem_minima
        self.tfidf_vectorizer = TfidfVectorizer()
        self.word2vec_modelo = None
        self.spell_checker = SpellChecker(language='pt')

    def preprocessar_texto(self, texto):
        """Preprocessa o texto para tokenização."""
        texto = texto.lower()
        return simple_preprocess(texto)

    def ajustar_tfidf(self, textos):
        """Ajusta o modelo TF-IDF aos textos fornecidos."""
        self.tfidf_vectorizer.fit(textos)

    def transformar_tfidf(self, textos):
        """Transforma os textos usando o modelo TF-IDF ajustado."""
        return self.tfidf_vectorizer.transform(textos)

    def treinar_word2vec(self, textos):
        """Treina o modelo Word2Vec com os textos fornecidos."""
        textos_tokenizados = [self.preprocessar_texto(texto) for texto in textos]
        self.word2vec_modelo = Word2Vec(
            sentences=textos_tokenizados,
            vector_size=self.tamanho_do_vetor,
            window=self.janela,
            min_count=self.contagem_minima,
            workers=4
        )

    def salvar_modelo_word2vec(self, caminho):
        """Salva o modelo Word2Vec em um arquivo especificado."""
        if self.word2vec_modelo:
            self.word2vec_modelo.save(caminho)
        else:
            raise ValueError("O modelo Word2Vec ainda não foi treinado.")

    def carregar_modelo_word2vec(self, caminho):
        """Carrega o modelo Word2Vec a partir de um arquivo especificado."""
        self.word2vec_modelo = Word2Vec.load(caminho)

    def obter_vetor_da_palavra(self, palavra):
        """Obtém o vetor de uma palavra específica do modelo Word2Vec."""
        if self.word2vec_modelo:
            return self.word2vec_modelo.wv[palavra]
        else:
            raise ValueError("O modelo Word2Vec ainda não foi treinado.")

    def gerar_texto(self, palavra_inicial, comprimento=10):
        """Gera um texto a partir de uma palavra inicial usando o modelo Word2Vec."""
        if not self.word2vec_modelo:
            raise ValueError("O modelo Word2Vec ainda não foi treinado.")
        
        palavra_atual = palavra_inicial
        texto_gerado = [palavra_atual]
        
        for _ in range(comprimento - 1):
            similares = self.word2vec_modelo.wv.most_similar(palavra_atual, topn=50)  # Aumenta o número de similares
            palavras_similares = [p for p, _ in similares]
            prob_similares = [prob for _, prob in similares]
            
            if not prob_similares:
                break
            
            # Amostragem ponderada com base na similaridade
            palavra_atual = np.random.choice(palavras_similares, p=prob_similares/np.sum(prob_similares))
            texto_gerado.append(palavra_atual)
        
        texto_gerado_corrigido = self.corrigir_ortografia(texto_gerado)
        return ' '.join(texto_gerado_corrigido)
    
    def corrigir_ortografia(self, palavras):
        """Corrige a ortografia das palavras usando o corretor ortográfico."""
        palavras_corrigidas = []
        for palavra in palavras:
            if palavra in self.spell_checker:
                palavras_corrigidas.append(palavra)
            else:
                sugestoes = self.spell_checker.candidates(palavra)
                if sugestoes:
                    # Usa o método `correction` para obter a melhor sugestão
                    palavra_corrigida = self.spell_checker.correction(palavra)
                    palavras_corrigidas.append(palavra_corrigida)
                else:
                    palavras_corrigidas.append(palavra)
        return palavras_corrigidas

    def contar_parametros(self):
        """Conta o número total de parâmetros do modelo Word2Vec."""
        if self.word2vec_modelo:
            num_palavras = len(self.word2vec_modelo.wv)
            tamanho_vetor = self.word2vec_modelo.vector_size
            total_parametros = num_palavras * tamanho_vetor
            return total_parametros
        else:
            raise ValueError("O modelo Word2Vec ainda não foi treinado.")
```

### 4. Exemplo de Uso

Abaixo está um exemplo de como usar o modelo para ajustar TF-IDF, treinar Word2Vec e gerar texto.

```python
if __name__ == "__main__":
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

### 5. Explicação dos Métodos

- **`preprocessar_texto`**: Transforma o texto em minúsculas e o tokeniza.
- **`ajustar_tfidf`**: Ajusta o modelo TF-IDF aos textos fornecidos.
- **`transformar_tfidf`**: Transforma os textos em uma matriz TF-IDF.
- **`treinar_word2vec`**: Treina um modelo Word2Vec com os textos fornecidos.
- **`salvar_modelo_word2vec`**: Salva o modelo Word2Vec em um arquivo.
- **`carregar_modelo_word2vec`**: Carrega um modelo Word2Vec a partir de um arquivo.
- **`obter_vetor_da_palavra`**: Obtém o vetor de uma palavra específica do modelo Word2Vec.
- **`gerar_texto`**: Gera um texto a partir de uma palavra inicial usando o modelo Word2Vec.
- **`corrigir_ortografia`**: Corrige a ortografia das palavras geradas.
- **`contar_parametros`**: Conta o número total de parâmetros do modelo Word2Vec.

### 6. Conclusão

Você agora tem um modelo de embedding de texto que pode processar e gerar texto usando TF-IDF e Word2Vec, com correção ortográfica e contagem de parâmetros. Experimente com diferentes dados e parâmetros para ver como o modelo se comporta!
