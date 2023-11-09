import gensim
from gensim.models import KeyedVectors

class WordEmbeddings:
    def __init__(self, binary_file, limit=1000000):
        self.wv = KeyedVectors.load_word2vec_format(binary_file, binary=True, limit=limit)
        self.wv.vectors = self.wv.vectors.copy()

    def save_as_csv(self, output_file):
        self.wv.save_word2vec_format(output_file)

    def get_word_embedding(self, word):
        if word in self.wv:
            return self.wv[word]
        else:
            return None
