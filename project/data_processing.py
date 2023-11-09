import csv
import numpy as np

class PhraseProcessor:
    def __init__(self, phrases_file, word_embeddings):
        self.phrases = self.load_phrases(phrases_file)
        self.word_embeddings = word_embeddings

    def load_phrases(self, file_path):
        with open(file_path, 'r',encoding='Windows-1252') as file:
            reader = csv.reader(file)
            return [row[0] for row in reader]

    def calculate_normalized_embedding(self, phrase):
        words = phrase.split()
        phrase_embedding = np.zeros(self.word_embeddings.wv.vector_size)
        for word in words:
            word_embedding = self.word_embeddings.get_word_embedding(word)
            if word_embedding is not None:
                phrase_embedding += word_embedding
        norm = np.linalg.norm(phrase_embedding)
        if norm > 0:
            phrase_embedding /= norm
        return phrase_embedding

    def find_closest_match(self, user_input):
        user_vector = self.word_embeddings.get_word_embedding(user_input)
        if user_vector is not None:
            user_vector /= np.linalg.norm(user_vector)
            similarities = [np.dot(user_vector, self.calculate_normalized_embedding(phrase)) for phrase in self.phrases]
            best_match_index = similarities.index(max(similarities))
            return self.phrases[best_match_index], similarities[best_match_index]
        else:
            return None, 0.0
