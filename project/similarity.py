from gensim import matutils
import numpy as np

class SimilarityCalculator:
    def __init__(self, word_embeddings, phrase_processor):
        self.word_embeddings = word_embeddings
        self.phrase_processor = phrase_processor

    def calculate_phrase_similarity(self, phrase1, phrase2, method='cosine'):
        vector1 = self.phrase_processor.calculate_normalized_embedding(phrase1)
        vector2 = self.phrase_processor.calculate_normalized_embedding(phrase2)
        
        if method == 'cosine':
            similarity = np.dot(vector1, vector2)
        elif method == 'euclidean':
            similarity = np.linalg.norm(vector1 - vector2)
        else:
            raise ValueError("Unsupported similarity method")
        
        return similarity

    def find_closest_match(self, user_input):
        user_vector = self.word_embeddings.get_word_embedding(user_input)
        if user_vector is not None:
            user_vector.setflags(write=True)
            user_vector /= np.linalg.norm(user_vector)
            similarities = [np.dot(user_vector, self.phrase_processor.calculate_normalized_embedding(phrase)) for phrase in self.phrase_processor.phrases]
            best_match_index = similarities.index(max(similarities))
            return self.phrase_processor.phrases[best_match_index], similarities[best_match_index]
        else:
            return None, 0.0
        


