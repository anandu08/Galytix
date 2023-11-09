from project.embeddings import WordEmbeddings
from project.data_processing import PhraseProcessor
from project.similarity import SimilarityCalculator

if __name__ == "__main__":
    word_embeddings_file = 'data/GoogleNews-vectors-negative300.bin'
    phrases_file = 'data/phrases.csv'

    word_embeddings = WordEmbeddings(word_embeddings_file)
    phrase_processor = PhraseProcessor(phrases_file,word_embeddings)
    similarity_calculator = SimilarityCalculator(word_embeddings, phrase_processor)


    user_input = input("Your Input: ")
    closest_match, similarity = similarity_calculator.find_closest_match(user_input)
    print(f"Closest Match: {closest_match}, Similarity: {similarity}")
