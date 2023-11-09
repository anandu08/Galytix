import logging
from project.embeddings import WordEmbeddings
from project.data_processing import PhraseProcessor
from project.similarity import SimilarityCalculator


logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    logging.info("Starting the script...")

    word_embeddings_file = 'data/GoogleNews-vectors-negative300.bin'
    phrases_file = 'data/phrases.csv'

    logging.info("Loading word embeddings...")
    word_embeddings = WordEmbeddings(word_embeddings_file)

    logging.info("Processing phrases...")
    phrase_processor = PhraseProcessor(phrases_file,word_embeddings)

    logging.info("Setting up similarity calculator...")
    similarity_calculator = SimilarityCalculator(word_embeddings, phrase_processor)

    user_input = input("Your Input: ")
    logging.info(f"Received user input: {user_input}")

    closest_match, similarity = similarity_calculator.find_closest_match(user_input)
    logging.info(f"Closest match: {closest_match}, Similarity: {similarity}")

    print(f"Closest Match: {closest_match}, Similarity: {similarity}")
