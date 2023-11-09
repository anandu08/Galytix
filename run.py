import logging
from project.embeddings import WordEmbeddings
from project.data_processing import PhraseProcessor
from project.similarity import SimilarityCalculator

class Pipeline:
    def __init__(self, config):
        self.config = config
        logging.basicConfig(level=logging.INFO)

    def load_embeddings(self):
        logging.info("Loading word embeddings...")
        return WordEmbeddings(self.config['word_embeddings_file'])

    def process_phrases(self, word_embeddings):
        logging.info("Processing phrases...")
        return PhraseProcessor(self.config['phrases_file'], word_embeddings)

    def setup_similarity_calculator(self, word_embeddings, phrase_processor):
        logging.info("Setting up similarity calculator...")
        return SimilarityCalculator(word_embeddings, phrase_processor)

    def run(self):
        logging.info("Starting the script...")
        word_embeddings = self.load_embeddings()
        phrase_processor = self.process_phrases(word_embeddings)
        similarity_calculator = self.setup_similarity_calculator(word_embeddings, phrase_processor)

        user_input = input("Your Input: ")
        logging.info(f"Received user input: {user_input}")

        closest_match, similarity = similarity_calculator.find_closest_match(user_input)
        logging.info(f"Closest match: {closest_match}, Similarity: {similarity}")

        print(f"Closest Match: {closest_match}, Similarity: {similarity}")

if __name__ == "__main__":
    config = {
        'word_embeddings_file': 'data/GoogleNews-vectors-negative300.bin',
        'phrases_file': 'data/phrases.csv'
    }
    pipeline = Pipeline(config)
    pipeline.run()
