from eval4ner import muc
from nltk import StanfordNERTagger
from utilities import sentence_to_words, reversePrediction


class StanfordCRF:
    def __init__(self, model_path, jar_file_path):
        self.model_path = model_path
        self.jar_file_path = jar_file_path
        self.model = StanfordNERTagger(self.model_path,
                                       self.jar_file_path,
                                       encoding='utf-8')

    def predict(self, sentence):
        tokens = sentence_to_words(sentence)
        return self.model.tag(tokens)

    def predict_all(self, sentences):
        filters = ['PERSON', 'ORGANIZATION', 'LOCATION']
        predictions = reversePrediction(self.model.tag_sents(sentences))
        return [[tup for tup in x if tup[0] in filters] for x in predictions]

    def evaluate(self, predictions, truth, sentences, verbose=False):
        muc.evaluate_all(predictions, truth, sentences, verbose=verbose)
