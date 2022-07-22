import spacy
from eval4ner import muc
from tqdm import tqdm


class SpacyTransformer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = spacy.load(self.model_path)

    def predict(self, sentence):
        doc = self.model(sentence)
        entities = []
        for e in doc.ents:
            if e.label_ == 'ORG':
                entities.append(('ORGANIZATION', e.text))
            elif e.label_ == 'LOC' or e.label_ == 'GPE':
                entities.append(('LOCATION', e.text))
            elif e.label_ == 'PERSON':
                entities.append(('PERSON', e.text))

        return entities

    def predict_all(self, sentences):
        predictions = []
        for sentence in tqdm(sentences):
            prediction = self.predict(sentence)
            predictions.append(prediction)
        return predictions

    def evaluate(self, predictions, truth, sentences, verbose=False):
        muc.evaluate_all(predictions, truth, sentences, verbose=verbose)
