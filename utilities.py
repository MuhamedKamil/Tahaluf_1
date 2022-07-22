import html2text
from bs4 import BeautifulSoup
from nltk import word_tokenize

h = html2text.HTML2Text()


def get_info_for_each_paragraph(paragraph):
    parser = BeautifulSoup(paragraph, 'html.parser')
    Entities = parser.find_all('enamex')
    Tags = []
    CleanedParagraph = h.handle(paragraph).strip().replace('\n', ' ').replace('``', '').replace("''", '')
    for entity in Entities:
        Tags.append((entity['type'], entity.text))
    return CleanedParagraph, sentence_to_words(CleanedParagraph), Tags


def get_difference_between_two_lists(list1, list2):
    return list(set(list1) - set(list2))


def merge_two_lists(list1, list2):
    return list1 + list2


def remove_punctuation(sentence):
    punctuations = """!()-[]{};:'"\,<>./?@#$%^&*_~ """
    updated_sentence = ""
    for char in sentence:
        if char not in punctuations:
            updated_sentence = updated_sentence + char
    return updated_sentence


def sentence_to_words(sentence):
    return word_tokenize(sentence)


def reversePrediction(predictions):
    return [[tup[::-1] for tup in x] for x in predictions]
