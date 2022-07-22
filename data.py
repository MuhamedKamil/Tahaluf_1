from bs4 import BeautifulSoup
from utilities import get_info_for_each_paragraph


class Data:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None

    def load_data(self):
        with open(self.data_path) as f:
            self.data = f.read()

    def get_paragraphs_tags(self):
        soup = BeautifulSoup(self.data, 'html.parser')
        ParagraphsWords = []
        WordsTags = []
        Paragraphs = []

        for text in soup.find_all('text'):
            for paragraphs in text.find_all('p'):
                for paragraph in str(paragraphs).split('<p>'):
                    if paragraph != '':
                        CleanedParagraph, ParagraphWords, Tags = get_info_for_each_paragraph(paragraph)
                        Paragraphs.append(CleanedParagraph)
                        ParagraphsWords.append(ParagraphWords)
                        WordsTags.append(Tags)

        return Paragraphs, ParagraphsWords, WordsTags
