import os
from data import Data
from CRF import StanfordCRF
from Transformer import SpacyTransformer

# *************************************** Paths *****************************************************
os.environ['JAVA_HOME'] = "C:/Program Files/Java/jdk-18.0.1.1/bin/java.exe"
data_path = "./news_sample_ner.txt"
CRF_model_path = './StanfordNERModel/english.all.3class.distsim.crf.ser.gz'
jar_file_path = './StanfordNERModel/stanford-ner-4.2.0.jar'
Transformer_model_path = 'en_core_web_trf'
# ************************************* Loading Data *************************************************
DataPreprocessor = Data(data_path)
DataPreprocessor.load_data()
ListOfParagraphs, ListOfParagraphsWords, ListOfTags = DataPreprocessor.get_paragraphs_tags()
# ************************************** Statical Model ***********************************************
CRF_model = StanfordCRF(CRF_model_path, jar_file_path)
CRF_predictions = CRF_model.predict_all(ListOfParagraphsWords)
print("CRF Evaluation")
print("---------------------------------------------------------")
CRF_model.evaluate(CRF_predictions, ListOfTags, ListOfParagraphs)
# ************************************** Transformer Model *********************************************
Transformer_model = SpacyTransformer(Transformer_model_path)
Transformer_predictions = Transformer_model.predict_all(ListOfParagraphs)
print("Transformer Evaluation")
print("---------------------------------------------------------")
Transformer_model.evaluate(Transformer_predictions, ListOfTags, ListOfParagraphs)
