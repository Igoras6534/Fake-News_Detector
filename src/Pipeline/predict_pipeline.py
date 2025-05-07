import sys
import pandas as pd
import spacy
from src.exception import CustomException

nlp = spacy.load("en_core_web_sm")


def avg_sentence_length(text):
    doc = nlp(text)
    lengths = [len([token for token in sent if not token.is_punct]) for sent in doc.sents]
    return sum(lengths) / len(lengths) if lengths else 0


class CustomData:
    def __init__(self, title: str, text: str, date: str):
        self.title = title
        self.text = text
        self.date = date  

    def get_data_as_data_frame(self):
        try:
            title_length = len(self.title)
            avg_title_sent_length = avg_sentence_length(self.title)
            avg_text_sent_length = avg_sentence_length(self.text)
            exclamation_count = self.text.count("!")
            weekday = pd.to_datetime(self.date).weekday()

            custom_data_input_dict = {
                "title": [self.title],
                "text": [self.text],
                "title_length": [title_length],
                "weekday": [weekday],
                "exclamation_count": [exclamation_count],
                "avg_text_sent_length": [avg_text_sent_length],
                "avg_title_sent_length": [avg_title_sent_length],
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
