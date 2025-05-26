import sys
import pandas as pd
import spacy
from src.exception import CustomException
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib

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
class BertFakeNews:
    def __init__(
        self,
        model_repo: str = "Igoras6534/fine_tuned_BERT_fakenews_8000",
        tokenizer_repo: str = "bert-base-uncased",  
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_repo)
        self.model.eval()

    def predict_fake_prob(self, title: str, text: str) -> float:
        joined = title + " " + text
        inputs   = self.tokenizer(joined,
                                truncation=True,
                                return_tensors="pt",
                                padding = True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)        
        return probs.squeeze().tolist()[0]

class New_LR_Predictor:
    def __init__(self, model_path='artifact/new_lr_model.pkl'):
        self.pipeline = joblib.load(model_path)
        
    def extract_features(self, statement, tweet):
        features = {}
        
        combined_text = statement + " " + tweet
        combined_doc = nlp(combined_text)
        
        entity_counts = {}
        total_tokens = len([token for token in combined_doc if not token.is_punct])
        
        if total_tokens > 0:
            for ent in combined_doc.ents:
                if ent.label_ not in entity_counts:
                    entity_counts[ent.label_] = 0
                entity_counts[ent.label_] += 1
            
            for entity_type in ['ORG', 'NORP', 'GPE', 'PERSON', 'MONEY', 'DATE', 'CARDINAL', 
                               'PERCENT', 'ORDINAL', 'FAC', 'LAW', 'PRODUCT', 'EVENT', 
                               'TIME', 'LOC', 'WORK_OF_ART', 'QUANTITY', 'LANGUAGE']:
                features[f'{entity_type}_percentage'] = entity_counts.get(entity_type, 0) / total_tokens * 100
        else:
            for entity_type in ['ORG', 'NORP', 'GPE', 'PERSON', 'MONEY', 'DATE', 'CARDINAL', 
                               'PERCENT', 'ORDINAL', 'FAC', 'LAW', 'PRODUCT', 'EVENT', 
                               'TIME', 'LOC', 'WORK_OF_ART', 'QUANTITY', 'LANGUAGE']:
                features[f'{entity_type}_percentage'] = 0
        
        words = [token.text.lower() for token in combined_doc if not token.is_punct and not token.is_space]
        unique_words = set(words)
        
        features['unique_count'] = len(unique_words)
        features['total_count'] = len(words)
        
        if words:
            features['Max word length'] = max(len(word) for word in words)
            features['Min word length'] = min(len(word) for word in words)
            features['Average word length'] = sum(len(word) for word in words) / len(words)
        else:
            features['Max word length'] = 0
            features['Min word length'] = 0 
            features['Average word length'] = 0
        
        features['present_verbs'] = len([token for token in combined_doc if token.pos_ == "VERB" and token.tag_ in ["VBP", "VBZ"]])
        features['past_verbs'] = len([token for token in combined_doc if token.pos_ == "VERB" and token.tag_ == "VBD"])
        features['adjectives'] = len([token for token in combined_doc if token.pos_ == "ADJ"])
        features['pronouns'] = len([token for token in combined_doc if token.pos_ == "PRON"])
        features['TOs'] = len([token for token in combined_doc if token.tag_ == "TO"])
        features['determiners'] = len([token for token in combined_doc if token.pos_ == "DET"])
        features['conjunctions'] = len([token for token in combined_doc if token.pos_ == "CCONJ" or token.pos_ == "SCONJ"])
        
        features['dots'] = combined_text.count('.')
        features['exclamation'] = combined_text.count('!')
        features['questions'] = combined_text.count('?')
        features['ampersand'] = combined_text.count('&')
        features['capitals'] = sum(1 for c in combined_text if c.isupper())
        features['digits'] = sum(1 for c in combined_text if c.isdigit())
        
        features['long_word_freq'] = len([word for word in words if len(word) > 6]) / len(words) if words else 0
        features['short_word_freq'] = len([word for word in words if len(word) < 4]) / len(words) if words else 0
        
        features['statement'] = statement
        features['tweet'] = tweet
        
        return features
    
    def predict(self, statement, tweet):
        try:
            features = self.extract_features(statement, tweet)
            
            features_df = pd.DataFrame([features])
            

            prediction = self.pipeline.predict(features_df)[0]
            prediction_proba = self.pipeline.predict_proba(features_df)[0][0]
            
            return {
                'prediction': 'Real' if prediction == 1 else 'Fake', 
                'probability': prediction_proba
            }
        except Exception as e:
            raise CustomException(e, sys)