from flask import Flask, request, render_template
import pandas as pd
import os
from src.utils import load_object
from src.exception import CustomException
from datetime import datetime
import sys
import spacy

from src.Pipeline.predict_pipeline import CustomData , BertFakeNews 
app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

model_path = os.path.join("artifact", "model.pkl")
preprocessor_path = os.path.join("artifact", "preprocessor.pkl")

model = load_object(model_path)
preprocessor = load_object(preprocessor_path)
bert_pipe = BertFakeNews()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            
            title = request.form["title"]
            text = request.form["text"]
            date = request.form["date"]

            # TFIDF + LR
            input_data = CustomData(title, text, date)
            df = input_data.get_data_as_data_frame()

            transformed = preprocessor.transform(df)
            prediction_lr = model.predict_proba(transformed)[0][1]  

            percent_lr = round(prediction_lr * 100, 2)

            #Fine Tuned BERT
            percent_BERT = bert_pipe.predict_fake_prob(title,text)
            return render_template("index.html",result = {
                                                "LogReg-TFIDF": f"{percent_lr:.2f} %",
                                                "Fine-tuned BERT": f"{percent_BERT*100:.2f} %"})

        except Exception as e:
            raise CustomException(e, sys)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
