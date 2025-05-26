from flask import Flask, request, render_template
import pandas as pd
import os
from src.utils import load_object
from src.exception import CustomException
from datetime import datetime
import sys
import spacy

from src.Pipeline.predict_pipeline import BertFakeNews, New_LR_Predictor
app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

model_path = os.path.join("artifact", "model.pkl")
preprocessor_path = os.path.join("artifact", "preprocessor.pkl")

lr_predictor = New_LR_Predictor()
bert_pipe = BertFakeNews()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            
            title = request.form["title"]
            text = request.form["text"]
            date = request.form["date"]

            # TFIDF + Logistic Regression
            new_lr_result = lr_predictor.predict(title,text)
            percent_lr = round(new_lr_result['probability'] * 100,2)

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
