# 📰 Fake-News Detector

A lightweight Flask app that flags misleading articles.  
Two models run side-by-side: a **TF-IDF+LR** and a **fine-tuned BERT** approach.

---

## Project Overview

| Approach | Snapshot |
|----------|----------|
| **TF-IDF → Logistic Regression** | Fast baseline, trained on a 2017 open corpus; instant prediction. |
| **Fine-Tuned BERT** | Based on `bert-base-uncased`, fine-tuned  on the 2023 *Truth Seeker* dataset - https://www.unb.ca/cic/datasets/truthseeker-2023|
---

![lr+bert](https://github.com/user-attachments/assets/a0496079-b1da-45a2-98f9-0689806a242a)

## Future Steps 🚧

- Retrain the TF-IDF pipeline on the 2023 corpus with full CV + grid-search (LR, SVM, XGBoost).  
- Explore ensemble voting / stacking of TF-IDF and BERT.  
