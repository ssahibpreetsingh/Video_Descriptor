import joblib
from keytotext import pipeline

try:
    text_model=joblib.load("keytotext.joblib")
except:
    text_model=pipeline("mrm8488/t5-base-finetuned-common_gen")
    joblib.dump(text_model,"text_model.pkl")
