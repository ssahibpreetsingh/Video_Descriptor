import joblib

try:
    text_model=joblib.load("keytotext.joblib")
except:
    from keytotext import pipeline
    text_model=pipeline("mrm8488/t5-base-finetuned-common_gen")