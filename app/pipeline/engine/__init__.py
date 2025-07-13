import joblib
import os

with open(r'C:\Users\Diyasha Chakrabarti\Documents\python projects\Fake_News_Detector\app\models\tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = joblib.load(f)

with open(r'C:\Users\Diyasha Chakrabarti\Documents\python projects\Fake_News_Detector\app\models\ipca.pkl', 'rb') as f:
    ipca = joblib.load(f)

with open(r'C:\Users\Diyasha Chakrabarti\Documents\python projects\Fake_News_Detector\app\models\rf_model.pkl', 'rb') as f:
    rf = joblib.load(f)

with open(r'C:\Users\Diyasha Chakrabarti\Documents\python projects\Fake_News_Detector\app\models\logistic_model.pkl', 'rb') as f:
    model_log = joblib.load(f)  

with open(r'C:\Users\Diyasha Chakrabarti\Documents\python projects\Fake_News_Detector\app\models\svm_model.pkl', 'rb') as f:
    svm_model = joblib.load(f)
    