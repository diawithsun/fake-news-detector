from app.pipeline.processors.preprocessing import preprocess
from app.pipeline.engine.vectorization import Tf_IDF_vectorization
from app.pipeline.engine.pca import PCA_transform
from app.pipeline.engine.run_model import model_run

def inference(title, text, model_type = 'logistic'):
    title_text = preprocess(title, text)
    tfidf_vectorized = Tf_IDF_vectorization(title_text)
    pca = PCA_transform(tfidf_vectorized)
    prediction= model_run(model_type, pca)
    prediction = 'Fake' if prediction == 0 else 'True'
    return prediction

