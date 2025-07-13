from app.pipeline.engine import tfidf_vectorizer

def Tf_IDF_vectorization(title_text):
    # Transform the title_text
    tfidf_vectorized = tfidf_vectorizer.transform([title_text])
    return tfidf_vectorized
