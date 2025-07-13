from app.pipeline.engine import ipca

def PCA_transform(tfidf_vectorized):
    # Transform using PCA
    tfidf_pca = ipca.transform(tfidf_vectorized)
    return tfidf_pca