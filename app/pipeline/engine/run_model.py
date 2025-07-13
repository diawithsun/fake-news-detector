from app.pipeline.engine import rf, model_log, svm_model
def model_run(model_type, tfidf_pca):
    print(f"Running model: {model_type}")
    if model_type == 'logistic':
        model = model_log
    elif model_type == 'rf':
        model = rf
    elif model_type == 'svm':
        model = svm_model
    else:
        raise ValueError("Invalid model type. Choose from 'logistic', 'rf', or 'svm'.")

    # Make prediction
    prediction = model.predict(tfidf_pca)
    return prediction[0]
    

    