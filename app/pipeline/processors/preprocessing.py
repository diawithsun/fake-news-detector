from app.pipeline.utils.preprocessing_utils import remove_nonalpha, lemmatizer_and_stop_word_removal

def preprocess(title, text):

    title_text = title + ' ' + text
    title_text = title_text.lower()
    title_text = remove_nonalpha(title_text)
    title_text = lemmatizer_and_stop_word_removal(title_text)
    return title_text
