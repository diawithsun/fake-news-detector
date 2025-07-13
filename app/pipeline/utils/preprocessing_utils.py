import re
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
import nltk

def remove_nonalpha(text):
    # Remove URLs (http, https, www)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove special characters (keeping alphanumerics and spaces)
    text = re.sub(r'[^A-Za-z\s]', '', text)

    # Optional: remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load these once (outside the function)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Helper to convert POS tag
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default

# Final function
def lemmatizer_and_stop_word_removal(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)

    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tags
        if word.lower() not in stop_words
    ]

    return " ".join(lemmatized)