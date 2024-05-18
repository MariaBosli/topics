from summarizer import Summarizer
from nltk.util import ngrams
from collections import Counter
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
import spacy
from keybert import KeyBERT

def spacy_tokenizer(sentence):
    parser = spacy.load("en_core_sci_sm")
    parser.max_length = 7000000
    mytokens = parser(sentence)
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
    mytokens = [word for word in mytokens if word not in STOP_WORDS and word not in punctuation]
    mytokens = " ".join([i for i in mytokens])
    return mytokens

def extract_bigrams(text):
    tokens = spacy_tokenizer(text)
    bigrams = list(ngrams(tokens.split(), 2))
    freq_dist = Counter(bigrams)
    most_common_bigrams = [bigram for bigram, _ in freq_dist.most_common(80)]
    return most_common_bigrams

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text

def summarize_text(file_path):
    text = read_text_file(file_path)
    model = Summarizer()
    summary = model(text)
    return summary

def extract_keywords_using_KeyBert(file_path, top_n=80):
    # Lire le contenu du fichier
    with open(file_path, 'r') as file:
        doc = file.read()

    # Initialiser le modèle KeyBERT
    kw_model = KeyBERT()

    # Extraire les mots-clés
    keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), top_n=top_n)

    return keywords