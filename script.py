import pandas as pd
from IPython.display import display

from tqdm import tqdm
tqdm.pandas()

import numpy as np

from spacy.lang.en import English
nlp = English()

from spacy.lang.en.stop_words import STOP_WORDS

import seaborn as sns
import matplotlib.pyplot as plt


def clean_text(document):
    """
    Text preprocessing with lemmatization and tokenization for forum posts. 
    
    Parameters: 
    document (str): forum post
    
    Returns: 
    list: list of tokenized lemmatized words
    """
    # create nlp object
    doc = nlp(document)
    # lemmatize each word
    lemmaed = [word.lemma_ for word in doc]
    # tokenize the lemmatized words
    tokens = [token.lower() for token in lemmaed]
    words = [token for token in tokens if token not in STOP_WORDS]
    return words