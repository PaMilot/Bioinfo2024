import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import gensim
from gensim.models import Word2Vec

def label_encoding(label : np.ndarray) -> np.ndarray :
    '''
    This function is used to encode the labels of the dataframe (ex : 'OBJECTIVE' -> 3)
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to encode
        
    Returns
    -------
    np.ndarray
        The encoded labels
    '''

    label_encoder = LabelEncoder()

    return np.array(label_encoder.fit_transform(label))

def create_df_clean(URL : str) -> pd.DataFrame :
    '''
    This function is used to create a clean dataframe from a URL (worked with train and test data)
    
    Parameters
    ----------
    URL : str
        The URL of the dataframe to clean
    
    Returns
    -------
    pd.DataFrame
        The clean dataframe    
    '''
    
    df = pd.read_csv(URL, sep= '\t')

    # reset because labels considered as index
    df.reset_index(inplace=True)
    
    # remove rows with NaN values (ex : ###24293578  |    NaN)
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)

    # rename columns
    df = df.rename(columns = {df.columns[1]:'sentence', 'index' : 'label'})

    # label encoding
    df['label'] = label_encoding(np.array(df['label']))

    return df

def create_word_embeddings(sentences, vector_size : int = 200, min_count : int = 2, sg : int = 0, epochs = 10, window = 3) -> gensim.models.word2vec.Word2Vec :
    '''
    This function is used to create word embeddings from an array of sentences
    
    Parameters
    ----------
    sentences :
        The sentences to create word embeddings
    vector_size : int
        The size of the vector
    min_count : int
        The minimum count of a word to be considered
    sg : int
        The training algorithm : 0 for CBOW, 1 for skip-gram
    epochs : int
        The number of epochs to train the model
    window : int
        The size of the window to consider for the context
    
    Returns
    -------
    gensim.models.word2vec.Word2Vec
        The word embeddings model
    '''
    return Word2Vec(sentences, vector_size=vector_size, min_count=min_count, sg=sg, epochs=epochs, workers=4, window=window)