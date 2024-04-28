import re

import pandas as pd
import numpy as np

import gensim
import spacy

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD


# Variables

new_features_temps : list[str] = ['ed ', 'were', 'been', 'will', 'have', 'has', 'we']
new_features_punctuation : list[str] = ['(', ';', '=']
new_features_specifics : list[str] = ['result', 'objective', 'http', ' NCT', '.gov', ' vs']

new_features : list[str] = new_features_temps + new_features_punctuation + new_features_specifics


# -------------------------------
# ----- Functions to add features
# -------------------------------

def get_sentence_vector(w2v : gensim.models.keyedvectors.KeyedVectors, sentences : np.ndarray) -> np.ndarray :
    '''
    This function is used to get the mean vector of the sentences
    
    Parameters
    ----------
    w2v : gensim.models.keyedvectors.KeyedVectors
        The Word2Vec model to use to get the mean vector of the sentences
    sentences : np.ndarray
        The sentences to get the mean vector
        
    Returns
    -------
    np.ndarray
        The mean vector of the sentences
    '''
    return np.array([w2v.get_mean_vector(sentence.split(' ')) for sentence in sentences])

# deprecated
def get_sentence_vector_with_lemma(w2v : gensim.models.keyedvectors.KeyedVectors, sentence : str) -> np.ndarray :
    '''
    (deprecated) 
    This function is used to get the mean vector of the sentences with the lemma
    
    Parameters
    ----------
    w2v : gensim.models.keyedvectors.KeyedVectors
        The Word2Vec model to use to get the mean vector of the sentences   
    sentence : str
        The sentence to get the mean vector
    
    Returns
    -------
    np.ndarray
        The mean vector of the sentences with the lemma
    '''
    
    nlp = spacy.load("en_core_sci_sm")

    l = [token.lemma_ for token in nlp(sentence) if token.ent_type_ != ""] # if is_entity 
    if l != []:
        return w2v.get_mean_vector(l)
    else :
        return np.zeros((300,))

def word_is_in_sentence_feature_count(sentences : np.ndarray, word : str) -> np.ndarray :
    '''
    This function is used to count the number of times a word is in a sentence
    
    Parameters
    ----------
    sentences : np.ndarray
        The dataframe to count the number of times a word is in a sentence
    word : str
        The word to count
        
    Returns
    -------
    np.ndarray
        The number of times the word is in a sentence
    '''
    return np.array([sentence.count(word) for sentence in sentences])

def proportion_numbers_in_sentence(sentences : np.ndarray) -> np.ndarray :
    '''
    This function is used to calculate the proportion of numbers in an array of sentences
    
    Parameters
    ----------
    sentences : np.ndarray
        The array to calculate the proportion of numbers
        
    Returns
    -------
    np.ndarray
        The proportion of numbers in each sentence
    '''
    return np.array([len(re.findall(r'\d+', sentence)) / len(sentence.split(' ')) for sentence in sentences])

def count_words_in_sentence(sentences : np.ndarray) -> np.ndarray :
    '''
    This function is used to count the number of words in an array of sentences
    
    Parameters
    ----------
    sentences : np.ndarray
        The array to count the number of words
        
    Returns
    -------
    np.ndarray
        The number of words in each sentence
    '''
    return np.array([len(sentence.split(' ')) for sentence in sentences])

def concat_2_np(array1 : np.ndarray, array2 : np.ndarray) -> np.ndarray :
    '''
    This function is used to concatenate 2 numpy arrays
    
    Parameters
    ----------
    array1 : np.ndarray
        The first array to concatenate
    array2 : np.ndarray
        The second array to concatenate
        
    Returns
    -------
    np.ndarray
        The concatenated array
    '''
    return np.concatenate((array1, array2[:, np.newaxis]), axis=1)


def get_final_data(df_train : pd.DataFrame, df_test : pd.DataFrame, w2v : gensim.models.word2vec.Word2Vec, to_do_pca : bool = True, additionnal_features = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] :
    '''
    This function is used to get the final data with Word2Vec model to use for the model (with or without PCA and with or without additionnal features) (X_train, y_train, X_test, y_test)
    
    Parameters
    ----------
    df_train : pd.DataFrame
        The train dataframe
    df_test : pd.DataFrame
        The test dataframe
    w2v : gensim.models.word2vec.Word2Vec
        The Word2Vec model to use to get the mean vector of the sentences
    to_do_pca : bool
        If we want to do PCA (10 components)
    additionnal_features : bool
        If we want to add additionnal features
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The final data to use for the model (X_train, y_train, X_test, y_test)
    '''
    
    if additionnal_features:
        for feature in new_features:
            df_train[feature] = word_is_in_sentence_feature_count(np.array(df_train['sentence']), feature)
            df_test[feature] = word_is_in_sentence_feature_count(np.array(df_test['sentence']), feature)

        df_train['nbr'] = proportion_numbers_in_sentence(np.array(df_train['sentence']))
        df_test['nbr'] = proportion_numbers_in_sentence(np.array(df_test['sentence']))

        df_train['words'] = count_words_in_sentence(np.array(df_train['sentence']))
        df_test['words'] = count_words_in_sentence(np.array(df_test['sentence']))

    X_train = get_sentence_vector(w2v.wv, np.array(df_train['sentence']))
    y_train = np.array(df_train['label'])

    X_test = get_sentence_vector(w2v.wv, np.array(df_test['sentence']))
    y_test = np.array(df_test['label'])

    # PCA if needed
    if to_do_pca:
        pca = PCA(n_components=10)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # add additionnal features
    if additionnal_features:
        for feature in new_features:
            X_train = concat_2_np(X_train, np.array(df_train[feature]))
            X_test = concat_2_np(X_test, np.array(df_test[feature]))

        X_train = concat_2_np(X_train, np.array(df_train['nbr']))
        X_test = concat_2_np(X_test, np.array(df_test['nbr']))

        X_train = concat_2_np(X_train, np.array(df_train['words']))
        X_test = concat_2_np(X_test, np.array(df_test['words']))
    
    return X_train, y_train, X_test, y_test


def get_final_data_for_cbow_sklearn(df_train : pd.DataFrame, df_test : pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] :
    '''
    This function is used to get the final data to use for the model (with or without PCA and with or without additionnal features) (X_train, y_train, X_test, y_test)
    
    Parameters
    ----------
    df_train : pd.DataFrame
        The train dataframe
    df_test : pd.DataFrame
        The test dataframe
   
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The final data to use for the model (X_train, y_train, X_test, y_test)
    '''
    
    for feature in new_features:
        df_train[feature] = word_is_in_sentence_feature_count(np.array(df_train['sentence']), feature)
        df_test[feature] = word_is_in_sentence_feature_count(np.array(df_test['sentence']), feature)

    df_train['nbr'] = proportion_numbers_in_sentence(np.array(df_train['sentence']))
    df_test['nbr'] = proportion_numbers_in_sentence(np.array(df_test['sentence']))

    df_train['words'] = count_words_in_sentence(np.array(df_train['sentence']))
    df_test['words'] = count_words_in_sentence(np.array(df_test['sentence']))

    vectorizer = CountVectorizer()

    X_train = vectorizer.fit_transform(df_train['sentence'])
    y_train = np.array(df_train['label'])

    X_test = vectorizer.transform(df_test['sentence'])
    y_test = np.array(df_test['label'])

    svd = TruncatedSVD(n_components=30)
    X_train = svd.fit_transform(X_train)
    X_test = svd.transform(X_test)


    for feature in new_features:
        X_train = concat_2_np(X_train, np.array(df_train[feature]))
        X_test = concat_2_np(X_test, np.array(df_test[feature]))

    X_train = concat_2_np(X_train, np.array(df_train['nbr']))
    X_test = concat_2_np(X_test, np.array(df_test['nbr']))

    X_train = concat_2_np(X_train, np.array(df_train['words']))
    X_test = concat_2_np(X_test, np.array(df_test['words']))

    return X_train, y_train, X_test, y_test # type: ignore



# -------------------------------
# ----- Functions to analyze if new features are useful
# -------------------------------

# its 3 functions just calculate the proportions by label

def words_number(df : pd.DataFrame) -> list[int] :
    '''
    This function is used to calculate the number of words in a sentence group by label
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to calculate the number of words
        
    Returns
    -------
    list[int]
        The number of words in a sentence group by label
    '''
    
    l : list[int] = []
    for i in range (5):
        sum : int = 0
        count : int = df[df['label'] == i]['sentence'].shape[0]
        
        for sentence in df[df['label'] == i]['sentence']:
            sum += len(sentence.split(' '))

        l.append(sum // count)
    return l

def word_is_in_sentence(df : pd.DataFrame, word : str) -> list[float] :
    '''
    This function is used to calculate the proportion of times a word is in a sentence group by label
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to calculate the proportion of times a word is in a sentence
    word : str
        The word to calculate the proportion
    
    Returns
    -------
    list[float]
        The proportion of times a word is in a sentence group by label
    '''
    
    l : list[float] = []
    for i in range (5):
        sum : int = 0
        count : int = df[df['label'] == i]['sentence'].shape[0]
        
        for sentence in df[df['label'] == i]['sentence']:
            if word in sentence:
                sum += 1

        l.append(sum / count)
    return l

def get_proportion_tag(df : pd.DataFrame, tag : str) -> list[float] :
    '''
    This function is used to calculate the proportion of a tag in a sentence group by label
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to calculate the proportion of a tag
    tag : str
        The tag to calculate the proportion
        
    Returns
    -------
    list[float]
        The proportion of a tag in a sentence group by label
    '''
    
    l : list[float] = []
    for i in range (5):
        sum : float = 0
        count : int = df[df['label'] == i]['sentence'].shape[0]
        
        for sentence in df[df['label'] == i]['sentence']:

            words = word_tokenize(sentence)
            tags = pos_tag(words)

            nouns = [word for word, pos in tags if pos.startswith(tag)]
            
            sum += len(nouns) / len(tags)
            
        l.append(sum / count)

    return l


# its 2 functions add the new feature directly to the dataframe
# deprecated because theses features are not useful

def add_num_doc(df : pd.DataFrame) -> pd.DataFrame :
    '''
    This function is used to add the number of the document in the dataframe
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to add the number of the document
        
    Returns
    -------
    pd.DataFrame
        The dataframe with the number of the document
    '''
    num_doc_array = np.zeros(df.shape[0], dtype=int)

    last_index : int = 0
    num_doc : int = 1

    # Boucle sur les index des lignes NaN (chaque lignes NaN correspond à un nouveau document)
    for index in df[df[df.columns[1]].isna()].index:
        num_doc_array[last_index : index] = num_doc

        last_index = index
        num_doc += 1

    # pour le dernier doc
    num_doc_array[last_index : df.shape[0]] = num_doc

    df['num_doc'] = num_doc_array

    return df

def add_before_after(df : pd.DataFrame) -> pd.DataFrame :
    '''
    This function is used to add the label before and after the current label in the dataframe
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to add the label before and after
        
    Returns
    -------
    pd.DataFrame
        The dataframe with the label before and after
    '''

    before_array = np.full(df.shape[0], np.nan, dtype=float)
    after_array = np.full(df.shape[0], np.nan, dtype=float)

    current_num_doc : int

    for index in range (1, df.shape[0] - 1):
        current_num_doc = df['num_doc'][index]
        
        if df['num_doc'][index - 1] == current_num_doc:
            before_array[index] = df['label'][index - 1]
            
        if df['num_doc'][index + 1] == current_num_doc:
            after_array[index] = df['label'][index + 1]

    # pour le premier et dernier sample
    after_array[0] = df['label'][1]
    before_array[df.shape[0] - 1] = df['label'][df.shape[0] - 2]

    df['label_before'] = before_array
    df['label_after'] = after_array

    return df

