import os
# Configure settings for project
# Need to run this before calling models from application!


from sklearn.cluster import KMeans

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import gensim
from collections import defaultdict
from nltk.corpus import stopwords
import nltk
import string
import re
import unidecode

# Define some global table to clean the strings
stop_words = set(stopwords.words('english'))
remove_punctuation = dict((ord(char), None) for char in string.punctuation)
remove_numbers = dict((ord(char), None) for char in string.punctuation)



def get_df_nouns_only(entry):
    """
    1) Remove terms with digits as well as terms with latex tags.
    2) Retains only the nouns from the abstract using nltk.pos_tag.

    Input:
    -----
    entry: the abstract from an article in the DataFrame

    Output:
    ------
    sentence_nouns_only: a string with only the nouns of the abstract
    """

    tokens = nltk.word_tokenize( re.sub('\w*\d+\w*','',re.sub('\$.*?\$','',entry)))
    tagged = nltk.pos_tag(tokens)
    first_letter = np.array([x[0] for x in np.array(tagged)[:,1]])

    sentence_nouns_only = ' '.join(np.array(tagged)[first_letter=='N'][:,0])

    return sentence_nouns_only


def dummy_fun(doc):

    """
    A dummy function for the vectorizer
    """
    return doc

def compute_cosine_sim(vec1,vec2):

    """
    Compute the cosine between two vectors
    """

    norm1 = np.dot(vec1,vec1)**0.5
    norm2 = np.dot(vec2,vec2)**0.5

    return np.dot(vec1,vec2)/(norm1*norm2)

def get_mean_vectors_idf(X,word2vec,tfidf):

    """
    Given a word2vec embedding and a TF-IDF vectorizer, computed the vector associated to a sentence by taking the
    average of the vectors for the words of the sentence, weigthed by the tfidf.

    Inputs:
    ------
    X: a iterable of sentences
    word2vec: a mapping between words and vector (using gensim)
    tfidf: a trained TF-IDF vectorizer

    Output:
    -----
    The array of averaged vectors for each sentence in the iterable X
    """

    max_idf = max(tfidf.idf_)
    word2weight = defaultdict(lambda: max_idf, [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

    return np.array([
                np.mean([word2vec[w] * word2weight[w]
                         for w in words if w in word2vec], axis=0)
                for words in X
            ])


def prepare_series_text_split(series):

    """
    Applies various cleaning method to the abstracts from a Series dataframe.

    Input:
    -----
    The original series

    Output:

    The cleaned series
    """

    series_text = series.str.replace('\Wm\w\W','')\
                                .str.lower()\
                                .str.replace('mml','')\
                                .str.replace('math display','')\
                                .str.replace('math formula','')\
                                .str.replace('inline','')\
                                .map(get_df_nouns_only)\
                                .str.translate(remove_punctuation)\
                                .str.translate(remove_numbers)\
                                .map(lambda x:unidecode.unidecode(x))\
                                .str.split()\
                                .map(lambda entry: [w for w in entry if not w in stop_words])

    return series_text


def get_highest_idf_words_title(df,ntopics=5):

    """
    Given a dataframe with the cluster labels, looks at the titles in each label and select the ntopics words with
    the highest idf in the cluster.

    Inputs:
    df: the dataframe of articles with their cluster label
    ntopics: the number of terms with highest idf to keep

    Output:
    ------
    most_frequentwords_dict: a dict relating each labels to its highest idf words

    """


    df_grouped = df.groupby(['Label']).agg({'title':' '.join})
    df_joined = prepare_series_text_split(df_grouped['title']).map(' '.join)

    tfidf = TfidfVectorizer(min_df=.0025, max_df=0.4, stop_words='english', ngram_range=(2,2))
    transformed_corpus = tfidf.fit_transform(df_joined)
    tvocab = np.array(tfidf.get_feature_names())
    df_transformed = pd.DataFrame(transformed_corpus.todense(), columns=tvocab,index=df_joined.index)
    most_frequentwords_dict = {}
    for i in df_transformed.index:

        most_frequentwords_dict[i] = df_transformed.loc[i].sort_values(ascending=False)[:ntopics].index

    return most_frequentwords_dict


def prepare_for_kmeans(df):

    """
    Applies all the method above to clean the data, computes the word2vec mapping (including bigrams) as well as the
    TF-IDF vectorizer on the same tokens as word2vec.
    Also applies a StandardScaler to the data to prepare for ML application.
    """

    df_text_abstract_split = prepare_series_text_split(df['abstract'])

    bigram_transformer = gensim.models.Phrases(df_text_abstract_split, min_count=1, threshold=2, delimiter=b' ')
    text_with_bigrams = bigram_transformer[df_text_abstract_split.values.tolist()]

    model = gensim.models.Word2Vec(text_with_bigrams, size=64, window=8, min_count=5, workers=4)
    word2vec = model.wv

    tvec_full = TfidfVectorizer(analyzer='word',
                                tokenizer=dummy_fun,
                                preprocessor=dummy_fun,
                                token_pattern=None,min_df=.0025, max_df=0.4)

    tvec_full.fit(text_with_bigrams)

    idf_weighted_vectors = get_mean_vectors_idf(text_with_bigrams,word2vec,tvec_full)

    df_vectorized = pd.DataFrame(idf_weighted_vectors, index=df_text_abstract_split.index)

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_vectorized),columns=df_vectorized.columns,index=df_vectorized.index)

    return df_scaled


def get_clusters(df,nclust=4,ntopics=5):
    """
    Takes as an input the dataframe containing articles with their abstract, applies a kmeans clustering algorithm,
    then returns the label as well as coordinates of the clusters centers.

    Inputs:
    ------
    df: the dataframe containing the data for the articles
    nclust: the number of Clusters
    ntopics: the number of words to represent each cluster

    Outputs:
    ------
    df_scaled: dataframe with each abstract cleaned and vectorized.
    df_label: the original dataframe with a new column containing the labels
    most_frequentwords_dict: a dict relating each labels to its highest idf words

    """

    df_scaled = prepare_for_kmeans(df)


    kmeans = KMeans(n_clusters=nclust, random_state=0,n_init=30,max_iter=1000).fit(df_scaled)

    SeriesLabel = pd.Series(kmeans.labels_,index=df_scaled.index)

    df_label = df.copy()

    df_label['Label'] = SeriesLabel
    print('Labels Created')

    most_frequentwords_dict = get_highest_idf_words_title(df_label,ntopics=ntopics)

    return df_scaled,df_label,most_frequentwords_dict

def get_recommendation(recid,df_label,df_scaled,nrecommendations=5):

    """
    For a given article (identified by its recid), find the nrecommendations in its cluster that are the closest
    according to their cosine distance.

    Inputs:
    ------
    recid: the identifier of the article
    df_label: the dataframe containing the data for the articles and their label
    df_scaled: the dataframe containing the vectorized version of the abstract of each article
    nrecommendations: the number of words to represent each cluster

    Output:
    ------
    most_sim_articles: an array of nrecommendations recid, sorted from most similar to least

    """

    label_article = df_label.loc[recid,'Label']
    vector_article = df_scaled.loc[recid]

    df_scaled_same_label = df_scaled[(df_label['Label']==label_article)]

    cosine_dist = np.array([compute_cosine_sim(vector_article,vec) for vec in df_scaled_same_label.values])



    most_sim_articles = df_scaled_same_label.index[cosine_dist.argsort()[::-1][1:nrecommendations+1]]

    return most_sim_articles
