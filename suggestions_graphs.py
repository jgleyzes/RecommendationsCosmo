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
import networkx as nx
import markov_clustering as mc


"""
This part of the program deals with vectorizing the abstracts for the articles, then computes the cosine distance between all pairs
of articles in that vector space. From there, it computes an adjacency matrix, when links are create between articles when
they are in the 5% closest in distance. This adjacency matrix defines a graph, over which we can run a clustering algorithm
to create group of articles. By looking at titles in each group, we can get the keywords that represent each label.
Finally, we get to the actual recommendation part, where for each articles, we keep the 5 closest in the graph.
"""


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
    if entry == '':
        sentence_nouns_only = ''
    else:
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

    series_text = series.str.replace(r'<inline-formula>.*?</inline-formula>','')\
                                .str.lower()\
                                .map(get_df_nouns_only)\
                                .str.translate(remove_punctuation)\
                                .str.translate(remove_numbers)\
                                .str.replace('\W\w\W','')\
                                .map(lambda x:unidecode.unidecode(x))\
                                .str.split()\
                                .map(lambda entry: [w for w in entry if not w in stop_words])

    lengthvectors = np.array([len(x) for x in series_text])

    series_text = series_text[lengthvectors>0]
    return series_text


def get_highest_idf_words_title(df_label,ntopics=5):

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


    df_grouped = df_label.groupby(['Label']).agg({'title':' '.join})
    df_joined = prepare_series_text_split(df_grouped['title']).map(' '.join)

    tfidf = TfidfVectorizer(min_df=.009, max_df=0.8, stop_words='english', ngram_range=(1,1))
    transformed_corpus = tfidf.fit_transform(df_joined)
    tvocab = np.array(tfidf.get_feature_names())
    df_transformed = pd.DataFrame(transformed_corpus.todense(), columns=tvocab,index=df_joined.index)
    most_frequentwords_dict = {}
    for i in df_transformed.index:

        most_frequentwords_dict[i] = list(zip(df_transformed.loc[i].sort_values(ascending=False)[:ntopics].index,df_transformed.loc[i].sort_values(ascending=False)[:ntopics]))

    return most_frequentwords_dict


def prepare_vector(df):

    """
    Applies all the method above to clean the data, trains and computes the word2vec mapping as well as the
    TF-IDF vectorizer on the same tokens as word2vec.
    Also applies a StandardScaler to the data to prepare for ML application.
    """

    df_text_abstract_split = prepare_series_text_split(df['abstract'])

    model = gensim.models.Word2Vec(df_text_abstract_split, size=128, window=8, min_count=1, workers=4)
    model.train(df_text_abstract_split,total_examples=len(df_text_abstract_split),epochs=10)
    word2vec = model.wv

    tvec_full = TfidfVectorizer(analyzer='word',
                                tokenizer=dummy_fun,
                                preprocessor=dummy_fun,
                                token_pattern=None,min_df=.0025, max_df=0.4)

    tvec_full.fit(df_text_abstract_split)

    idf_weighted_vectors = get_mean_vectors_idf(df_text_abstract_split,word2vec,tvec_full)

    df_vectorized = pd.DataFrame(idf_weighted_vectors, index=df_text_abstract_split.index)




    return df_vectorized



def get_distance_matrix(df_vectorized):

    """
    Takes as an input the dataframe vectorized absctract them, then create a matrix
    M_ij = cosine_distance(entry_i,entry_j).

    Inputs:
    ------
    df_vectorized: the dataframe containing the vectorized abstracts


    Outputs:
    ------
    cosine_distance_matrix: the matrix M_ij = cosine_distance(entry_i,entry_j)

    """


    gram_matrix = np.dot(df_vectorized.values,df_vectorized.values.T)

    norms_matrix = np.sqrt(np.outer(np.diag(gram_matrix),np.diag(gram_matrix)))

    cosine_distance_matrix = gram_matrix/norms_matrix

    return cosine_distance_matrix


def get_stats_nodiag(cosine_distance_matrix,p=95):
    """
    Takes as an input the matrix M_ij = cosine_distance(entry_i,entry_j) and compute the value of the percentile "p"
    when ignoring the diagonal (which is only 1 by definition). This is because we want to keep only the connection
    with distance larger than this threshold. We also compute the standard deviation to rescale the distances.


    Inputs:
    ------
    cosine_distance_matrix: the matrix M_ij = cosine_distance(entry_i,entry_j)
    p: the percentile in percentage (default is 95%)


    Outputs:
    ------
    value_percentile: the value of the "p" percentile of the matrix without diagonal
    stdnogiag: the standard deviation of the matrix without diagonal


    """

    mask = np.ones(cosine_distance_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, 0)

    cosine_distance_matrix_nodiag = cosine_distance_matrix[mask]

    value_percentile = np.percentile(cosine_distance_matrix_nodiag,p)

    stdnogiag = cosine_distance_matrix_nodiag.std()

    return (value_percentile,stdnogiag)

def get_adjacency_matrix(cosine_distance_matrix,p=95):
    """
    We want to create graph of the articles where the weigths of the edges are given by the distance between articles
    in the vector space. We want to keep only the articles that are close enough, that is, whose distance is larger
    than a given threshold. By default, this threshold will be given by the p% percentile of the set of distances.
    In other words, two articles are connected if they are in the (1-p)% closest. To get more meaningful weigths, we will
    rescale it by the standard deviation of the set of distances.



    Inputs:
    ------
    cosine_distance_matrix: the matrix of distance between two articles
    p: the percentile for the threshold (default is 95%)

    Outputs:
    ------
    adjacency_matrix_scaled_above_thres: the adjacency matrix keeping only the distance above the threshold,
                                         and scaled by the initial standard deviation.


    """

    cosine_distance_matrix_diag0 = cosine_distance_matrix - np.diag(np.diag(cosine_distance_matrix))

    value_percentile,stdnogiag = get_stats_nodiag(cosine_distance_matrix,p=p)

    adjacency_matrix_scaled = (cosine_distance_matrix_diag0 - value_percentile)/stdnogiag

    mask_threshold = adjacency_matrix_scaled > 0

    adjacency_matrix_scaled_above_thres = adjacency_matrix_scaled*mask_threshold

    return adjacency_matrix_scaled_above_thres





def get_clusters(df,inflation=1.1,p=95,ntopics=5):
    """
    Takes as an input the dataframe containing articles with their abstract, vectorize it,
    then returns the label as well as coordinates of the clusters centers.

    Inputs:
    ------
    df: the dataframe containing the data for the articles
    inflation: parameter that enter in the clustering algorithm. Increase value for more clusters.
               default typically yield ~10 clusters.
    p: percentile for the threshold.

    Outputs:
    ------
    adjacency_matrix: the adjacency matrix
    df_vectorized: dataframe with each abstract cleaned and vectorized.
    df_label: the original dataframe with a new column containing the labels
    most_frequentwords_dict: a dict relating each labels to its highest idf words with their weight

    """

    df_vectorized = prepare_vector(df)

    df_label = df.copy()


    cosine_distance_matrix = get_distance_matrix(df_vectorized)

    adjacency_matrix = get_adjacency_matrix(cosine_distance_matrix,p=p)

    G = nx.from_numpy_matrix(adjacency_matrix)

    matrix = nx.to_scipy_sparse_matrix(G)
    result = mc.run_mcl(matrix, inflation=inflation)
    clusters = mc.get_clusters(result)


    index_df = df_vectorized.index
    clusters_recid = [index_df[np.array(cluster)] for cluster in clusters]

    for i,cluster in enumerate(clusters_recid):
        df_label.loc[cluster,'Label'] = i

    most_frequentwords_dict = get_highest_idf_words_title(df_label,ntopics=ntopics)

    return adjacency_matrix,df_vectorized,df_label,most_frequentwords_dict



def get_recommendation(recid,adjacency_matrix,df_label,nrecommendations=15):

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

    indextorecid = df_label.index.values

    index_article = np.argwhere(indextorecid==recid)[0][0]

    column_article = adjacency_matrix[index_article]

    recommendations = list(zip(indextorecid[column_article.argsort()[::-1]][:nrecommendations],column_article[column_article.argsort()[::-1]][:nrecommendations]))

    return recommendations
