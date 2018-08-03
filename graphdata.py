import pandas as pd
import numpy as np
import networkx as nx
import markov_clustering as mc
import json
import os

"""
This part of the program deals with preparing the data to show a summary graph between different groups (create by the clustering
of the full graph). From the full adjacency matrix, it creates a reduced adjacency matrix between labels, taking averages of
the full matrix. We retrieve more information about the groups (such as the most cited articles within them), and combines
everything into a json file, which we will feed to a d3.js force directed graph.
"""


THIS_PATH = os.path.dirname(os.path.realpath(__file__))



def get_clusters_ids(df_label):

    """
    Gets the indices of the adjacency matrix that corresponds to each label in df_label

    Input:
    -----
    df_label: the dataframe with all the information on the articles, in particular the labels

    Output:
    ------
    dict_cluster_id: a dictionary linking each label to its corresponding indices

    """

    labels = np.unique(df_label['Label'].values)
    dict_cluster_recid = {}
    dict_cluster_id = {}

    for label in labels:
        cluster_recid = df_label[df_label['Label']==label].index.values
        cluster_id = [df_label.index.get_loc(recid) for recid in cluster_recid]
        dict_cluster_id[label] = cluster_id

    return dict_cluster_id


def put_indice_shape_LR(indices,leftorright):
    """
    Given a set of indices, turns them into the correct shape to use as mask for the matrix:
    if right ('R') indices, puts in the shape (1,n)
    if left ('L') indices, puts in the shape (n,1)
    """

    n_indices = indices.shape[0]
    if leftorright == 'L':
        return indices.reshape((n_indices,1))
    elif leftorright == 'R':
        return indices.reshape((1,n_indices))
    else:
        raise Exception('leftorright must be either "L" or "R", not {} '.format(leftorright))


def get_mean_adjacency_matrix(adjacency_matrix,indices_left,indices_right):

    """
    Selects the submatrix of adjacency_matrix corresponding to the (i,j) in [indices_left,indices_right]
    and takes its average.
    """

    adjacency_matrix_red = adjacency_matrix[indices_left,indices_right]

    return adjacency_matrix_red.mean()



def get_adjacency_matrix_label(adjacency_matrix,df_label):

    """
    Computes a adjacency matrix for the different labels by taking average of the submatrices associated
    with each pair of label. The matrix is then normalized so that its maximum value is one.

    Input:
    -----
    adjacency_matrix: the adjacency matrix for all the articles
    df_label: the dataframe with all the information on the articles, in particular the labels

    Output:
    ------
    get_adjacency_matrix_label: the adjacency matrix which describe the graph of labels.

    """

    dict_cluster_id = get_clusters_ids(df_label)
    labels = list(dict_cluster_id.keys())
    nlabels = len(labels)

    total_size = df_label.shape[0]

    df_groupedby_label = df_label.groupby('Label')
    size_groups = df_groupedby_label['title'].apply(np.size)

    adjacency_matrix_label = np.zeros((nlabels,nlabels))

    for i in range(nlabels):
            indices = np.array(dict_cluster_id[labels[i]])

            #Need to put the indices in right shape to keep the matrix shape

            indices_left = put_indice_shape_LR(indices,'L')
            indices_right = put_indice_shape_LR(indices,'R')

            adjacency_matrix_label[i,i] = get_mean_adjacency_matrix(adjacency_matrix,indices_left,indices_right)

            for j in range(i):
                indices = np.array(dict_cluster_id[labels[j]])
                indices_right = put_indice_shape_LR(indices,'R')

                adjacency_matrix_label[i,j] = get_mean_adjacency_matrix(adjacency_matrix,indices_left,indices_right)

                #matrix is symmetric

                adjacency_matrix_label[j,i] = get_mean_adjacency_matrix(adjacency_matrix,indices_left,indices_right)

    adjacency_matrix_label = adjacency_matrix_label/adjacency_matrix_label.max()

    return adjacency_matrix_label


def get_info_topcitation(label,df_label,nmostcited=3):
    """
    For a given label, gets the recid of the three most cited article in the category, and return links to their
    hep-inspire record as well as their title. This is to be fed to the label graph so it has a json type structure.

    Input:
    -----
    label: the label for which we want to get the info
    df_label: the dataframe with all the information on the articles, in particular the labels
    nmostcited: number of most cited records to keep. Default is 3.

    Output:
    ------
    infotopncited: a list of dictionary with the link and title of the most cited articles with the given label.

    """

    df_groupedby_label = df_label.groupby('Label')
    topnmostcitedlabel = df_groupedby_label['number_of_citations'].nlargest(nmostcited)

    series_label = topnmostcitedlabel[label]

    recids = np.array(list(series_label.index))

    infotopncited  = [{'recid':int(recid),'title':df_label.loc[recid,'title']} for recid in recids]

    return infotopncited



def graph_to_json(adjacency_matrix,df_label,most_frequentwords_dict,nmostcited=3,threshold_percent=0.05):

    """
    Compute the json-type structure for the nodes and links of the d3.js force-directed graph.
    The nodes have into about the most frequent words in the group, as well as the 3 most cited articles of the group.

    Input:
    -----
    adjacency_matrix: the adjacency matrix for the pair of articles
    df_label: the dataframe with all the information on the articles, in particular the labels
    most_frequentwords_dict: dictionary which contains the most frequent words for each label
    mostcited: number of most cited records to keep. Default is 3.
    threshold_percent: keeps groups whose size are larger than threshold_percent*total_size

    Output:
    ------
    datajsonlabel: a dictionary of lists with the info on the nodes and links.

    """
    datajsonlabel = {}
    datajsonlabel['nodes'] = []
    datajsonlabel['links'] = []

    labels = np.unique(df_label['Label'].values)
    nlabels = len(labels)

    df_groupedby_label = df_label.groupby('Label')
    size_groups = df_groupedby_label['title'].apply(np.size)

    total_size = df_label.shape[0]
    adjacency_matrix_label = get_adjacency_matrix_label(adjacency_matrix,df_label)

    #get the min-max distance to normalize adjacency_matrix_label (reduced to the groups large enough) between 0 and 1

    labels_red = size_groups[size_groups>threshold_percent*total_size].index.values
    adjacency_matrix_label_red = adjacency_matrix_label[labels_red.reshape((1,-1)).astype(int),labels_red.reshape((-1,1)).astype(int)]

    min_distance = adjacency_matrix_label_red[adjacency_matrix_label_red>0].min()

    max_distance = adjacency_matrix_label_red[adjacency_matrix_label_red>0].max()



    for i in range(nlabels):

        label = labels[i]
        if size_groups.loc[label] > threshold_percent*total_size:
            most_frequentwords = most_frequentwords_dict[label]

            #get the min-max weight to normalize weigths
            weigths = np.array([float(weight) for _,weight in most_frequentwords])

            minweight = weigths[weigths>0].min()

            maxweight = weigths[weigths>0].max()
            if minweight == maxweight:
                dictinfo = [{'text':word,'size':1.} for word,weight in most_frequentwords]
            else:
                dictinfo = [{'text':word,'size':(weight-minweight)/(maxweight-minweight)+0.5} for word,weight in most_frequentwords]
            size = int(size_groups[label])

            infotopncited = get_info_topcitation(label,df_label,nmostcited=nmostcited)

            node = {'id':i,'info':dictinfo,'group':labels[i],'size':size,'infotopn':infotopncited}
            datajsonlabel['nodes'].append(node)

            for j in range(0,i):
                if adjacency_matrix_label[i,j]>0 and size_groups.loc[labels[j]] > threshold_percent*total_size:

                    value = int(round(30*(adjacency_matrix_label[i,j]-min_distance)/(max_distance-min_distance)))
                    link = {'source':i,'target':j,"value":value}
                    datajsonlabel['links'].append(link)

    with open(os.path.join(THIS_PATH,'data/datalabels.json'), 'w') as fp:
         json.dump(datajsonlabel, fp)
    return adjacency_matrix_label,datajsonlabel
