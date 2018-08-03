import os
# Configure settings for project
# Need to run this before calling models from application!
os.environ.setdefault('DJANGO_SETTINGS_MODULE','AcademicResults.settings')
THIS_PATH = os.path.dirname(os.path.realpath(__file__))
import django
# Import settings
django.setup()

from app_compare.models import Article,Adj_list,Tag_list
import suggestions_graphs
import graphdata
import requests
import pandas as pd
import numpy as np
import unidecode
import scipy
import pickle
from haystack.management.commands import update_index

"""
This is the main code which populates the django database. It first download the information about articles from
INSPIRE, cleans it, then apply the method in suggestions_graphs to create the recommendations, as well as an
adjacency matrix representing the graphs of recommendations. This graph is used to cluster articles into groups,
which can be found in df_label ['Label'] column. The actual populating is then done.
Finally, using the graphdata module, we create a JSON file which describes the graphs between the different groups,
which will allow use to give its interactive representation using the D3.js javascript library.

"""


def list_tuples_to_str(arr):

    str_sug = '-'.join([str(x) for x in arr])
    return str_sug




def get_json_to_df(url,df=[]):

    """
    Get a url for the inspire-hep api and store the json information to a dataframeself.

    Input:
    -----
    url: a string for the url
    df: a dataframe. Default is empty, but if one is provided, appends the new dataframe to it

    Output:
    -----
    respons_json: the actual json response to the API query. This is to check its length and see if the whole database
    has been accessed.
    df_res: the dataframe with the query stored.
    """
    response = requests.get(url)
    response_json = response.json()
    df_aux = pd.DataFrame(response_json)
    if len(df)>0:
        df_res = df.append(df_aux)
    else:
        df_res = df_aux

    return response_json,df_res

def get_authors_names_entry(array_author):

    """
    Get the name from the 'authors' entry of the dataframe.

    Input:
    -----
    array_author: a single entry from the dataframe corresponding to the columns 'authors', which returns a list.

    Output:
    ------
    list_authors: a list with the name of the authors for the article.
    """


    list_authors = []
    for element in array_author:
        firstname = unidecode.unidecode(element['first_name']).replace("'","")
        lastname = unidecode.unidecode(element['last_name']).replace("'","")
        fullname = unidecode.unidecode(element['full_name']).replace("'","")
        if firstname == '':
            firstname = fullname.split(' ')[0]
            lastname = ' '.join(fullname.split(' ')[1:])
            name = lastname + ', ' + firstname[0] +'.'

        else:
            if len(firstname.split(' '))>0:
                firstnames = firstname.split(' ')

                try:
                    name = lastname + ', ' + firstnames[0][0]+'.'
                except:
                    print(fullname)

            else:
                name = lastname + ', ' + firstname[0] +'.'
        list_authors.append(name)
    return list_authors

def get_title_entry(entrytitle):
    """
    Put the title column in the right shape

    Input:
    -----
    entrytitle: a title entry coming from the json data

    Output:
    ------
    The actual title
    """

    if type(entrytitle) == list:
        return entrytitle[0]['title']
    else:
        try:
            return entrytitle['title']
        except:
            raise Exception(entrytitle)

def get_abstract_entry(entryabs):
    """
    Put the abstract column in the right shape

    Input:
    -----
    entryabs: a abstract entry coming from the json data

    Output:
    ------
    The actual abstract
    """
    if type(entryabs) == list:
        try:
            arxivindex = np.argwhere([ 'arXiv' in element['number'] for element in entryabs])[0][0]
            abstract = entryabs[arxivindex]['summary']
        except:
            abstract = entryabs[1]['summary']
    else:
        if entryabs == None:
            abstract = ''
        else:
            abstract = entryabs['summary']
    return abstract

def get_arXiv_number_entry(entry):

    """
    Get the arXiv number for the entry of the dataframe.

    Input:
    -----
    entry: the entry from the database corresponding to one article

    Output:
    ------
    arXiv_number: The identifier of the article on arXiv.org
    """

    arXiv_number = 'N/A'

    if type(entry['system_control_number']) == list:
        for dictionary in entry['system_control_number']:
            if dictionary['institute'] == 'arXiv':
                values = dictionary.values()
                for value in values:
                    if 'arXiv.org' in value:

                        arXiv_number = value.split(':')[-1]
    else:
        if 'arXiv' in entry['system_control_number'].keys():
            values = dictionary.values()
            for value in values:
                if 'arXiv.org' in value:

                    arXiv_number = value.split(':')[-1]



    return arXiv_number


def get_arXiv_link(arXiv_number):

    """
    Get the arXiv link from the arXiv number (if existing).

    Input:
    -----
    arXiv_number: the arXiv number (or N/A if no number)

    Output:
    ------
    link: The link to the abstract page of the preprint on arxiv
    """

    if arXiv_number == 'N/A':
        return 'N/A'
    else:
        link = 'https://arxiv.org/abs/' + arXiv_number
        return link



def add_nameanddate_title(list_authors,date,title):

    """
    Add the names and date of publication to the title (format First Author et al, Year).

    Input:
    -----
    list_authors: the list of authors for the article
    date: the date of publication
    title: the actual title


    Output:
    ------
    new_title: The title with author name and date in parenthesis
    """


    if len(list_authors)==1:
        add_title = '(' + list_authors[0].split(',')[0]
    elif len(list_authors)==2:
        add_title = '(' + list_authors[0].split(',')[0] + ' & ' + list_authors[1].split(',')[0]
    else:
        add_title = '(' + list_authors[0].split(',')[0] + ' et al'

    Year = date[:4]

    add_title += ' ' + Year + ')'

    new_title = title+ ' ' + add_title
    return new_title


def prepare_df_clean(df):

    """
    Prepares the raw dataframe into a usable dataframe:
        - transforms the title field to a string
        - transforms the abstract field to a string

    Input:
    -----
    df: the dataframe obtained from the Inspire API.

    Output:
    ------
    the cleaned dataframe

    """

    df_final = df.copy()
    df_final['title'] = df_final['title'].apply(get_title_entry)
    df_final['abstract'] = df_final['abstract'].apply(get_abstract_entry)
    return df_final



def fill_Article_DB(entry,df_adjacency_matrix,df_label,nrecommendations=15):

    """
    Fills the django database with entry

    Input:
    -----
    entry: the row of the dataframe to put in the database
    df_adjacency_matrix: a dataframe representing the full adjacency matrix


    """

    # Prepare everything to fill the Article model

    recid = entry.name
    try:
         title = entry['title']
    except:
        print('Error with recid %s' %recid)

    creation_date = entry['creation_date'][:10]
    citation_count = entry['number_of_citations']

    abstract = entry['abstract']

    label = entry['Label']

    try:
        arXiv_number =  get_arXiv_number_entry(entry)
    except:
        raise Exception(str(recid))
    arXiv_link = get_arXiv_link(arXiv_number)

    list_authors = get_authors_names_entry(entry['authors'])

    full_title = add_nameanddate_title(list_authors,creation_date,title)

    authors = '-'.join(list_authors)

    # Fills database entry corresponding to a given recid. If existing, updates the fields.

    new_article = Article.objects.get_or_create(recid=recid,defaults={'title':full_title,'abstract':abstract,
                                               'creation_date':creation_date,'citation_count':citation_count,'arXiv_link':arXiv_link,
                                               'slug':recid,'label':label,'authors':authors}
                                               )[0]
    new_article.save()

    # Fills the corresponding entry in the adjacency list

    list_suggestions = suggestions_graphs.get_recommendation(recid,df_adjacency_matrix,nrecommendations=nrecommendations)

    str_sug = list_tuples_to_str(list_suggestions)

    new_sug = Adj_list.objects.get_or_create(recid=recid,defaults={'adj_list':str_sug})[0]
    new_sug.save()

def fill_tags_DB(most_frequentwords_dict):

    """
    Fills the django database with entry

    Input:
    -----
    most_frequentwords_dict


    """
    for label in most_frequentwords_dict:
        str_tag = list_tuples_to_str(most_frequentwords_dict[label])
        new_tags = Tag_list.objects.get_or_create(label=label,defaults={'tag_list':str_tag})[0]
        new_tags.save()

def main(nrecommendations=15,inflation=1.7,threshold_percent=0.05,nmostcited=3,ntopics=5):

    """
    Combines everything and populates the django database.

    Input:
    -----
    nrecommendations : the number of recommendations to give per article
    inflation: controls the markov clustering (larger values mean more groups)
    threshold_percent: keeps groups whose size are larger than threshold_percent*total_size
    nmostcited: for each group will save the nmostcited most cited articles in the group
    ntopics: for each group, gives the ntopics words most frequent among the titles of said group


    """

    # Get the article corresponding to the arXiv tags astro-ph.CO and hep-th (check inspire-hep page on API for tags)
    # We request the fields number_of_citations,authors,title,creation_date,recid,abstract. Recid is a unique id attributed
    # to a given article on inspire.

    t0  = time.time()

    URL_BASE = 'http://inspirehep.net/search?p=astro-ph.CO+and+hep-th&of=recjson&ot=number_of_citations,authors,title,creation_date,recid,abstract,system_control_number&'
    response_json,df_res = get_json_to_df(URL_BASE+'&rg=250',df=[])
    i = 1
    while len(response_json)>=250:
        start_point = 'jrec={}'.format(251*i)
        url = URL_BASE + start_point + '&rg=250'
        response_json,df_res = get_json_to_df(url,df_res)
        print(i)
        i = i+1
    df_res = df_res.set_index('recid')
    df_clean = prepare_df_clean(df_res)

    df_clean_with_abstract_and_authors = df_clean[(df_clean['abstract'] != '')&(df_clean['authors'].astype(str).ne('None'))]
    print('Database Imported!')

    """
    Use the functions from the suggestions_graphs.py module to group the articles into similar clusters.
    Returns:
    - adjacency_matrix from the graph of distances between articles
    - df_vectorized: the dataframe with the vector representation of each article
    - df_label: the original dataframe with an extra column indicating the cluster the article belongs to.
    -  most_frequentwords_dict is a dictionary which returns the ntopics most representative words of each cluster
    """

    df_adjacency_matrix,df_vectorized,df_label,most_frequentwords_dict = suggestions_graphs.get_clusters(df_clean_with_abstract_and_authors,inflation=inflation,ntopics=ntopics)
    print('Clusters created!')

    df_label = df_label.dropna()



    #Populate the django models

    #objects = [get_bulk_DB(entry) for _,entry in df_label.iterrows()]

    #Article.objects.bulk_create(objects)
    df_label.apply(lambda x:fill_Article_DB(x,df_adjacency_matrix,df_label,nrecommendations=nrecommendations),axis=1)
    fill_tags_DB(most_frequentwords_dict)


    print('Populating Completed after %s' %(time.time()-t0))

    # Save the graph data to a JSON file to use for plotting

    datajson = graphdata.graph_to_json(df_adjacency_matrix.values,df_label,most_frequentwords_dict,nmostcited=nmostcited,threshold_percent=threshold_percent)

    print('JSON saved!')

    update_index.Command().handle()
    print('Index updated!')


if __name__ == '__main__':
    import time
    t0 = time.time()
    print('running at ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))
    main()
    print('it took %s seconds to fill the database'%(time.time()-t0))
