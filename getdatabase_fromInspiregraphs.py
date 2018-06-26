import os
# Configure settings for project
# Need to run this before calling models from application!
os.environ.setdefault('DJANGO_SETTINGS_MODULE','AcademicResults.settings')
THIS_PATH = os.path.dirname(os.path.realpath(__file__))
import django
# Import settings
django.setup()

from app_compare.models import Author,Article,Suggestions,Tags
import suggestions_graphs
import graphdata
import requests
import pandas as pd
import numpy as np
import unidecode
import scipy
from haystack.management.commands import update_index
"""
This is the main code which populates the django database. It first download the information about articles from
INSPIRE, cleans it, then apply the method in suggestions_graphs to create the recommendations, as well as an
adjacency matrix representing the graphs of recommendations. This graph is used to cluster articles into groups,
which can be found in df_label ['Label'] column. The actual populating is then done.
Finally, using the graphdata module, we create a JSON file which describes the graphs between the different groups,
which will allow use to give its interactive representation using the D3.js javascript library.

"""




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
    for recid in df.index:
        entry = df.loc[recid]

        # Title
        if type(entry['title']) == list:
            df_final.loc[recid,'title'] = entry['title'][0]['title']
        else:
            try:
                df_final.loc[recid,'title'] = entry['title']['title']
            except:
                raise Exception(str(recid))

        # Abstract
        if type(entry['abstract']) == list:
            try:
                arxivindex = np.argwhere([ 'arXiv' in element['number'] for element in entry['abstract']])[0][0]
                abstract = entry['abstract'][arxivindex]['summary']
            except:
                abstract = entry['abstract'][1]['summary']
        else:
            if entry['abstract'] == None:
                abstract = ''
            else:
                abstract = entry['abstract']['summary']
        df_final.loc[recid,'abstract'] = abstract
    return df_final

def main():

    """
    Combines everything and populates the django database.

    """

    # Get the article corresponding to the arXiv tags astro-ph.CO and hep-th (check inspire-hep page on API for tags)
    # We request the fields number_of_citations,authors,title,creation_date,recid,abstract. Recid is a unique id attributed
    # to a given article on inspire.


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

    # Use the functions from the suggestions.py module to cluster the articles into nclust clusters.
    # returns df_scaled which a dataframe for the vectorized representation of the abstracts
    # df_label is the initial dataframe with a new column containing the labels.
    # most_frequentwords_dict is a dictionary which returns the ntopics most representative words of each cluster

    adjacency_matrix,df_vectorized,df_label,most_frequentwords_dict = suggestions_graphs.get_clusters(df_clean_with_abstract_and_authors)

    df_label = df_label.dropna()
    print('Clusters created!')


    #Populate the django models

    for recid in df_label.index:

        entry = df_label.loc[recid]
        try:
             title = entry['title']
        except:
            print('Error with recid %s' %recid)

        creation_date = entry['creation_date'][:10]
        citation_count = entry['number_of_citations']

        abstract = entry['abstract']

        label = df_label.loc[recid,'Label']
        most_frequentwords = most_frequentwords_dict[int(label)]

        try:
            arXiv_number =  get_arXiv_number_entry(entry)
        except:
            raise Exception(str(recid))
        arXiv_link = get_arXiv_link(arXiv_number)

        list_authors = get_authors_names_entry(df_label.loc[recid,'authors'])

        full_title = add_nameanddate_title(list_authors,creation_date,title)


        new_article = Article.objects.get_or_create(recid=recid,defaults={'title':full_title,'abstract':abstract,
                                                   'creation_date':creation_date,'citation_count':citation_count,'arXiv_link':arXiv_link,
                                                   'slug':recid,'label':label}
                                                   )[0]




        list_suggestions = suggestions_graphs.get_recommendation(recid,adjacency_matrix,df_label,nrecommendations=15)

        new_article.suggestion.clear()
        new_article.tags.clear()

        # Attributes tag (most frequent words for each cluster) to the article
        for tag,weight in most_frequentwords:
            new_tag = Tags.objects.get_or_create(nametag=tag,defaults={'weight':weight,'label':label})[0]
            new_tag.save()
            new_article.tags.add(new_tag)


        # Attributes authors to the article
        for author in list_authors:
            new_author = Author.objects.get_or_create(name=author)[0]
            new_author.save()
            new_article.authors.add(new_author)
            new_author.article_set.add(new_article)


        # Attributes suggestions with their strength to the article, retaining the 5 articles closest to
        # the one at hand according to their cosine distance in the vectorized space, provided they are close enough.
        for suggestion,strength in list_suggestions:
            entry = df_label.loc[suggestion]
            try:
                 title_sug = entry['title']
            except:
                print('Error with recid %s' %recid)
            creation_date_sug = entry['creation_date'][:10]
            label_sug = entry['Label']
            list_authors_sug = get_authors_names_entry(df_label.loc[suggestion,'authors'])
            citation_count_sug = entry['number_of_citations']

            full_title_sug = add_nameanddate_title(list_authors_sug,creation_date_sug,title_sug)

            new_suggestion = Suggestions.objects.get_or_create(recid=suggestion,strength=strength,defaults={'title':full_title_sug,
                                                       'slug':suggestion,'label':label_sug,'citation_count':citation_count_sug})[0]
            new_suggestion.save()
            new_article.suggestion.add(new_suggestion)
            new_suggestion.article_set.add(new_article)


        new_article.save()

    print('Populating Completed!')
    update_index.Command().handle()
    print('Index updated!')
    # Save the graph data to a JSON file to use for plotting

    datajson = graphdata.graph_to_json(adjacency_matrix,df_label,most_frequentwords_dict,nmostcited=3)

    print('JSON saved!')


if __name__ == '__main__':
    import time
    print('running at ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())))
    main()
