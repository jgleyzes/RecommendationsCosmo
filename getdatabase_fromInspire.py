import os
# Configure settings for project
# Need to run this before calling models from application!
os.environ.setdefault('DJANGO_SETTINGS_MODULE','AcademicResults.settings')
THIS_PATH = os.path.dirname(os.path.realpath(__file__))
import django
# Import settings
django.setup()

from app_compare.models import Author,Article,Suggestions,Tags
import suggestions

import requests
import pandas as pd
import numpy as np
import unidecode
import scipy

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
        try:
            df_final.loc[recid,'title'] = entry['title']['title']
        except:
            print(recid)
        if type(entry['abstract']) == list:
            try:
                abstract = entry['abstract'][0]['summary']
            except:
                abstract = entry['abstract'][1]['summary']
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


    URL_BASE = 'http://inspirehep.net/search?p=astro-ph.CO+and+hep-th&of=recjson&ot=number_of_citations,authors,title,creation_date,recid,abstract&'
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
    print('Database Imported!')

    # Use the functions from the suggestions.py module to cluster the articles into nclust clusters.
    # returns df_scaled which a dataframe for the vectorized representation of the abstracts
    # df_label is the initial dataframe with a new column containing the labels.
    # most_frequentwords_dict is a dictionary which returns the ntopics most representative words of each cluster

    df_scaled,df_label,most_frequentwords_dict = suggestions.get_clusters(df_clean,nclust=4,ntopics=5)

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



        new_article = Article.objects.get_or_create(title=title,recid=recid,abstract=abstract,
                                                   creation_date=creation_date,citation_count=citation_count,
                                                   slug=recid
                                                   )[0]


        list_authors = get_authors_names_entry(df_label.loc[recid,'authors'])

        list_suggestions = suggestions.get_recommendation(recid,df_label,df_scaled,nrecommendations=5)

        new_article.suggestion.clear()
        new_article.tags.clear()
        # Attributes tag (most frequent words for each cluster) to the article
        for tag in most_frequentwords:
            new_tag = Tags.objects.get_or_create(nametag=tag)[0]
            new_tag.save()
            new_article.tags.add(new_tag)


        # Attributes authors to the article
        for author in list_authors:
            new_author = Author.objects.get_or_create(name=author)[0]
            new_author.save()
            new_article.authors.add(new_author)
            new_author.article_set.add(new_article)


        # Attributes suggestions to the article, using the clusters and retaining the 5 articles closest to
        # the one at hand according to their cosine distance in the vectorized space.
        for suggestion in list_suggestions:
            entry = df_label.loc[suggestion]
            try:
                 title = entry['title']
            except:
                print('Error with recid %s' %recid)

            new_suggestion = Suggestions.objects.get_or_create(title=title,recid=suggestion,
                                                       slug=suggestion)[0]
            new_suggestion.save()
            new_article.suggestion.add(new_suggestion)
            new_suggestion.article_set.add(new_article)


        new_article.save()

    print('Populating Completed!')


if __name__ == '__main__':
    main()
