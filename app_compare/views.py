from django.shortcuts import render
from django.views.generic import (View,TemplateView,
                                  ListView,DetailView,
                                  CreateView,UpdateView,
                                  DeleteView)
from app_compare import models
from app_compare.documents import articleDocument
from django.urls import reverse_lazy
from app_compare import forms
from django.http import HttpResponse,JsonResponse
from app_compare.models import Article,Adj_list,Tag_list
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
import json
from django.conf import settings
import os
import pandas as pd
import pickle
f = os.path.join( settings.BASE_DIR, 'data/datalabels.json' )
AM_path = os.path.join( settings.BASE_DIR, 'data/df_adjacency_matrix.csv' )
tags_path = os.path.join(settings.BASE_DIR,'data/dict_tags.pickle')
# Create your views here.




def search_form(request):
    return render(request, 'search/search.html')

def graph_json(request):
    datajson = open(f)
    datajson = json.load(datajson)
    return JsonResponse(datajson,safe=False)

def graph_label(request):
    return render(request, 'app_compare/graph_label.html')


class AboutView(TemplateView):
    template_name = "about.html"


class ArticleListView(ListView):
    model = models.Article
    context_object_name = 'articles'

class Suggestion:
    def __init__(self,recid,citation_count,strength,title):
        self.recid = recid
        self.citation_count = citation_count
        self.strength = strength
        self.title = title



class ArticleDetailView(DetailView):

    context_object_name = 'article_details'
    model = models.Article
    adj_list = models.Adj_list
    tag_list = models.Tag_list
    template_name = 'app_compare/article_detail.html'

    def get_citation_suggestions(self,recid,orderby = 'strength'):

        list_suggestions = self.adj_list.objects.get(recid=recid).get_list_recid_weights()

        dict_sug_prepared = {}
        dict_sug_prepared['recid'] = []
        dict_sug_prepared['strength'] = []
        dict_sug_prepared['citation_count'] = []
        dict_sug_prepared['title'] = []

        for recid_sug,strength in list_suggestions:
            suggestion_details = self.model.objects.get(recid=recid_sug)
            dict_sug_prepared['recid'].append(recid_sug)
            dict_sug_prepared['strength'].append(strength)
            dict_sug_prepared['citation_count'].append(suggestion_details.citation_count)
            dict_sug_prepared['title'].append(suggestion_details.title)

        df_suggestions = pd.DataFrame.from_dict(dict_sug_prepared).set_index('recid')
        df_suggestions.sort_values(by=[orderby],ascending=False,inplace=True)

        list_sug = [Suggestion(recid,df_suggestions.loc[recid,'citation_count'],df_suggestions.loc[recid,'strength'],df_suggestions.loc[recid,'title']) for recid in df_suggestions.index]

        return list_sug

    def get_context_data(self,**kwargs):
        context = super(ArticleDetailView, self).get_context_data(**kwargs)
        orderby = self.request.GET.get("orderby")
        if type(orderby) == type(None):
            orderby = 'strength'
        recid = self.kwargs['slug']
        article = self.model.objects.get(recid=recid)

        recommendations_list_full = self.get_citation_suggestions(recid,orderby=orderby)


        paginator = Paginator(recommendations_list_full, 5)
        page = self.request.GET.get('page')

        recommendations = paginator.get_page(page)

        label = article.label
        list_tags = self.tag_list.objects.get(label=label).get_list_tags_weights()

        list_authors = article.get_list_authors()
        context['authors'] = list_authors
        context['recommendations'] = recommendations
        context['tags'] = list_tags
        context['magic_url'] = self.request.get_full_path()

        return context
