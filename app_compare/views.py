from django.shortcuts import render
from django.views.generic import (View,TemplateView,
                                  ListView,DetailView,
                                  CreateView,UpdateView,
                                  DeleteView)
from app_compare import models
from django.urls import reverse_lazy
from app_compare import forms
from django.http import HttpResponse
from app_compare.models import Author,Article,Suggestions
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator


# Create your views here.


def search_form(request):
    return render(request, 'search_form.html')


def search(request):
    if 'qauthor' in request.GET and request.GET['qauthor'] != '':
        name = request.GET['qauthor']
        authors_list = Author.objects.filter(name__icontains=name)

        paginator = Paginator(authors_list, 25)
        page = request.GET.get('page')

        authors = paginator.get_page(page)

        return render(request, 'results.html',
                      {'authors': authors,'magic_url': request.get_full_path()})
    elif 'qarticle' in request.GET:

        query = request.GET['qarticle']

        words_query = query.split()

        articles_list_full = Article.objects.filter(title__icontains=words_query[0])
        for words in words_query:

            articles_list = Article.objects.filter(title__icontains=words)
            articles_list_full = list(set(articles_list_full) & set(articles_list))

        paginator = Paginator(articles_list_full, 25)
        page = request.GET.get('page')

        articles = paginator.get_page(page)


        return render(request, 'results.html',
                      {'articles': articles,'magic_url': request.get_full_path()})

    else:
        message = 'You submitted an empty form.'
        return HttpResponse(message)

class AboutView(TemplateView):
    template_name = "about.html"

class AuthorListView(ListView):
    model = models.Author
    context_object_name = 'authors'
    paginate_by = 10



class AuthorDetailView(DetailView):
    context_object_name = 'author_details'
    model = models.Author
    template_name = 'app_compare/author_detail.html'



class ArticleListView(ListView):
    model = models.Article
    context_object_name = 'articles'


class ArticleDetailView(DetailView):
    context_object_name = 'article_details'
    model = models.Article
    template_name = 'app_compare/article_detail.html'

class SuggestionsDetailView(DetailView):
    context_object_name = 'suggestions_details'
    model = models.Suggestions
    template_name = 'app_compare/article_detail.html'
