from django.urls import path,re_path
from app_compare import views

# TEMPLATE URLs
app_name = 'app_compare'

urlpatterns = [
    path('authors/',views.AuthorListView.as_view(),name='author_list'),
    re_path('authors/(?P<pk>\d+)/',views.AuthorDetailView.as_view(),name='author_detail'),
    path('articles/',views.ArticleListView.as_view(),name='article_list'),
    re_path('articles/(?P<slug>\d+)/',views.ArticleDetailView.as_view(),name='article_detail')

]
