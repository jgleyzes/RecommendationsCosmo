import datetime
from haystack import indexes
from app_compare.models import Article,Author


class ArticleIndex(indexes.SearchIndex, indexes.Indexable):
    text = indexes.CharField(document=True, use_template=True)
    title = indexes.CharField(model_attr='title')
    abstract = indexes.CharField(model_attr='abstract')
    creation_date = indexes.CharField(model_attr='creation_date')
    authors = indexes.CharField()

    def prepare_authors(self, obj):
            return [ author.name for author in obj.authors.all()]

    def get_model(self):
        return Article

    def index_queryset(self, using=None):
        """Used when the entire index for model is updated."""
        return self.get_model().objects.all()


# class AuthorIndex(indexes.SearchIndex, indexes.Indexable):
#     text = indexes.CharField(document=True, use_template=True)
#     name = indexes.CharField(model_attr='name')
#     articles = indexes.CharField()
#
#     def prepare_articles(self, obj):
#             return [ article.title for article in obj.article_set.all()]
#
#     def get_model(self):
#         return Author
#
#     def index_queryset(self, using=None):
#         """Used when the entire index for model is updated."""
#         return self.get_model().objects.all()
