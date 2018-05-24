from django.db import models
from django.contrib.postgres.fields import ArrayField
from collections import Counter
# Create your models here.

class Author(models.Model):


    name = models.TextField(blank=True)

    def get_tags(self):
        tags = []
        for article in self.article_set.all():
            tags_article = article.tags.all()
            for tag in tags_article:
                tags.append(tag.nametag)
        counted_tags = Counter(tags)
        sorted_tags = sorted(counted_tags,key=counted_tags.get,reverse=True)
        tags = sorted_tags

        return tags

    def get_h_index(self):
        citations = []
        for article in self.article_set.all():
            citations.append(article.citation_count)
        h = 0
        citations.sort(reverse=True)
        for citation in citations:
            if citation >= h + 1:
                h += 1
            else:
                break
        return h

    def get_tot_citations(self):
        citations = 0
        for article in self.article_set.all():
            citations += article.citation_count
        return citations



    def get_number_article(self):

        return len(self.article_set.all())

    def get_avg_citations(self):
        citations = 0
        for article in self.article_set.all():
            citations += article.citation_count
        return '{:.2f}'.format(float(citations)/len(self.article_set.all()))


    def __str__(self):
        return self.name

class Suggestions(models.Model):
        title = models.TextField(blank=True)
        recid = models.TextField(blank=True)
        slug = models.SlugField(blank=True)
        def get_inspire_link(self):
            return "https://inspirehep.net/record/"+self.recid

        def __str__(self):
            return self.title

class Tags(models.Model):
    nametag = models.TextField(blank=True)

    def __str__(self):
        return self.nametag


class Article(models.Model):

    title = models.TextField(blank=True)
    authors = models.ManyToManyField(Author)
    creation_date = models.TextField(blank=True)
    recid = models.TextField(blank=True)
    citation_count = models.IntegerField(blank=True)
    abstract = models.TextField(blank=True)
    suggestion = models.ManyToManyField(Suggestions)
    tags = models.ManyToManyField(Tags)
    slug = models.SlugField(blank=True)
    arXiv_link = models.TextField(blank=True)


    def get_inspire_link(self):
        return "https://inspirehep.net/record/"+self.recid

    def __str__(self):
        return self.title
