from django.db import models

# Create your models here.



class Adj_list(models.Model):
    recid = models.TextField(blank=True)
    adj_list = models.TextField(blank=True)

    def get_list_recid_weights(self):

        list_terms = self.adj_list.split('-')
        list_tuples = [(int(x.split(', ')[0][1:]),float(x.split(', ')[1][:-1])) for x in list_terms]

        return list_tuples

class Tag_list(models.Model):
    label = models.TextField(blank=True,unique=True)
    tag_list = models.TextField(blank=True)

    def get_list_tags_weights(self):

        list_terms = self.tag_list.split('-')
        list_tuples = [x.split(', ')[0][1:].replace("'",'') for x in list_terms]

        return list_tuples




class Article(models.Model):


    title = models.TextField(blank=True)
    label = models.TextField(blank=True)
    creation_date = models.TextField(blank=True)
    recid = models.TextField(blank=True)
    authors = models.TextField(blank=True)
    citation_count = models.IntegerField(blank=True)
    abstract = models.TextField(blank=True)
    slug = models.SlugField(blank=True)
    arXiv_link = models.TextField(blank=True)

    def get_list_authors(self):
        list_authors = self.authors.split('-')
        return list_authors



    def get_inspire_link(self):
        return "https://inspirehep.net/record/"+self.recid

    def __str__(self):
        return self.title
