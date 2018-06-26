This is the source code for a website found at http://jgleyzes.pythonanywhere.com/, where one can enter the name of an article published in the categories
Cosmology and Nongalactic Astrophysics cross High Energy Physics - Theory on arXiv and find recommendations for similar reads.

One can also enter the name of an author, find his publications, and find recommendations for those publications.The database is obtained from the API of https://inspirehep.net/.

The recommendations are made by vectorizing the abstract of the articles using word embeddings (more precisely, word2vec from the gensim library) and then keeping the most similar articles. We also group articles using a markov clustering algorithm on the graphs of recommendations.
