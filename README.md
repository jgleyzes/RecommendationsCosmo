This is the source code for a website found at http://jgleyzes.pythonanywhere.com/, where one can enter the name of an article published in the categories
Cosmology and Nongalactic Astrophysics cross High Energy Physics - Theory on arXiv and find recommendations for similar reads.

One can also enter the name of an author, find his publications, and find recommendations for those publications.

The recommendations are made by vectorizing the abstract of the articles using word embeddings (more precisely, word2vec from the gensim library) and then using a kmeans clustering algorithms.
For a given article, only the five publications which are closest to it in the word2vec space (in terms of cosine distance) are displayed.
