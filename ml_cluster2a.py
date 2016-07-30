""" clustering and retrieval week 2 - programming assignment 1 """

import operator
import numpy as np                       # dense matrices
import pandas as pd
# import matplotlib.pyplot as plt          # plotting


from scipy.sparse import csr_matrix      # sparse matrices
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances as euc_dist


# %matplotlib inline


def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']

    return csr_matrix((data, indices, indptr), shape)


def unpack_dict(matrix, map_index_to_word):

    map_index_to_word.sort()
    table = list(map_index_to_word.index)
    data = matrix.data
    indices = matrix.indices
    indptr = matrix.indptr

    num_doc = matrix.shape[0]

    return [{k: v for k, v in zip([table[word_id] for word_id in indices[indptr[i]:indptr[i + 1]]],
                                  data[indptr[i]:indptr[i + 1]].tolist())} \
            for i in range(num_doc)]


def top_words(name, top_n = None):
    """
    Get a table of the most frequent words in the given person's wikipedia page.
    """
    row = wiki[wiki['name'] == name]
    word_count_table = row['word_count'].item()
    # return sorted(word_count_table.items(), key = lambda x: x[1], reverse = True)[:top_n]
    return sorted(word_count_table.items(), key = operator.itemgetter(1), reverse = True)[:top_n]


def get_common_rows(set_1, set_2, set_1_name = None, set_2_name = None):
    """ get common list of words between 2 names """
    words_1 = {k for k in dict(set_1).keys()}
    words_2 = {k for k in dict(set_2).keys()}
    common_words = words_1.intersection(words_2)
    common_1 = [dict(set_1)[i] for i in common_words]
    common_2 = [dict(set_2)[i] for i in common_words]

    name_1 = "set_1"
    name_2 = "set_2"
    if set_1_name:
        name_1 = set_1_name
    if set_2_name:
        name_2 = set_2_name

    df = pd.DataFrame({'words': [w for w in common_words],
                         name_1:common_1, name_2: common_2})
    return df[['words', name_1, name_2]]


def top_words_tf(name, top_n = None):
    """
    Get a table of the most frequent words in the given person's wikipedia page.
    """
    row = wiki[wiki['name'] == name]
    word_count_table = row['tf_idf'].item()
    # return sorted(word_count_table.items(), key = lambda x: x[1], reverse = True)[:top_n]
    return sorted(word_count_table.items(), key=operator.itemgetter(1), reverse=True)[:top_n]


#   Loading data
wiki = pd.read_csv('data\people_wiki.csv')
word_count = load_sparse_csr('data\people_wiki_word_count.npz')


# Finding kNN
model = NearestNeighbors(metric='euclidean', algorithm='brute')
model.fit(word_count)
distances, indices = model.kneighbors(word_count[35817],
                                      n_neighbors=10)  # 1st arg: word count vector


map_index_to_word = pd.read_json('data\people_wiki_map_index_to_word.json', typ = "Series")
wiki['word_count'] = unpack_dict(word_count, map_index_to_word)
wiki['words'] = [list(wiki['word_count'].ix[i].keys()) for i in range(len(wiki))]


# Number of wiki articles containing Obama's top 5 words
obama_words = top_words('Barack Obama')
barrio_words = top_words('Francisco Barrio')
common_rows = get_common_rows(obama_words, barrio_words, "Obama", "Barrio")
obama_top_5 = list(common_rows.sort_values('Obama', ascending = False)['words'][:5])
wiki['has_top_words'] = wiki['words'].apply(lambda x: set(obama_top_5).issubset(x))
print('{0:<6}{1}'.format(sum(wiki['has_top_words']),
                              'articles contain the top 5 words in Obama\'s article'))


# Checker for unique words
print('{0}{1:<8}{2:<5}'.format('our count: ', len(wiki.ix[32]['words']), 'correct_count: 167'))
print('{0}{1:<8}{2:<5}'.format('our count: ', len(wiki.ix[33]['words']), 'correct_count: 188'))


# Computing pairwise Euclidean Distance
bo_vec = word_count.getrow(wiki[wiki['name'] == 'Barack Obama'].index[0])
gb_vec = word_count.getrow(wiki[wiki['name'] == 'George W. Bush'].index[0])
jb_vec = word_count.getrow(wiki[wiki['name'] == 'Joe Biden'].index[0])
print('{0}{1:>8.4f} '.format('bo-gb:', euc_dist(bo_vec, gb_vec)[0][0]))
print('{0}{1:>8.4f} '.format('bo-jb:', euc_dist(bo_vec, jb_vec)[0][0]))
print('{0}{1:>8.4f} '.format('jb-gb:', euc_dist(jb_vec, gb_vec)[0][0]))
print('{0}'.format('jb-gb has the smallest distance'))


# Obama's top 10 words in a set of common words between Obama and Bush
obama_words = top_words('Barack Obama')
bush_words = top_words('George W. Bush')
common_rows = get_common_rows(obama_words, bush_words, 'Obama', 'Bush')
common_rows.sort('Obama', ascending=False)
for i, j in enumerate(list(common_rows.sort_values('Obama', ascending = False)['words'][:10])):
    print('{0:<4}{1}'.format(i, j))


# Extracting the TFIDF vectors
tf_idf = load_sparse_csr('data\people_wiki_tf_idf.npz')
wiki['tf_idf'] = unpack_dict(tf_idf, map_index_to_word)


# Finding NN using TFIDF
model_tf_idf = NearestNeighbors(metric='euclidean', algorithm='brute')
model_tf_idf.fit(tf_idf)
distances, indices = model_tf_idf.kneighbors(tf_idf[35817], n_neighbors=10)
for i in range(len(indices[0])):
    print('{0:<7}{1:^30}{2:<4.8f}'.format(indices[0][i],
                                          wiki.ix[indices[0][i]]['name'], distances[0][i]))


# Checker - Obama top 10 weighted words taken from the Obama/Schrillo common word set
obama_words_tf = top_words_tf('Barack Obama')
schrillo_words_tf = top_words_tf('Phil Schiliro')
common_rows_tf = get_common_rows(obama_words_tf, schrillo_words_tf, 'Obama', 'Schrillo')
for i, j in enumerate(list(common_rows_tf.sort_values('Obama',ascending = False)['words'][:10])):
    print('{0:<4}{1}'.format(i, j))


# Number of wiki articles containing Obama's top 5 words by TF-IDF weighting
obama_top_5_tf = list(common_rows_tf.sort_values('Obama', ascending = False)['words'][:5])
wiki['has_top_words_tf'] = wiki['words'].apply(lambda x: set(obama_top_5_tf).issubset(x))
print('{0:<6}{1}'.format(sum(wiki['has_top_words_tf']),
                              'articles contain the top 5 weighted words in Obama\'s article'))


# Computing Obama-Biden pairwise euclidean distance using TFIDF features
bo_vec_tf = tf_idf.getrow(wiki[wiki['name'] == 'Barack Obama'].index[0])
jb_vec_tf = tf_idf.getrow(wiki[wiki['name'] == 'Joe Biden'].index[0])
print('{0}{1:>10.5f} '.format('bo-jb:', euc_dist(bo_vec_tf, jb_vec_tf)[0][0]))











