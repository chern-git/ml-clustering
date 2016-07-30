""" clustering and retrieval week 2 - programming assignment 2 """

import numpy as np                                             # dense matrices
import pandas as pd
from itertools import combinations

from scipy.sparse import csr_matrix                            # sparse matrices
from scipy.sparse.linalg import norm                           # norms of sparse matrices
from sklearn.metrics.pairwise import pairwise_distances        # pairwise distances
import matplotlib.pyplot as plt                                # plotting
from copy import copy                                          # deep copies


# %matplotlib inline


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


def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']

    return csr_matrix((data, indices, indptr), shape)


def generate_random_vectors(num_vector, dim):
    return np.random.randn(dim, num_vector)


def train_lsh(data, num_vector=16, seed=None):
    dim = data.shape[1]
    if seed is not None:
        np.random.seed(seed)
    random_vectors = generate_random_vectors(num_vector, dim)
    powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)
    table = {}

    # Partition data points into bins
    bin_index_bits = (data.dot(random_vectors) >= 0)

    # Encode bin index bits into integers
    bin_indices = bin_index_bits.dot(powers_of_two)

    # Update `table` so that `table[i]` is the list of document ids with bin index equal to i.
    for data_index, bin_index in enumerate(bin_indices):
        if bin_index not in table:
            # If no list yet exists for this bin, assign the bin an empty list.
            table[bin_index] = []

        # Fetch the list of document ids associated with the bin and add the document id to the end.
        temp_array = (np.array(data[data_index,:].dot(random_vectors)) >= 0)
        if temp_array.dot(powers_of_two) == bin_index:
            table[bin_index].append(data_index)

    model = {'data': data,
             'bin_index_bits': bin_index_bits,
             'bin_indices': bin_indices,
             'table': table,
             'random_vectors': random_vectors,
             'num_vector': num_vector}

    return model


def cosine_distance(x, y):
    xy = x.dot(y.T)
    dist = xy/(norm(x)*norm(y))
    return 1-dist[0,0]


def search_nearby_bins(query_bin_bits, table, search_radius=2, initial_candidates=set()):
    """
    For a given query vector and trained LSH model, return all candidate neighbors for
    the query among all bins within the given search radius.

    Example usage
    -------------
    >>> model = train_lsh(corpus, num_vector=16, seed=143)
    >>> q = model['bin_index_bits'][0]  # vector for the first document
    >>> candidates = search_nearby_bins(q, model['table'])
    """
    num_vector = len(query_bin_bits)
    powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)

    # Allow the user to provide an initial set of candidates.
    candidate_set = copy(initial_candidates)

    for different_bits in combinations(range(num_vector), search_radius):
        # Flip the bits (n_1,n_2,...,n_r) of the query bin to produce a new bit vector.
        ## Hint: you can iterate over a tuple like a list
        alternate_bits = copy(query_bin_bits)

        for i in different_bits:
            alternate_bits[i] = False if query_bin_bits[i] else True

        # Convert the new bit vector to an integer index
        nearby_bin = alternate_bits.dot(powers_of_two)

        # Fetch the list of documents belonging to the bin indexed by the new bit vector.
        # Then add those documents to candidate_set
        # Make sure that the bin exists in the table!
        # Hint: update() method for sets lets you add an entire list to the set
        if nearby_bin in table:
            candidate_set.update(model['table'][nearby_bin])
    return candidate_set


def query(vec, model, k, max_search_radius):
    data = model['data']
    table = model['table']
    random_vectors = model['random_vectors']
    num_vector = random_vectors.shape[1]

    # Compute bin index for the query vector, in bit representation.
    bin_index_bits = (vec.dot(random_vectors) >= 0)[0]

    # Search nearby bins and collect candidates
    candidate_set = set()
    for search_radius in range(max_search_radius + 1):
        candidate_set = search_nearby_bins(bin_index_bits, table, search_radius,
                                           initial_candidates=candidate_set)

    # # Sort candidates by their true distances from the query
    df = pd.DataFrame()
    df['candidates'] = list(candidate_set)
    nn = corpus[df['candidates']]
    df['names'] = df['candidates'].apply(lambda x: wiki['name'][x])
    df['distance'] = df['candidates'].apply(lambda x: pairwise_distances(corpus[x], vec, metric='cosine')[0][0])
    return df.sort_values('distance')[:k], len(candidate_set)


# Loading work files
corpus = load_sparse_csr('data\people_wiki_tf_idf.npz')
map_index_to_word = pd.read_json("data\people_wiki_map_index_to_word.json", typ = "Series")
wiki = pd.read_csv('data\people_wiki.csv')
word_count = load_sparse_csr('data\people_wiki_word_count.npz')
wiki['word_count'] = unpack_dict(word_count, map_index_to_word)


# Demo code to create Bins
def create_demo_bins():
    np.random.seed(0)
    random_vectors = generate_random_vectors(num_vector=16, dim=547979)
    doc = corpus[0, :]  # vector of tf-idf values for document 0
    print(doc.dot(random_vectors[:, 0]) >= 0)  # True if positive sign; False if negative sign
    print(doc.dot(random_vectors[:, 1]) >= 0)  # True if positive sign; False if negative sign
    print(doc.dot(random_vectors) >= 0)  # should return an array of 16 True/False bits
    print(np.array(doc.dot(random_vectors) >= 0, dtype=int))  # display index bits in 0/1's
    print(corpus[0:2].dot(random_vectors) >= 0)  # compute bit indices of first two documents
    print(corpus.dot(random_vectors) >= 0)  # compute bit indices of ALL documents

    index_bits = (doc.dot(random_vectors) >= 0)
    powers_of_two = (1 << np.arange(15, -1, -1))
    print(index_bits)
    print(powers_of_two)           # [32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    print(index_bits.dot(powers_of_two))


# Check point for train_lsh function
model = train_lsh(corpus, num_vector=16, seed=143)
table = model['table']
if   0 in table and table[0]   == [39583] and \
   143 in table and table[143] == [19693, 28277, 29776, 30399]:
    print('Passed!')
else:
    print('Check your code.')


# Obama's index in wiki
obama_wiki_idx = wiki[wiki['name'] == 'Barack Obama'].index[0]
biden_wiki_idx = wiki[wiki['name'] == 'Joe Biden'].index[0]
print('{0} {1}'.format('Obama\'s index in wiki:', obama_wiki_idx))


# Obama and Biden's bin index
for k, vals in table.items():
    if obama_wiki_idx in vals:
        obama_bin_idx = k
        print('{0} {1}'.format('Obama\'s bin number:', k))
    if biden_wiki_idx in vals:
        biden_bin_idx = k
        print('{0} {1}'.format('Bidens\'s bin number:', k))


# Obama and Biden's difference in bin bit representation
print(sum(model['bin_index_bits'][obama_wiki_idx] == model['bin_index_bits'][biden_wiki_idx]))


print(wiki[wiki['name']=='Wynn Normington Hugh-Jones'])
print(np.array(model['bin_index_bits'][22745], dtype=int)) # list of 0/1's
print(sum(model['bin_index_bits'][35817] == model['bin_index_bits'][22745]))


# 4 other documents in same bin as Obama:
doc_ids = list(model['table'][model['bin_indices'][35817]])
doc_ids.remove(35817)  # display documents other than Obama
print(wiki.ix[doc_ids]['name'])


# Measuring the 4 similar persons using cos similarity
obama_tf_idf = corpus[35817,:]
biden_tf_idf = corpus[24478,:]
print('================= Cosine distance from Barack Obama')
print('Barack Obama - {0:24s}: {1:f}'.format('Joe Biden',
                                             cosine_distance(obama_tf_idf, biden_tf_idf)))
for doc_id in doc_ids:
    doc_tf_idf = corpus[doc_id,:]
    print('Barack Obama - {0:24s}: {1:f}'.format(wiki.ix[doc_id]['name'],
                                                 cosine_distance(obama_tf_idf, doc_tf_idf)))


# Checker for LSH model
def checker_lsh_query():
    print('Checking radius = 0 case:')
    obama_bin_index = model['bin_index_bits'][35817] # bin index of Barack Obama
    candidate_set = search_nearby_bins(obama_bin_index, model['table'], search_radius=0)
    if candidate_set == set([35817, 21426, 53937, 39426, 50261]):
        print('Passed test')
    else:
        print('Check your code')

    print('Checking radius = 1 case:')
    candidate_set = search_nearby_bins(obama_bin_index, model['table'], search_radius=1,
                                       initial_candidates=candidate_set)
    if candidate_set == set([39426, 38155, 38412, 28444, 9757, 41631, 39207, 59050, 47773, 53937, 21426, 34547,
                             23229, 55615, 39877, 27404, 33996, 21715, 50261, 21975, 33243, 58723, 35817, 45676,
                             19699, 2804, 20347]):
        print('Passed test')
    else:
        print('Check your code')
checker_lsh_query()


# Checker for LSH query
query(corpus[35817,:], model, k=10, max_search_radius=3)


num_candidates_history = []
query_time_history = []
max_distance_from_query_history = []
min_distance_from_query_history = []
average_distance_from_query_history = []

for max_search_radius in range(17):
    start = time.time()
    # Perform LSH query using Barack Obama, with max_search_radius
    result, num_candidates = query(corpus[35817, :], model, k=10,
                                   max_search_radius=max_search_radius)
    end = time.time()
    query_time = end - start  # Measure time

    print('Radius:', max_search_radius)
    # Display 10 nearest neighbors, along with document ID and name
    # print(result.join(wiki[['id', 'name']], on='id').sort('distance'))
    print(result)

    # Collect statistics on 10 nearest neighbors
    average_distance_from_query = result['distance'][1:].mean()
    max_distance_from_query = result['distance'][1:].max()
    min_distance_from_query = result['distance'][1:].min()

    num_candidates_history.append(num_candidates)
    query_time_history.append(query_time)
    average_distance_from_query_history.append(average_distance_from_query)
    max_distance_from_query_history.append(max_distance_from_query)
    min_distance_from_query_history.append(min_distance_from_query)
