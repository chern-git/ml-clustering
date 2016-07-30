""" clustering and retrieval week 6 - programming assignment 6 """

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans                # we'll be using scikit-learn's KMeans for this assignment
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize


def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']

    return csr_matrix((data, indices, indptr), shape)


map_index_to_word = pd.read_json("data\people_wiki_map_index_to_word.json", typ = "Series")
wiki = pd.read_csv('data\people_wiki.csv')
tf_idf = load_sparse_csr('data\people_wiki_tf_idf.npz')
tf_idf = normalize(tf_idf)

# Using df because map_index_to_words won't read correctly with read.json
df = pd.DataFrame()
df['idx'] = map_index_to_word.values
df['word'] = map_index_to_word.index


def bipartition(cluster, maxiter=400, num_runs=4, seed=None):
    '''cluster: should be a dictionary containing the following keys
                * dataframe: original dataframe
                * matrix:    same data, in matrix format
                * centroid:  centroid for this particular cluster'''

    data_matrix = cluster['matrix']
    dataframe = cluster['dataframe']

    # Run k-means on the data matrix with k=2. We use scikit-learn here to simplify workflow.
    kmeans_model = KMeans(n_clusters=2, max_iter=maxiter, n_init=num_runs, random_state=seed, n_jobs=-1)
    kmeans_model.fit(data_matrix)
    centroids, cluster_assignment = kmeans_model.cluster_centers_, kmeans_model.labels_

    # Divide the data matrix into two parts using the cluster assignments.
    data_matrix_left_child, data_matrix_right_child = data_matrix[cluster_assignment == 0], \
                                                      data_matrix[cluster_assignment == 1]

    # Divide the dataframe into two parts, again using the cluster assignments.
    cluster_assignment_sa = np.array(cluster_assignment)  # minor format conversion
    dataframe_left_child, dataframe_right_child = dataframe[cluster_assignment_sa == 0], \
                                                  dataframe[cluster_assignment_sa == 1]

    # Package relevant variables for the child clusters
    cluster_left_child = {'matrix': data_matrix_left_child,
                          'dataframe': dataframe_left_child,
                          'centroid': centroids[0]}
    cluster_right_child = {'matrix': data_matrix_right_child,
                           'dataframe': dataframe_right_child,
                           'centroid': centroids[1]}

    return (cluster_left_child, cluster_right_child)


wiki_data = {'matrix': tf_idf, 'dataframe': wiki} # no 'centroid' for the root cluster
left_child, right_child = bipartition(wiki_data, maxiter=100, num_runs=8, seed=1)

athletes = left_child
non_athletes = right_child


# Visualize the partition
def display_single_tf_idf_cluster(cluster, df_map):
    '''map_index_to_word: SFrame specifying the mapping betweeen words and column indices'''

    wiki_subset = cluster['dataframe']
    tf_idf_subset = cluster['matrix']
    centroid = cluster['centroid']

    # Print top 5 words with largest TF-IDF weights in the cluster
    idx = centroid.argsort()[::-1]
    for i in range(5):
        print('{0:s}:{1:.3f}'.format(df_map[df_map['idx'] == idx[i]]['word'].values[0], centroid[idx[i]]))
    print('')

    # Compute distances from the centroid to all data points in the cluster.
    distances = pairwise_distances(tf_idf_subset, [centroid], metric='euclidean').flatten()
    # compute nearest neighbors of the centroid within the cluster.
    nearest_neighbors = distances.argsort()
    # For 8 nearest neighbors, print the title as well as first 180 characters of text.
    # Wrap the text at 80-character mark.
    for i in range(8):
        text = ' '.join(wiki_subset.iloc[nearest_neighbors[i]]['text'].split(None, 25)[0:25])
        print('* {0:50s} {1:.5f}\n  {2:s}\n  {3:s}'.format(wiki_subset.iloc[nearest_neighbors[i]]['name'],
              distances[nearest_neighbors[i]], text[:90], text[90:180] if len(text) > 90 else ''))
        print(text)
    print()


display_single_tf_idf_cluster(left_child, df)
display_single_tf_idf_cluster(right_child, df)


# Further partitioning the left (athletes) partition
athletes_L, athletes_R = bipartition(athletes, maxiter=100, num_runs=8, seed=1)
display_single_tf_idf_cluster(athletes_L, df)   # baseball
display_single_tf_idf_cluster(athletes_R, df)   # football + icehockey

athletes_RL, athletes_RR = bipartition(athletes_R, maxiter=100, num_runs=8, seed=1)
display_single_tf_idf_cluster(athletes_RL, df)  # football
display_single_tf_idf_cluster(athletes_RR, df)  # track & field, golf


# Partitioning the right (non_athletes) partition
non_athletes_L, non_athletes_R = bipartition(non_athletes, maxiter=100, num_runs=8, seed=1)
display_single_tf_idf_cluster(non_athletes_L, df)   # pol + govt officials + scholars
display_single_tf_idf_cluster(non_athletes_R, df)   # musicians + artists + actors

non_athletes_LL, non_athletes_LR = bipartition(non_athletes_L, maxiter=100, num_runs=8, seed=1)
display_single_tf_idf_cluster(non_athletes_LL, df)   # pol + gov officials
display_single_tf_idf_cluster(non_athletes_LR, df)   # scholars

non_athletes_RL, non_athletes_RR = bipartition(non_athletes_R, maxiter=100, num_runs=8, seed=1)
display_single_tf_idf_cluster(non_athletes_RL, df)   # actors + writers
display_single_tf_idf_cluster(non_athletes_RR, df)   # musicians