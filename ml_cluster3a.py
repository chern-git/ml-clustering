""" clustering and retrieval week 3 - programming assignment 3 """

import matplotlib.pyplot as plt                                # plotting
import numpy as np                                             # dense matrices
import pandas as pd

from scipy.sparse import csr_matrix                            # sparse matrices
from sklearn.preprocessing import normalize                    # normalizing vectors
from sklearn.metrics import pairwise_distances                 # pairwise distances
import sys
import os


def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']

    return csr_matrix((data, indices, indptr), shape)


def get_initial_centroids(data, k, seed=None):
    '''Randomly choose k data points as initial centroids'''
    if seed is not None:  # useful for obtaining consistent results
        np.random.seed(seed)
    n = data.shape[0]  # number of data points

    # Pick K indices from range [0, N).
    rand_indices = np.random.randint(0, n, k)

    # Keep centroids as dense format, as many entries will be nonzero due to averaging.
    # As long as at least one document in a cluster contains a word,
    # it will carry a nonzero weight in the TF-IDF vector of the centroid.
    centroids = data[rand_indices, :].toarray()

    return centroids


def assign_clusters(data, centroids):
    # Compute distances between each data point and the set of centroids:
    # Fill in the blank (RHS only)
    distances_from_centroids = pairwise_distances(data, centroids)

    # Compute cluster assignments for each data point:
    # Fill in the blank (RHS only)
    # cluster_assignment = np.apply_along_axis(np.argmin, axis=1, arr=distances_from_centroids)
    cluster_assignment = [np.argmin(r) for r in distances_from_centroids]

    return cluster_assignment


def revise_centroids(data, k, cluster_assignment):
    new_centroids = []
    for i in range(k):
        # Select all data points that belong to cluster i. Fill in the blank (RHS only)
        member_data_points = data[np.array(cluster_assignment) == i]
        # Compute the mean of the data points. Fill in the blank (RHS only)
        centroid = data[np.array(cluster_assignment) == i].mean(axis = 0)
        # centroid = np.apply_along_axis(np.mean, axis=0, arr=member_data_points)

        # Convert numpy.matrix type to numpy.ndarray type
        centroid = centroid.A1
        new_centroids.append(centroid)
    new_centroids = np.array(new_centroids)
    return new_centroids


def compute_heterogeneity(data, k, centroids, cluster_assignment):
    heterogeneity = 0.0
    for i in range(k):

        # Select all data points that belong to cluster i. Fill in the blank (RHS only)
        member_data_points = data[np.array(cluster_assignment) == i]
        # print('data shpe ', member_data_points.shape[0])

        if member_data_points.shape[0] > 0:  # check if i-th cluster is non-empty
            # Compute distances from centroid to data points (RHS only)
            distances = pairwise_distances(member_data_points, [centroids[i]], metric='euclidean')
            squared_distances = distances ** 2
            heterogeneity += np.sum(squared_distances)

    return heterogeneity


def kmeans(data, k, initial_centroids, maxiter, record_heterogeneity=None, verbose=False):
    centroids = initial_centroids[:]
    prev_cluster_assignment = None

    for itr in range(maxiter):
        if verbose:
            print(itr)

        # 1. Make cluster assignments using nearest centroids
        # YOUR CODE HERE
        cluster_assignment = assign_clusters(data, centroids)

        # 2. Compute a new centroid for each of the k clusters, averaging all data points assigned to that cluster.
        # YOUR CODE HERE
        centroids = revise_centroids(data, k, cluster_assignment)

        # Check for convergence: if none of the assignments changed, stop
        if prev_cluster_assignment is not None and \
                (np.array(prev_cluster_assignment) == np.array(cluster_assignment)).all():
            break

        # Print number of new assignments
        if prev_cluster_assignment is not None:
            num_changed = np.sum(np.array(prev_cluster_assignment) != np.array(cluster_assignment))
            if verbose:
                print('    {0:5d} elements changed their cluster assignment.'.format(num_changed))

                # Record heterogeneity convergence metric
        if record_heterogeneity is not None:
            score = compute_heterogeneity(data, k, centroids, cluster_assignment)
            record_heterogeneity.append(score)

        prev_cluster_assignment = cluster_assignment[:]

    return centroids, cluster_assignment


def plot_heterogeneity(heterogeneity, k, outfile = None):
    plt.figure(figsize=(7,4))
    plt.plot(heterogeneity,linewidth=4)
    plt.xlabel('# Iterations')
    plt.ylabel('Heterogeneity')
    plt.title('Heterogeneity of clustering over time, K={0:d}'.format(k))
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    if outfile:
        plt.savefig('output/' + outfile + '.png')


def smart_initialize(data, k, seed=None):
    '''Use k-means++ to initialize a good set of centroids'''
    if seed is not None:  # useful for obtaining consistent results
        np.random.seed(seed)
    centroids = np.zeros((k, data.shape[1]))

    # Randomly choose the first centroid.
    # Since we have no prior knowledge, choose uniformly at random
    idx = np.random.randint(data.shape[0])
    centroids[0] = data[idx, :].toarray()
    # Compute distances from the first centroid chosen to all the other data points
    distances = pairwise_distances(data, centroids[0:1], metric='euclidean').flatten()

    for i in range(1, k):
        # Choose the next centroid randomly, so that the probability for each data point to be chosen
        # is directly proportional to its squared distance from the nearest centroid.
        # Roughtly speaking, a new centroid should be as far as from ohter centroids as possible.
        idx = np.random.choice(data.shape[0], 1, p=distances / sum(distances))
        centroids[i] = data[idx, :].toarray()
        # Now compute distances from the centroids to all data points
        distances = np.min(pairwise_distances(data, centroids[0:i + 1], metric='euclidean'), axis=1)

    return centroids


def kmeans_multiple_runs(data, k, maxiter, num_runs, seed_list=None, verbose=False):
    heterogeneity = {}

    min_heterogeneity_achieved = float('inf')
    best_seed = None
    final_centroids = None
    final_cluster_assignment = None

    for i in xrange(num_runs):

        # Use UTC time if no seeds are provided
        if seed_list is not None:
            seed = seed_list[i]
            np.random.seed(seed)
        else:
            seed = int(time.time())
            np.random.seed(seed)

        # Use k-means++ initialization
        # YOUR CODE HERE
        initial_centroids = ...

        # Run k-means
        # YOUR CODE HERE
        centroids, cluster_assignment = ...

        # To save time, compute heterogeneity only once in the end
        # YOUR CODE HERE
        heterogeneity[seed] = ...

        if verbose:
            print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity[seed]))
            sys.stdout.flush()

        # if current measurement of heterogeneity is lower than previously seen,
        # update the minimum record of heterogeneity.
        if heterogeneity[seed] < min_heterogeneity_achieved:
            min_heterogeneity_achieved = heterogeneity[seed]
            best_seed = seed
            final_centroids = centroids
            final_cluster_assignment = cluster_assignment

    # Return the centroids and cluster assignments that minimize heterogeneity.
    return final_centroids, final_cluster_assignment


# Loading and pre-processing
map_index_to_word = pd.read_json("data\people_wiki_map_index_to_word.json", typ = "Series")
wiki = pd.read_csv('data\people_wiki.csv')
tf_idf = load_sparse_csr('data\people_wiki_tf_idf.npz')
tf_idf = normalize(tf_idf)


# Bunch of checkers
def bunch_of_checkers():
    query = tf_idf[0:3, :]
    distances = pairwise_distances(tf_idf, query)
    closest_cluster = np.apply_along_axis(np.argmin, axis=1, arr=distances)
    cluster_assignment = closest_cluster

    tests = 0
    reference = [list(row).index(min(row)) for row in distances]
    print('Checking if we correctly assigned closest')
    if np.allclose(closest_cluster, reference):
        print('Pass')
        tests +=  1
    else:
        print('Check your code again')


    print()
    print('Checking if we correctly assigned closest cluster based on distances')
    if len(cluster_assignment)==59071 and \
       np.array_equal(np.bincount(cluster_assignment), np.array([23061, 10086, 25924])):
        print('Pass') # count number of data points for each cluster
        tests +=  1
    else:
        print('Check your code again.')

    print()
    print('Checking correctness of assign_cluster()')
    if np.allclose(assign_clusters(tf_idf[0:100:10], tf_idf[0:8:2]), np.array([0, 1, 1, 0, 0, 2, 0, 2, 2, 1])):
        print('Pass')
        tests +=  1
    else:
        print('Check your code again.')

    print()
    print('Checking correctness of revise_centroids()')
    result = revise_centroids(tf_idf[0:100:10], 3, np.array([0, 1, 1, 0, 0, 2, 0, 2, 2, 1]))
    if np.allclose(result[0], np.mean(tf_idf[[0,30,40,60]].toarray(), axis=0)) and \
       np.allclose(result[1], np.mean(tf_idf[[10,20,90]].toarray(), axis=0))   and \
       np.allclose(result[2], np.mean(tf_idf[[50,70,80]].toarray(), axis=0)):
        print('Pass')
        tests +=  1
    else:
        print('Check your code')

    print()
    print(tests,'/4 tests passed', sep = "")

# bunch_of_checkers()


# Sample code - Revising clusters
def revising_clusters_demo():
    data = np.array([[1., 2., 0.],
                     [0., 0., 0.],
                     [2., 2., 0.]])
    centroids = np.array([[0.5, 0.5, 0.],
                          [0., -0.5, 0.]])
    cluster_assignment = assign_clusters(data, centroids)
    print(cluster_assignment)  # prints [0 1 0]

    # print(data[cluster_assignment == 0])
    print(data[np.array(cluster_assignment) == 0])
    print(data[np.array(cluster_assignment) == 1])
    print(data[np.array(cluster_assignment) == 0].mean(axis=0))

# revising_clusters_demo()


# Checker for k-means, using k = 3
def plot_het_k3():
    k = 3
    heterogeneity = []
    initial_centroids = get_initial_centroids(tf_idf, k, seed=0)
    centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                           record_heterogeneity=heterogeneity, verbose=True)
    plot_heterogeneity(heterogeneity, k, 'output\het_k3')

plot_het_k3()

# Clustering and calculating heterogeneity using km for k = 10
def calc_km_het_k10():
    k = 10
    heterogeneity = {}
    import time
    start = time.time()
    for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
        initial_centroids = get_initial_centroids(tf_idf, k, seed)
        centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                               record_heterogeneity=None, verbose=False)
        # To save time, compute heterogeneity only once in the end
        heterogeneity[seed] = compute_heterogeneity(tf_idf, k, centroids, cluster_assignment)
        print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity[seed]))
        sys.stdout.flush()
    end = time.time()
    print(end-start)


# Clustering and calculating heterogeneity using km++ for k = 10
def calc_kmpp_het_k10():
    k = 10
    heterogeneity_smart = {}
    start = time.time()
    for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
        initial_centroids = smart_initialize(tf_idf, k, seed)
        centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                               record_heterogeneity=None, verbose=False)
        # To save time, compute heterogeneity only once in the end
        heterogeneity_smart[seed] = compute_heterogeneity(tf_idf, k, centroids, cluster_assignment)
        print('seed={0:06d}, heterogeneity={1:.5f}'.format(seed, heterogeneity_smart[seed]))
        sys.stdout.flush()
    end = time.time()
    print(end-start)


# Plotting the results, this time using km++ to initialize the algorithm
def plot_het_km_vs_kmpp():
    plt.figure(figsize=(8,5))
    het_values = [i for i in heterogeneity.values()]
    het_smart_values = [i for i in heterogeneity_smart.values()]
    plt.boxplot([het_values, het_smart_values], vert=False)
    plt.yticks([1, 2], ['k-means', 'k-means++'])
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()


# How to choose K
def plot_k_vs_heterogeneity(k_values, heterogeneity_values, outfile = None):
    plt.figure(figsize=(7, 4))
    plt.plot(k_values, heterogeneity_values, linewidth=4)
    plt.xlabel('K')
    plt.ylabel('Heterogeneity')
    plt.title('K vs. Heterogeneity')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    if outfile:
        plt.savefig('output/' + outfile + '.png')


filename = 'data/kmeans-arrays.npz'
heterogeneity_values = []
k_list = [2, 10, 25, 50, 100]

if os.path.exists(filename):
    arrays = np.load(filename)
    centroids = {}
    cluster_assignment = {}
    for k in k_list:
        print(k)
        sys.stdout.flush()
        centroids[k] = arrays['centroids_{0:d}'.format(k)]
        cluster_assignment[k] = arrays['cluster_assignment_{0:d}'.format(k)]
        score = compute_heterogeneity(tf_idf, k, centroids[k], cluster_assignment[k])
        heterogeneity_values.append(score)

    plot_k_vs_heterogeneity(k_list, heterogeneity_values, 'k_vs_het')
else:
    print('File not found. Skipping.')


# Visualize clusters of docs
def viz_doc_clusters(wiki, tf_idf, centroids, cluster_assignment, k, map_index_to_word, disp_content=True):
    print('==========================================================')
    for c in range(k):
        print('Cluster {0:d}    '.format(c)),
        idx = centroids[c].argsort()[::-1]
        for i in range(5):  # Print each word along with the TF-IDF weight
            print('{0:s}:{1:.3f}'.format(map_index_to_word.index[idx[i]], centroids[c, idx[i]]))
        print('')

        if disp_content:
            distances = pairwise_distances(tf_idf, [centroids[c]], metric='euclidean').flatten()
            distances[np.array(cluster_assignment) != c] = float('inf')  # remove non-members from consideration
            nearest_neighbors = distances.argsort()
            # For 8 nearest neighbors, print the title as well as first 180 characters of text.
            for i in range(8):
                text = ' '.join(str(wiki.ix[nearest_neighbors[i]]['text']).split(None,25))
                print('\n* {0:25s} {1:.5f}\n  {2:s}\n  {3:s}'.format(wiki.ix[nearest_neighbors[i]]['name'],
                                                                     distances[nearest_neighbors[i]], text[:90],
                                                                     text[90:180] if len(text) > 90 else ''))
                print(text)
        print('==========================================================')

viz_doc_clusters(wiki, tf_idf, centroids[2], cluster_assignment[2], 2, map_index_to_word)


k = 10
viz_doc_clusters(wiki, tf_idf, centroids[k], cluster_assignment[k], k, map_index_to_word)
np.bincount(cluster_assignment[10])

k = 100
viz_doc_clusters(wiki, tf_idf, centroids[k], cluster_assignment[k], k, map_index_to_word, disp_content=False)
sum(np.array(np.bincount(cluster_assignment[100]) < 236))