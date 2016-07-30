""" clustering and retrieval week 4 - programming assignment 5 """

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags
from scipy.stats import multivariate_normal
from copy import deepcopy
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans


def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']

    return csr_matrix((data, indices, indptr), shape)


map_index_to_word = pd.read_json("data\\4_map_index_to_word.json", typ = "Series")
map_index_to_word2 = pd.read_json("data\\4_map_index_to_word.json")
wiki = pd.read_csv('data\people_wiki.csv')
tf_idf = load_sparse_csr('data\\4_tf_idf.npz')
tf_idf = normalize(tf_idf)


def diag(array):
    n = len(array)
    return spdiags(array, 0, n, n)


def logpdf_diagonal_gaussian(x, mean, cov):
    '''
    Compute logpdf of a multivariate Gaussian distribution with diagonal covariance at a given point x.
    A multivariate Gaussian distribution with a diagonal covariance is equivalent
    to a collection of independent Gaussian random variables.

    x should be a sparse matrix. The logpdf will be computed for each row of x.
    mean and cov should be given as 1D numpy arrays
    mean[i] : mean of i-th variable
    cov[i] : variance of i-th variable'''

    n = x.shape[0]
    dim = x.shape[1]
    assert(dim == len(mean) and dim == len(cov))

    # multiply each i-th column of x by (1/(2*sigma_i)), where sigma_i is sqrt of variance of i-th variable.
    scaled_x = x.dot( diag(1./(2*np.sqrt(cov))) )
    # multiply each i-th entry of mean by (1/(2*sigma_i))
    scaled_mean = mean/(2*np.sqrt(cov))

    # sum of pairwise squared Eulidean distances gives SUM[(x_i - mean_i)^2/(2*sigma_i^2)]
    return -np.sum(np.log(np.sqrt(2*np.pi*cov))) - pairwise_distances(scaled_x, [scaled_mean], 'euclidean').flatten()**2


def log_sum_exp(x, axis):
    '''Compute the log of a sum of exponentials'''
    x_max = np.max(x, axis=axis)
    if axis == 1:
        return x_max + np.log( np.sum(np.exp(x-x_max[:,np.newaxis]), axis=1) )
    else:
        return x_max + np.log( np.sum(np.exp(x-x_max), axis=0) )


def EM_for_high_dimension(data, means, covs, weights, cov_smoothing=1e-5, maxiter=int(1e3), thresh=1e-4, verbose=False):
    # cov_smoothing: specifies the default variance assigned to absent features in a cluster.
    #                If we were to assign zero variances to absent features, we would be overconfient,
    #                as we hastily conclude that those featurese would NEVER appear in the cluster.
    #                We'd like to leave a little bit of possibility for absent features to show up later.
    n = data.shape[0]
    dim = data.shape[1]
    mu = deepcopy(means)
    Sigma = deepcopy(covs)
    K = len(mu)
    weights = np.array(weights)

    ll = None
    ll_trace = []

    for i in range(maxiter):
        # E-step: compute responsibilities
        logresp = np.zeros((n,K))
        for k in range(K):
            logresp[:,k] = np.log(weights[k]) + logpdf_diagonal_gaussian(data, mu[k], Sigma[k])
        ll_new = np.sum(log_sum_exp(logresp, axis=1))
        if verbose:
            print(ll_new)
        logresp -= np.vstack(log_sum_exp(logresp, axis=1))
        resp = np.exp(logresp)
        counts = np.sum(resp, axis=0)

        # M-step: update weights, means, covariances
        weights = counts / np.sum(counts)
        for k in range(K):
            mu[k] = (diag(resp[:,k]).dot(data)).sum(axis=0)/counts[k]
            mu[k] = mu[k].A1

            Sigma[k] = diag(resp[:,k]).dot( data.power(2)-2*data.dot(diag(mu[k])) ).sum(axis=0) \
                       + (mu[k]**2)*counts[k]
            Sigma[k] = Sigma[k].A1 / counts[k] + cov_smoothing*np.ones(dim)

        # check for convergence in log-likelihood
        ll_trace.append(ll_new)
        if ll is not None and (ll_new-ll) < thresh and ll_new > -np.inf:
            ll = ll_new
            break
        else:
            ll = ll_new

    out = {'weights':weights,'means':mu,'covs':Sigma,'loglik':ll_trace,'resp':resp}

    return out


np.random.seed(5)
num_clusters = 25

# Use scikit-learn's k-means to simplify workflow
kmeans_model = KMeans(n_clusters=num_clusters, n_init=5, max_iter=400, random_state=1, n_jobs=-1)
kmeans_model.fit(tf_idf)
centroids, cluster_assignment = kmeans_model.cluster_centers_, kmeans_model.labels_
means = [centroid for centroid in centroids]


# Initial cluster weights
num_docs = tf_idf.shape[0]
weights = []
for i in range(num_clusters):
    # Compute the number of data points assigned to cluster i:
    num_assigned = sum(cluster_assignment == i)
    w = float(num_assigned) / num_docs
    weights.append(w)

# Initializing covariances
covs = []
for i in range(num_clusters):
    member_rows = tf_idf[cluster_assignment==i]
    cov = (member_rows.power(2) - 2*member_rows.dot(diag(means[i]))).sum(axis=0).A1 / member_rows.shape[0] \
          + means[i]**2
    cov[cov < 1e-8] = 1e-8
    covs.append(cov)

# Running EM
out = EM_for_high_dimension(tf_idf, means, covs, weights, cov_smoothing=1e-10)
print(out['loglik']) # print history of log-likelihood over time
# [3855847476.7012835, 4844053202.46348, 4844053202.46348]

# Using df because map_index_to_words won't read correctly with read.json
df = pd.DataFrame()
df['idx'] = map_index_to_word.values
df['word'] = map_index_to_word.index

def visualize_EM_clusters(means, covs, df):
    print('')
    print('==========================================================')

    num_clusters = len(means)
    for c in range(num_clusters):
        print('Cluster {0:d}: Largest mean parameters in cluster '.format(c))
        print('\n{0: <12}{1: <12}{2: <12}{3: <12}'.format('Word', 'Mean', 'Variance','Index'))

        sorted_word_ids = means[c].argsort()[::-1]
        for i in sorted_word_ids[:5]:
            # print(df[df['idx'] == i])
            print(
            '{0: <12}{1:<10.2e}{2:10.2e}{3:>10}'.format(df[df['idx'] == i]['word'].values[0],
                                                 means[c][i],
                                                 covs[c][i],
                                                 i))
        print('\n=====================================================')


visualize_EM_clusters(out['means'], out['covs'], df)


# Random assignment
np.random.seed(5)
num_clusters = len(means)
num_docs, num_words = tf_idf.shape
random_means = []
random_covs = []
random_weights = []

for k in range(num_clusters):
    # Create a numpy array of length num_words with random normally distributed values.
    # Use the standard univariate normal distribution (mean 0, variance 1).
    mean = np.random.normal(0, 1, num_words)

    # Create a numpy array of length num_words with random values uniformly distributed between 1 and 5.
    cov = np.random.uniform(1,5, num_words)

    # Initially give each cluster equal weight.
    weight = 1/len(means)

    random_means.append(mean)
    random_covs.append(cov)
    random_weights.append(weight)


# Running EM with random initialization
out_random_init = EM_for_high_dimension(tf_idf, random_means, random_covs, random_weights, cov_smoothing=1e-5)

for i in out_random_init['loglik']:
    print('{0:.0f}'.format(i))

visualize_EM_clusters(out_random_init['means'], out_random_init['covs'], df)

