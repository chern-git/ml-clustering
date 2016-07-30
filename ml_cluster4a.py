""" clustering and retrieval week 4 - programming assignment 4 """

import os
import copy                                                    # deep copies
import numpy as np                                             # dense matrices
import pandas as pd
import matplotlib.pyplot as plt                                # plotting
import matplotlib.mlab as mlab


from scipy.stats import multivariate_normal                    # multivariate Gaussian distribution
from scipy import misc
from PIL import Image


def generate_MoG_data(num_data, means, covariances, weights):
    """ Creates a list of data points """
    num_clusters = len(weights)
    data = []
    for i in range(num_data):
        #  Use np.random.choice and weights to pick a cluster id greater than or equal to 0 and less than num_clusters.
        k = np.random.choice(len(weights), 1, p=weights)[0]

        # Use np.random.multivariate_normal to create data from this cluster
        x = np.random.multivariate_normal(means[k], covariances[k])

        data.append(x)
    return data


# Model parameters
init_means = [
    [5, 0], # mean of cluster 1
    [1, 1], # mean of cluster 2
    [0, 5]  # mean of cluster 3
]
init_covariances = [
    [[.5, 0.], [0, .5]], # covariance of cluster 1
    [[.92, .38], [.38, .91]], # covariance of cluster 2
    [[.5, 0.], [0, .5]]  # covariance of cluster 3
]
init_weights = [1/4., 1/2., 1/4.]  # weights of each cluster

# Generate data and plot
np.random.seed(4)
data = generate_MoG_data(100, init_means, init_covariances, init_weights)

plt.figure()
d = np.vstack(data)
plt.plot(d[:,0], d[:,1],'ko')
plt.rcParams.update({'font.size':16})
plt.tight_layout()


# Implementation of EM

def log_sum_exp(Z):
    """ Compute log(\sum_i exp(Z_i)) for some array Z."""
    return np.max(Z) + np.log(np.sum(np.exp(Z - np.max(Z))))


def loglikelihood(data, weights, means, covs):
    """ Compute the loglikelihood of the data for a Gaussian mixture model with the given parameters. """
    num_clusters = len(means)
    num_dim = len(data[0])

    ll = 0
    for d in data:

        Z = np.zeros(num_clusters)
        for k in range(num_clusters):
            # Compute (x_mu)^T * Sigma^{-1} * (x_mu)
            delta = np.array(d) - means[k]
            exponent_term = np.dot(delta.T, np.dot(np.linalg.inv(covs[k]), delta))

            # Compute loglikelihood contribution for this data point and this cluster
            Z[k] += np.log(weights[k])
            Z[k] -= 1 / 2. * (num_dim * np.log(2 * np.pi) + np.log(np.linalg.det(covs[k])) + exponent_term)

        # Increment loglikelihood contribution of this data point across all clusters
        ll += log_sum_exp(Z)

    return ll


def EM(data, init_means, init_covariances, init_weights, maxiter=1000, thresh=1e-4):
    # Make copies of initial parameters, which we will update during each iteration
    means = init_means[:]
    covariances = init_covariances[:]
    weights = init_weights[:]

    # Infer dimensions of dataset and the number of clusters
    num_data = len(data)
    num_dim = len(data[0])
    num_clusters = len(means)

    # Initialize some useful variables
    resp = np.zeros((num_data, num_clusters))
    ll = loglikelihood(data, weights, means, covariances)
    ll_trace = [ll]

    for i in range(maxiter):
        if i % 5 == 0:
            print("Iteration %s" % i)

        # E-step: compute responsibilities
        # Update resp matrix so that resp[j, k] is the responsibility of cluster k for data point j.
        # Hint: To compute likelihood of seeing data point j given cluster k, use multivariate_normal.pdf.
        for j in range(num_data):
            for k in range(num_clusters):
                # YOUR CODE HERE
                resp[j, k] = weights[k] * multivariate_normal.pdf(data[j], means[k], covariances[k])
        row_sums = resp.sum(axis=1)[:, np.newaxis]
        resp = resp / row_sums  # normalize over all possible cluster assignments

        # M-step
        # Compute the total responsibility assigned to each cluster, which will be useful when
        # implementing M-steps below. In the lectures this is called N^{soft}
        counts = np.sum(resp, axis=0)

        for k in range(num_clusters):

            # Update the weight for cluster k using the M-step update rule for the cluster weight, \hat{\pi}_k.
            # YOUR CODE HERE
            weights[k] = counts[k] / num_data

            # Update means for cluster k using the M-step update rule for the mean variables.
            # This will assign the variable means[k] to be our estimate for \hat{\mu}_k.
            weighted_sum = 0
            for j in range(num_data):
                # YOUR CODE HERE
                weighted_sum += resp[j, k] * data[j]
            # YOUR CODE HERE
            means[k] = 1/counts[k] * weighted_sum

            # Update covariances for cluster k using the M-step update rule for covariance variables.
            # This will assign the variable covariances[k] to be the estimate for \hat{Sigma}_k.
            weighted_sum = np.zeros((num_dim, num_dim))
            for j in range(num_data):
                # YOUR CODE HERE (Hint: Use np.outer on the data[j] and this cluster's mean)
                weighted_sum += resp[j, k] * np.outer((data[j] - means[k]), data[j] - means[k])
            # YOUR CODE HERE
            covariances[k] = 1/counts[k] * weighted_sum

        # Compute the loglikelihood at this iteration
        # YOUR CODE HERE
        ll_latest = loglikelihood(data, weights, means, covariances)
        ll_trace.append(ll_latest)

        # Check for convergence in log-likelihood and store
        if (ll_latest - ll) < thresh and ll_latest > -np.inf:
            break
        ll = ll_latest

    if i % 5 != 0:
        print("Iteration %s" % i)

    out = {'weights': weights, 'means': means, 'covs': covariances, 'loglik': ll_trace, 'resp': resp}

    return out


# Checker code for EM:
np.random.seed(4)

# Initialization of parameters
chosen = np.random.choice(len(data), 3, replace=False)
initial_means = [data[x] for x in chosen]
initial_covs = [np.cov(data, rowvar=0)] * 3
initial_weights = [1/3.] * 3

# Run EM and print answers to questions 1-3
results = EM(data, initial_means, initial_covs, initial_weights)
print('{0:s}{1:.4f}'.format('Weight assigned: ',results['weights'][0]))
print('{0:s}{1:.4f}'.format('Mean assigned: ',results['means'][1][0]))
print('{0:s}{1:.4f}'.format('Var assigned: ',results['covs'][2][0][0]))


# Plotting progress of parameters
def plot_contours(data, means, covs, title, outfile = None):
    plt.figure()
    plt.plot([x[0] for x in data], [y[1] for y in data],'ko') # data

    delta = 0.025
    k = len(means)
    x = np.arange(-2.0, 7.0, delta)
    y = np.arange(-2.0, 7.0, delta)
    X, Y = np.meshgrid(x, y)
    col = ['green', 'red', 'indigo']
    for i in range(k):
        mean = means[i]
        cov = covs[i]
        sigmax = np.sqrt(cov[0][0])
        sigmay = np.sqrt(cov[1][1])
        sigmaxy = cov[0][1]/(sigmax*sigmay)
        Z = mlab.bivariate_normal(X, Y, sigmax, sigmay, mean[0], mean[1], sigmaxy)
        plt.contour(X, Y, Z, colors = col[i])
        plt.title(title)
    plt.rcParams.update({'font.size':16})
    plt.tight_layout()
    if outfile:
        plt.savefig('output/' + outfile + '.png')


# Visualizing EM
# Parameters after initialization
plot_contours(data, initial_means, initial_covs, 'Initial clusters', 'contours_initial')

# Parameters after 12 iterations
results = EM(data, initial_means, initial_covs, initial_weights, maxiter=12)
plot_contours(data, results['means'], results['covs'], 'Clusters after 12 iterations', 'contours_12_iters')

# Parameters after running EM to convergence
results = EM(data, initial_means, initial_covs, initial_weights)
plot_contours(data, results['means'], results['covs'], 'Final clusters', ' contours_final')


# Plotting Likelihood
results = EM(data, initial_means, initial_covs, initial_weights)

loglikelihoods = results['loglik']

plt.figure()
plt.plot(range(len(loglikelihoods)), loglikelihoods, linewidth=4)
plt.xlabel('Iteration')
plt.ylabel('Log-likelihood')
plt.rcParams.update({'font.size':16})
plt.tight_layout()


# Fitting a Gaussian mixture model for image Data

# Reading in images
sky     = 'data\\images\\cloudy_sky\\'
river   = 'data\\images\\rivers\\'
sunset  = 'data\\images\\sunsets\\'
forest  = 'data\\images\\trees_and_forest\\'

img_dirs = [sky, river, sunset, forest]
img_files = []
for d in img_dirs:
    img_files += [d+f for f in os.listdir(d) if '.db' not in f]
img_arr = [np.array(Image.open(f)) for f in img_files]
rgb_arr = [[np.mean(img[:,:,0]/256.0), np.mean(img[:,:,1]/256.0), np.mean(img[:,:,2]/256.0)] for img in img_arr]


# Initalize parameters and run algo
np.random.seed(1)
init_means = [rgb_arr[x] for x in np.random.choice(len(rgb_arr), 4, replace=False)]
cov = np.diag(np.var(rgb_arr, axis=0))
init_covariances = [cov, cov, cov, cov]
init_weights = [1/4., 1/4., 1/4., 1/4.]
img_data = [np.array(i) for i in rgb_arr]

out = EM(img_data, init_means, init_covariances, init_weights)


# Evaluating convergence via LL plot
ll = out['loglik']
plt.plot(range(len(ll)),ll,linewidth=4)
plt.xlabel('Iteration')
plt.ylabel('Log-likelihood')
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure()
plt.plot(range(3,len(ll)),ll[3:],linewidth=4)
plt.xlabel('Iteration')
plt.ylabel('Log-likelihood')
plt.rcParams.update({'font.size':16})
plt.tight_layout()


# Evaluating uncertainty
import colorsys


def plot_responsibilities_in_RB(img, resp, title, outfile = None):
    N, K = resp.shape

    HSV_tuples = [(x * 1.0 / K, 0.5, 0.9) for x in range(K)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    RGB_tuples = [i for i in RGB_tuples]

    R = [i[0] for i in img]
    B = [i[1] for i in img]

    resp_by_img_int = [[resp[n][k] for k in range(K)] for n in range(N)]
    cols = [(np.dot(resp_by_img_int[n], np.array(RGB_tuples))) for n in range(N)]

    plt.figure()
    for n in range(len(R)):
        plt.plot(R[n], B[n], 'o', c=cols[n])
    plt.title(title)
    plt.xlabel('R value')
    plt.ylabel('B value')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    if outfile:
        plt.savefig('output\\' + outfile + '.png')

# Plot given a random assignment
N, K = out['resp'].shape
random_resp = np.random.dirichlet(np.ones(K), N)
plot_responsibilities_in_RB(rgb_arr, random_resp, 'Random responsibilities', 'resp_rand')

# Plot after 1 iteration
out = EM(img_data, init_means, init_covariances, init_weights, maxiter=1)
plot_responsibilities_in_RB(rgb_arr, out['resp'], 'After 1 iteration', 'resp_1_iter')

# Plot after 20 iteration
out = EM(img_data, init_means, init_covariances, init_weights, maxiter=20)
plot_responsibilities_in_RB(rgb_arr, out['resp'], 'After 20 iterations', 'resp_20_iter')


# Calculating image distance for first image in array, rgb_arr[0]
for i in range(len(init_means)):
    print(multivariate_normal.pdf(rgb_arr[0], out['means'][i], out['covs'][i]))


# Cluster assignment for each image
img_ll = [max([multivariate_normal.pdf(rgb_arr[i], out['means'][k], out['covs'][k])
                    for k in range(4)]) for i in range(len(rgb_arr))]
img_cluster = [np.argmax([multivariate_normal.pdf(rgb_arr[i], out['means'][k], out['covs'][k])
                          for k in range(4)]) for i in range(len(rgb_arr))]

img_df = pd.DataFrame()
img_df['cluster'] = img_cluster
img_df['ll'] = img_ll
img_df['file'] = img_files

top_5 = img_df[img_df['cluster']==0].sort_values('ll', ascending= False)[:5]

for i in top_5['file']:
    img = Image.open(i)
    img.show()







