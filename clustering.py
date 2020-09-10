import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
import progressbar
from IPython.core.display import clear_output

from nba_stats.read_write.basic_stats import ReadDatabase

class NormaliseFeatures:
    def __init__(self, X):
        self.mu = np.mean(X, axis=0)
        self.s = (np.max(X, axis=0) - np.min(X, axis=0))
        self.X = X
        
    def normalised(self, data=np.zeros((0,0))):
        transform_data = self.X if np.size(data) == 0 else data
        return (transform_data - self.mu) / self.s
    
    def revert(self, results):
        return results * self.s + self.mu

def find_clusters(X, clusters):
    (m, n) = np.shape(X)
    K = np.size(clusters, axis=0)
    current_distance = np.zeros((m, K))
    
    for i in range(K):
        temp_diff = X - clusters[i,:]
        current_distance[:, i] = np.linalg.norm(temp_diff, axis=1)
        
    return np.argmin(current_distance, axis=1)

def new_clusters(X, closest, K):
    (m, n) = np.shape(X)
    
    if 0 in [np.size(X[closest == k,:], axis=0) for k in range(K)]:
        # print('new clusters: \n', init_clusters)
        return np.zeros((0,0))
    else:
        return np.array([np.mean(X[closest == k,:], axis=0) for k in range(K)])

def cost_function(X, closest, current_clusters):
    (m, n) = np.shape(X)
    
    matching_clusters = current_clusters[closest, :]
    distance_from = X - matching_clusters
    
    return 1 / m * sum(np.linalg.norm(distance_from, axis=1))

def cluster_matches(X_norm, clusters_norm, closest):
    K = np.size(clusters_norm, 0)
    (m, n) = np.shape(X_norm)

    idx_x = np.concatenate((np.reshape(np.arange(m), (m, 1)), X_norm), axis=1)
    close_min_max = np.zeros((K, 6))
    for k in range(K):
        matched_samples = idx_x[closest == k, :].copy()
        temp_diff = matched_samples[:,1:] - clusters_norm[k,:]
        distance = np.linalg.norm(temp_diff, axis=1)
        sign_diff = np.sign(temp_diff[:,0]) * distance

        close_min_max[k, 0] = matched_samples[np.argmin(distance), 0]
        close_min_max[k, 1] = matched_samples[np.argmin(sign_diff), 0]
        close_min_max[k, 2] = matched_samples[np.argmax(sign_diff), 0]

        remaining = np.array([[x for x in matched_samples[:,0] if x not in close_min_max[k, 0:2]]])
        rand_idx = np.random.randint(1, np.size(remaining), 3)
        close_min_max[k, 3:6] = remaining[0,rand_idx]
    
    return close_min_max

def run_clustering(X, K, runs, iterations, suppress=False):
    nan_fields = [max(x) for x in np.isnan(X).transpose()]
    assert max(nan_fields) != True, 'Nans contained in the following fields: {}'.format(list(np.where(nan_fields)[0]))

    normaliser = NormaliseFeatures(X)
    X_norm = normaliser.normalised()

    (m, n) = np.shape(X)
    best_cost = 10^7
    first_cost = None
    best_run = 0

    for j in progressbar.progressbar(range(runs)):
        cost_tracking = np.zeros((iterations, 1))
        rand_idx = np.random.randint(1, m, K)
        init_clusters = X_norm[rand_idx, :]
        # print('init clusters: \n', init_clusters)
        clusters = init_clusters

        for i in range(iterations):
            closest_cluster = find_clusters(X_norm, clusters)
            clusters = new_clusters(X_norm, closest_cluster, K)
            if np.size(clusters) == 0:
                cost_tracking[i] = 10^8
                if not suppress:
                    clear_output()
                    print('Cluster has no samples. Run skipped: {}, Iteration: {}'.format(j+1, i+1))
                break
            cost_tracking[i] = cost_function(X_norm, closest_cluster, clusters)

        if cost_tracking[i][0] < best_cost:
            final_clusters = clusters
            final_closest = closest_cluster
            final_cost = cost_tracking
            best_cost = cost_tracking[i][0]
            if first_cost == None:
                first_cost = best_cost
            best_run = j

    matches = cluster_matches(X_norm, final_clusters, final_closest)
    final_clusters = normaliser.revert(final_clusters)
    print('Best Run: {},  Improvement from first run: {:.1%}'.format(best_run, best_cost/first_cost-1))
    
    return final_clusters, final_closest, final_cost, matches

def plot_cluster(X, closest_cluster=np.zeros((0,0)), clusters=np.zeros((0,0)), init_clusters=np.zeros((0,0))):
    plt.figure()
    dims = min(np.size(X, axis=1), 3)
    if dims == 3:
        ax = plt.axes(projection='3d')
        
    if np.size(closest_cluster) != 0:
        colours = np.array(sns.color_palette("colorblind", len(set(closest_cluster))))[closest_cluster]
    else:
        colours = 'k'
    if dims == 3:
        ax.scatter3D(X[:,0], X[:,1], X[:,2], marker='o', alpha=0.5, c=colours)
    else:
        plt.scatter(X[:,0], X[:,1], marker='o', alpha=0.5, c=colours)

    if np.size(clusters) != 0:
        if dims == 3:
            ax.scatter3D(clusters[:,0], clusters[:,1], clusters[:,2], marker='d', c='k')
        else:
            plt.plot(clusters[:,0], clusters[:,1], 'kd')
            
    if np.size(init_clusters) != 0:
        if dims == 3:
            ax.scatter3D(init_clusters[:,0], init_clusters[:,1], init_clusters[:,2], marker='d', c='r')
        else:
            plt.plot(init_clusters[:,0], init_clusters[:,1], 'rd')