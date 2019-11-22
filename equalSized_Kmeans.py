from random import sample
import numpy as np

def distance_function(X, Y=None, Y_norm_squared=None, squared=False):
    """Default distance function"""
    result = []
    for y in Y:
        result.append(np.sum(np.abs(X - y)))
    return result


class ESKMeans:
    """
    Equal-sized Kmeans.

    Attributes
    ----------
    cids : list of ints
        list of all ids, will be removed in future.
    n_clusteres : int
        Number of clusters needed. Default is 2.
    iteration_limit : int
        Limitation of total iterations for Kmeans. Default is 10.
        Notice that as soon as threshold in fit function is met, program will stop.
    df : function(X, Y=None):
        Distance function takes two inputs. X is an one dimensional array. Y is
        a list of one dimensional arrays. This function should return a list
        contains distances from X to every array in Y.

    Methods
    -------
    fit(self, dataset, initial_centers=None, threshold=0.001)
        Use equal-sized kmeans clusterring the dataset. Fitted centers will be
        stored in self.centers and clusterring result of dataset will be returned.
    """
    def __init__(self, cids, n_clusters=2, iteration_limit=10, df=None):
        self.cids = cids
        self.n_clusters = n_clusters
        self.iteration_limit = iteration_limit
        if df:
            self.df = df
        else:
            self.df = distance_function

    def fit(self, dataset, initial_centers=None, threshold=0.001):
        """
        Use equal-sized kmeans clusterring the dataset and store the centers in self.centers.

        Arguments
        ---------
        self : ESKMeans class object.
        dataset : pandas.DataFrame
            Dataframe contains all the data. Each row should be one data point.
        initial_centers : list of data points
            initial centers for starting Kmeans with. If not specified will be
            randomly generated. Length of initial_centers should equal n_clusters.
        threshold : float
            If none of the centers changes more than the threshold in
            this iteration, the program will stop and return the result.
            Otherwise keeps running till meets threshold or reaches iteration_limit.
            Change of center is measured by self.df.

        Returns
        -------
        clusterring result of dataset : 2d array
            a list of arrays where each array contains data points in that group.
            order is the same as self.centers.
        """
        dataset = dataset.copy()
        # check if initial centers are provided
        # randomly choose n points as centers if not provided
        if initial_centers:
            self.centers = initial_centers
        else:
            self.centers = sample(list(dataset.values), self.n_clusters)
        # initialize empty clusters
        clusters = [[] for _ in range(self.n_clusters)]
        # flag for threshold
        flag = True
        # counter times of iterations
        iteration_count = 0
        # while threshold not meets and iteration_limit not reaches
        while flag and iteration_count <= self.iteration_limit:
            iteration_count += 1 # add one iteration
            flag = False # reset flag
            clusters = [[] for _ in range(self.n_clusters)]

            opportunity_cost = {}

            # for each data point in dataset
            # determine which cluster it belongs to
            for idx, r in dataset.iterrows():
                distance_all_centers = list(self.df(r.values, Y=self.centers))
                min_distance_all_centers = min(distance_all_centers)
                c_i = distance_all_centers.index(min_distance_all_centers)
                clusters[c_i].append(idx)
                distance_all_centers.remove(min_distance_all_centers)
                opportunity_cost[idx] = (sum(distance_all_centers)/len(distance_all_centers) - min_distance_all_centers)\
                                        if len(distance_all_centers) > 0 else 0

            # for each cluster
            # check if size exceeds desired equal-sized
            for i in range(self.n_clusters):
                # if overfilled
                if len(clusters[i]) > len(dataset)/self.n_clusters:
                    # diff data points need to be removed and reassigned
                    diff = len(clusters[i]) - len(dataset)/self.n_clusters
                    cluster_ds = dataset.loc[clusters[i], :]

                    # # compute the distances of each data points in this cluster to the cluster center
                    # # order the points by distance to the center
                    # # reassign data points farthest from the center
                    # distances = self.df(self.centers[i], Y=cluster_ds.values)
                    # distance_idx = list(zip(distances, clusters[i]))
                    # distance_idx_sorted = sorted(distance_idx,
                    #                              key=lambda tup: tup[0],
                    #                              reverse=True)
                    # reassign_ds = [tup[1] for tup in distance_idx_sorted[:int(diff)]]

                    cost = [opportunity_cost[idx] for idx, _ in cluster_ds.iterrows()]
                    cost_idx = list(zip(cost, clusters[i]))
                    cost_idx_sorted = sorted(cost_idx,
                                             key=lambda tup: tup[0],
                                             reverse=False)
                    reassign_ds = [tup[1] for tup in cost_idx_sorted[:int(diff)]]

                    # for each data point needs to be reassigned
                    # check for the closest unfull cluster it can go into
                    for d in reassign_ds:
                        unfull_centers = [m for m in range(self.n_clusters)
                                           if len(clusters[m]) <= len(dataset)/self.n_clusters]
                        if len(unfull_centers) == 0:
                            break
                        else:
                            distances_to_unfull = list(self.df(dataset.loc[d, :].values,
                                                          Y=[self.centers[m] for m in unfull_centers]))
                            min_dis_i = distances_to_unfull.index(min(distances_to_unfull))
                            new_c = unfull_centers[min_dis_i]
                            clusters[new_c].append(d)
                            clusters[i].remove(d)

            # Update center based on the new cluster assignment
            new_centers = []
            for c in clusters:
                cluster_ds = dataset.loc[c, :]
                new_centers.append(cluster_ds.mean(axis=0))
            # print(new_centers, iteration_count)

            # check if threshold is met
            for center in new_centers:
                changes = list(self.df(center, self.centers))
                # print("min change is {}".format(min(changes)))
                if min(changes) > threshold:
                    flag = True
            self.centers = list(new_centers)
        return clusters
