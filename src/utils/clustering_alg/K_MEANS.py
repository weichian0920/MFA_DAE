#/--coding:utf-8/
#/author:Ethan Wang/

from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def create_cluster(sparse_data, nclust):
    def euc_dist(X, Y = None, Y_norm_squared = None, squared = False):
        return 1-(cosine_similarity(X, Y))
    kmeans = KMeans(n_clusters = nclust, init = 'k-means++',max_iter = 300)
    kmeans.labels = kmeans.fit_predict(sparse_data)
    return kmeans.labels, kmeans.cluster_centers_



