import numpy as np
import math
from tqdm import tqdm

K_MEANS_MAXITTS = 100
CENTROID_MERGE_DIST = 0.05

def kmeans_clustering(points, k_means=-1):
    # K-means clustering with class reduction if centers too close
    centroids = points if k_means == -1 else points[np.random.randint(points.shape[0], size=k_means)]
    k_means = centroids.shape[0]
    classes = np.zeros(points.shape[0], dtype=np.float64)
    distances = np.zeros([points.shape[0], k_means], dtype=np.float64)
    for itt in tqdm(range(K_MEANS_MAXITTS)):
        for cnt_idx, c in enumerate(centroids):
            distances[:, cnt_idx] = np.linalg.norm(points - c, axis=1)
        classes = np.argmin(distances, axis=1)
        c = 0
        while (c < centroids.shape[0]):
            centroids[c] = np.mean(points[classes == c], 0)
            centroid_dists = np.concatenate([centroids[:c], centroids[c+1:]], axis=0) - centroids[c]
            if centroid_dists.shape[0] > 0:
                if np.min(np.linalg.norm(centroid_dists, axis=1)) < CENTROID_MERGE_DIST:
                    centroids = np.delete(centroids, c, axis=0)
                    break
            c += 1

    cluster_counts = np.array([len(points[classes == c]) for c in range(len(centroids))])
    return centroids, cluster_counts, classes