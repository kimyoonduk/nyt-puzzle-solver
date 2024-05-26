import re
import nltk

from gensim.models import KeyedVectors

import gensim.downloader as api

import numpy as np
from sklearn.cluster import KMeans
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score
from scipy.spatial import ConvexHull
import time


def modified_kmeans(model, word_list, K=4):
    word_vectors = [model[word] for word in word_list]
    kmeans = KMeans(n_clusters=K, random_state=42).fit(word_vectors)
    labels = kmeans.labels_

    clusters = [[] for _ in range(K)]
    for i, label in enumerate(labels):
        clusters[label].append(i)

    # Ensure equal-sized partitions
    while True:
        max_cluster = max(clusters, key=len)
        min_cluster = min(clusters, key=len)

        if len(max_cluster) - len(min_cluster) <= 1:
            break

        max_cluster_center = np.mean([word_vectors[i] for i in max_cluster], axis=0)
        distances = [
            np.linalg.norm(word_vectors[i] - max_cluster_center) for i in max_cluster
        ]
        furthest_index = max_cluster[np.argmax(distances)]

        max_cluster.remove(furthest_index)
        min_cluster.append(furthest_index)

    string_clusters = [[word_list[i] for i in cluster] for cluster in clusters]

    return string_clusters


def measure_avg_cosine_similarity(model, clusters, verbose=False):
    avg_cosine_similarities = []
    start_time = time.time()

    for cluster in clusters:
        cluster_vectors = [model[word] for word in cluster]

        # Average Cosine Similarity
        cosine_similarities = []
        for i in range(len(cluster_vectors)):
            for j in range(i + 1, len(cluster_vectors)):
                cosine_similarity = np.dot(cluster_vectors[i], cluster_vectors[j]) / (
                    np.linalg.norm(cluster_vectors[i])
                    * np.linalg.norm(cluster_vectors[j])
                )
                cosine_similarities.append(cosine_similarity)
        avg_cosine_similarity = np.mean(cosine_similarities)
        avg_cosine_similarities.append(avg_cosine_similarity)

    if verbose:
        print(f"Average Cosine Similarity Time: {time.time() - start_time:.4f}s")

        for i, cluster in enumerate(clusters):
            print(f"ACS {i + 1}: {cluster}: {avg_cosine_similarities[i]:.4f}")

    return avg_cosine_similarities


def measure_centroid_similarity(model, clusters, verbose=False):
    centroid_similarities = []

    start_time = time.time()

    for cluster in clusters:
        cluster_vectors = [model[word] for word in cluster]

        # Cluster Centroid
        centroid = np.mean(cluster_vectors, axis=0)

        # Centroid Similarity
        similarities = []
        for vector in cluster_vectors:
            similarity = np.dot(vector, centroid) / (
                np.linalg.norm(vector) * np.linalg.norm(centroid)
            )
            similarities.append(similarity)

        centroid_similarity = np.mean(similarities)
        centroid_similarities.append(centroid_similarity)

    if verbose:
        print(f"Centroid Similarity Time: {time.time() - start_time:.4f}s")
        for i, cluster in enumerate(clusters):
            print(f"CS {i + 1}: {cluster}: {centroid_similarities[i]:.4f}")

    return centroid_similarities


def measure_convex_hull(model, clusters, verbose=False):
    quadrangle_areas = []

    start_time = time.time()

    for cluster in clusters:
        cluster_vectors = [model[word] for word in cluster]

        if len(cluster_vectors) == 4:
            # Extract x and y coordinates from the cluster vectors
            x = [vector[0] for vector in cluster_vectors]
            y = [vector[1] for vector in cluster_vectors]

            # Calculate the area using the shoelace formula
            area = 0.5 * abs(
                sum(x[i] * y[(i + 1) % 4] - x[(i + 1) % 4] * y[i] for i in range(4))
            )

            quadrangle_areas.append(area)
        else:
            quadrangle_areas.append(0)

    if verbose:
        print(f"Quadrangle Area Time: {time.time() - start_time:.4f}s")
        for i, cluster in enumerate(clusters):
            print(f"QA {i + 1}: {cluster}: {quadrangle_areas[i]:.4f}")

    return quadrangle_areas


def measure_silhouette_score(model, clusters, verbose=False):
    start_time = time.time()

    # Flatten the clusters into a single list of words
    words = [word for cluster in clusters for word in cluster]

    # Create a list of labels corresponding to the cluster each word belongs to
    labels = [i for i, cluster in enumerate(clusters) for _ in range(len(cluster))]

    # Get the word vectors for all words
    word_vectors = [model[word] for word in words]

    # Calculate the silhouette score
    silhouette_scores = silhouette_score(word_vectors, labels)

    if verbose:
        print(f"Silhouette Score Time: {time.time() - start_time:.4f}s")
        print(f"Silhouette Score: {silhouette_scores:.4f}")

    return silhouette_scores


def __main__():

    resource_dir = "resources"
    word_file = "wordlist-20210729.txt"
    input_file = "strands_input.json"

    with open(Path(resource_dir, word_file), "r") as f:
        word_list = f.read().splitlines()

    with open(Path(resource_dir, input_file), "r") as f:
        input_data = json.load(f)

    path_w2v = api.load("word2vec-google-news-300", return_path=True)
    print(path_w2v)
    model_w2v = KeyedVectors.load_word2vec_format(path_w2v, binary=True)

    path_glove_wiki = api.load("glove-wiki-gigaword-300", return_path=True)
    print(path_glove_wiki)

    model = model_w2v

    input_words = input_data["words"]

    # Perform modified K-means clustering
    clusters = modified_kmeans(model, input_words, K=4)
    measure_avg_cosine_similarity(model, clusters, True)
    measure_centroid_similarity(model, clusters, True)
    measure_convex_hull(model, clusters, True)

    measure_silhouette_score(model, clusters, True)


#
"""
Representation



Clustering method

"""
