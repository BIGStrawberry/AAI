import math
import random
import numpy as np
import matplotlib.pyplot as plt


def get_distance(a, b):
    """
    Calculates the Euclidean distance between point A and point B
    :param a: point A
    :param b: point B
    :return:
    """
    distance = 0
    for x in range(len(b)):
        distance += pow(a[x] - b[x], 2)
    return math.sqrt(distance)


def generate_random_centroids(amount, from_data):
    temp_centroids = []

    for temp_centroid in range(0, amount):
        random_data_point = random.randint(1, len(test_set))

        # if random_data_point not in temp_centroids:
        temp_centroids.append(from_data[random_data_point])

    return temp_centroids


def prepare_cluster_list(amount):
    temp_clusters = []
    for temp_cluster in range(0, amount):
        temp_clusters.append([])
    return temp_clusters


# Main
test_set = np.genfromtxt('dataset.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

amount_of_centroids = 4  # Amount of random centroids the program will use
centroids = generate_random_centroids(amount_of_centroids, test_set)

if len(centroids) < 1:
    centroids = generate_random_centroids(amount_of_centroids, test_set)  # Generate N random centroids
    print("generating starting centroids")

for x in range(1, 10):
    for loop in range(1, 25):
        print("Loop: #", loop)

        print("Using centroids: ", centroids)
        clusters = prepare_cluster_list(amount_of_centroids)  # Prepare cluster list with N centroids

        # Loop each data point
        for data_point in test_set:
            distance_to_centroids = []

            for centroid in centroids:
                distance_to_centroids.append(get_distance(data_point, centroid))

            clusters[np.argmin(distance_to_centroids)].append(data_point)

        # Calculate the center of a cluster to place our centroids

        # centroids = []
        for cluster in clusters:
            average_point = 0

            for point in cluster:
                average_point += point

            # A bug in Numpy causes this line to give an error randomly.
            # https://github.com/numpy/numpy/issues/7453
            centroids[clusters.index(cluster)] = (average_point / len(cluster))
            # centroids.append(average_point/len(cluster))



    print("Finished.")
    print("Centroids Used: ")
    print(centroids)
    print("Clusters Found: ")
    print(clusters)

