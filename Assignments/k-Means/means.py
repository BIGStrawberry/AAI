import matplotlib.pyplot as plt
import math
import random
import time

import numpy as np


def get_season(date):
    date %= 10000
    if date < 301:
        return "winter"
    elif 301 <= date < 601:
        return "lente"
    elif 601 <= date < 901:
        return "zomer"
    elif 901 <= date < 1201:
        return "herfst"
    else:  # from 01−12 to end of year
        return "winter"


def get_distance(a, b, length):
    distance = 0
    for x in range(length):
        distance += pow(a[x] - b[x], 2)
    return math.sqrt(distance)


def generate_random_centroids(amount, from_data):
    temp_centroids = []

    for temp_centroid in range(0, amount):
        random_data_point = random.randint(1, len(test_set))
        temp_centroids.append(from_data[random_data_point])

    return temp_centroids


def prepare_cluster_list(amount):
    temp_clusters = []
    for temp_cluster in range(0, amount):
        temp_clusters.append([])
    return temp_clusters

plt.ion()
plt.show()

test_set = np.genfromtxt('dataset.csv', delimiter=';', usecols=[2,3], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
# test_set = np.genfromtxt('dataset.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

colors = np.array(10*["r", "g", "c", "b", "k"])

amount_of_centroids = 4  # Amount of random centroids the program will use
centroids = []
if (len(centroids) < 1):
    centroids = generate_random_centroids(amount_of_centroids, test_set)  # Generate N random centroids
    print("generating starting centroids")

for loop in range(1, 10):
    plt.clf()
    print("Loop: #", loop)

    plt.ylabel('Loop #' + str(loop))
    print("Using centroids: ", centroids)
    clusters = prepare_cluster_list(amount_of_centroids)  # Prepare cluster list with N centroids

    # Loop each data point
    for data_point in test_set:
        distance_to_centroids = []

        for centroid in centroids:
            distance_to_centroids.append(get_distance(data_point, centroid, len(centroid)))
            # print("data: " , data_point)
            # print("centroid: " , centroid)
            # print("centroidSSS: ", centroids)
            plt.scatter(centroid[0], centroid[1], c='k', s=10**2, marker='x')  # plot centroids (x)

        clusters[np.argmin(distance_to_centroids)].append(data_point)

    # Calculate the center of a cluster to place our centroids
    for cluster in clusters:
        average_point = 0

        for point in cluster:
            plt.scatter(point[0], point[1], c=colors[clusters.index(cluster)], alpha=0.2)
            average_point += point

        centroids[clusters.index(cluster)] = (average_point / len(cluster))

    plt.draw()
    plt.pause(0.001)

time.sleep(100000)


