import operator
import math
import random

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
        temp_centroids.append([random_data_point, from_data[random_data_point]])
    return temp_centroids


def prepare_centroid_list(amount):
    temp_clusters = []
    for temp_cluster in range(0, amount):
        temp_clusters.append([])
    return temp_clusters


test_set = np.genfromtxt('dataset.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

amount_of_centroids = 4  # Amount of random centroids the program will use
centroids = generate_random_centroids(amount_of_centroids, test_set)  # Generate N random centroids
clusters = prepare_centroid_list(amount_of_centroids)  # Prepare cluster list with N centroids


# Loop each data point
for data_point in test_set:
    distance_to_centroids = []

    for centroid in centroids:
        distance_to_centroids.append(get_distance(data_point, centroid[1], len(centroid)))

    clusters[np.argmin(distance_to_centroids)].append(data_point)

for cluster in clusters:
    average_point = 0

    for point in cluster:
        average_point += point

    centroids[clusters.index(cluster)] = (average_point / len(cluster))

print('New centroids: ', centroids)