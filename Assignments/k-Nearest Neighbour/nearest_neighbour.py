import operator
import math

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


def get_neighbors(data, predict, k=1):
    assert len(data) >= k, "k is set too low"

    neighbors = []

    for group in range(len(data)):
        neighbors.append([
            math.sqrt(sum([(a - b) ** 2 for a, b in zip(predict, data[group][1::])])),
            data[group][0]]
        )
    neighbors.sort(key=operator.itemgetter(0))
    return neighbors[:k]


def get_most_common_label(neighbors):
    label_votes = {}
    for x in range(len(neighbors)):
        label = get_season(neighbors[x][1])
        if label in label_votes:
            label_votes[label] += 1
        else:
            label_votes[label] = 1
    sorted_votes = sorted(label_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def knn(data, point, k):
    return get_most_common_label(get_neighbors(data, point, k))


# Main
validation = np.genfromtxt('validation.csv', delimiter=';', usecols=[0],
                           converters={
    5: lambda x: 0 if x == '-1' else float(x),
    7: lambda x: 0 if x == '-1' else float(x)
})

test_set = np.genfromtxt('dataset.csv', delimiter=';',
                         usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                         converters={
                             5: lambda x: 0 if x == '-1' else float(x),
                             7: lambda x: 0 if x == '-1' else float(x)
                         })

training = np.genfromtxt('validation.csv', delimiter=';',
                         usecols=[1, 2, 3, 4, 5, 6, 7],
                         converters={
                             5: lambda x: 0 if x == '-1' else float(x),
                             7: lambda x: 0 if x == '-1' else float(x)
                         })

days = np.genfromtxt('days.csv', delimiter=';',
                     usecols=[1, 2, 3, 4, 5, 6, 7],
                     converters={
                         5: lambda x: 0 if x == '-1' else float(x),
                         7: lambda x: 0 if x == '-1' else float(x)
                     })

validation_labels = []

for x in validation:
    validation_labels.append(get_season(x))


# run nearest neighbour 100 times over the training set and compare it to validation

results = []
times = 100
print('Running kNN %d times' % times)

for k in range(1, times):
    predicted_seasons = []
    correct = 0

    for point in training:
        predicted_seasons.append(knn(test_set, point, k))

    for x in range(len(predicted_seasons)):
        if predicted_seasons[x] == validation_labels[x]:
            correct += 1

        # if (predicted_seasons[x] == validation_labels[x]):
        #     print('k = ', k, ' -- ', predicted_seasons[x], ' == ', validation_labels[x])
        # else:
        #     print('k = ', k, ' -- ', predicted_seasons[x], ' != ', validation_labels[x])

    accuracy = len([x for x in range(len(predicted_seasons)) if predicted_seasons[x] == validation_labels[x]])

    results.append((accuracy, k))

winner = sorted(results, key=lambda tup: tup[0], reverse=True)[0]

print('Best k = ', winner[1], 'with ', winner[0], '% accuracy')

for result in results:
    print('%d%% for k = %d' % (result[0], result[1]))

print('Ran %d times' % times)

unlabeled = [
        (40, 52, 2, 102, 103, 0, 0),
        (25, 48, -18, 105, 72, 6, 1),
        (23, 121, 56, 150, 25, 18, 18),
        (27, 229, 146, 308, 130, 0, 0),
        (41, 65, 27, 123, 95, 0, 0),
        (46, 162, 100, 225, 127, 0, 0),
        (23, -27, -41, -16, 0, 0, -1),
        (28, -78, -106, -39, 67, 0, 0),
        (38, 166, 131, 219, 58, 16, 41),
        (50, 66, 111, 119, 85, 26, 51)
    ]

for x in unlabeled:
    print(knn(test_set, x, 58))

