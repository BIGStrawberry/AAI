import operator
import math
import numpy as np


def get_season(date):
    """
    Gets season based on a date (y-m-d)
    :param date: date to convert to season
    :return: season (winter, lente, zomer or herfst)
    """
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


def get_neighbors(test_set, point, k=1):
    """
    Calculates the euclidean distance between every point in test set and point to determine the neighbours of point
    :param test_set: data set
    :param point: data point to get neighbours for
    :param k: amount of neighbours of point to return
    :return: k amount of neighbours to point
    """
    assert len(test_set) >= k, "k is set too low"

    neighbors = []

    for group in range(len(test_set)):
        neighbors.append([
            math.sqrt(sum([(a - b) ** 2 for a, b in zip(point, test_set[group][1::])])),
            test_set[group][0]]
        )
    neighbors.sort(key=operator.itemgetter(0))
    return neighbors[:k]


def get_most_common_label(neighbors):
    """
    Gets the most common label for each neighbour
    :param neighbors: list of neighbours
    :return: most common label
    """
    label_votes = {}
    for x in range(len(neighbors)):
        label = get_season(neighbors[x][1])
        if label in label_votes:
            label_votes[label] += 1
        else:
            label_votes[label] = 1
    sorted_votes = sorted(label_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def calculate_k(max_k=100):
    """
    Calculates the most effective K value to use for the KNN algorithm with the given dataset
    :param max_k: maximum amount of neighbours the function is allowed to test
    :return: most effective K value
    """
    print("Calculating best K, max K =", max_k)
    results = []

    for k in range(1, max_k):
        predicted_seasons = []
        correct = 0

        for point in training:
            predicted_seasons.append(get_most_common_label(get_neighbors(test_set, point, k)))

        for x in range(len(predicted_seasons)):
            if predicted_seasons[x] == validation_labels[x]:
                correct += 1

        accuracy = len([x for x in range(len(predicted_seasons)) if predicted_seasons[x] == validation_labels[x]])
        results.append((accuracy, k))

    winner = sorted(results, key=lambda tup: tup[0], reverse=True)[0]

    print("Optimal K =", winner[1])
    return winner[1]

if __name__ == "__main__":

    # Read all the data from datasets
    validation = np.genfromtxt('validation.csv', delimiter=';',
                            usecols=[0],
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

    # Unlabeled weather data
    unlabeled_data = [
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

    # Calculate optimal K to use
    optimal_k = calculate_k()

    # Classify unlabeled data
    for unlabeled_point in unlabeled_data:
        print(unlabeled_point, "is from season", get_most_common_label(get_neighbors(test_set, unlabeled_point, optimal_k)))

