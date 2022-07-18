from sklearn import cluster
import numpy as np
from sklearn.datasets import load_iris


def cross_table(target, predict):
    result = [[0] * len(set(target)) for i in range(len(set(predict)))]

    for p, t in zip(predict, target):
        result[p][t] += 1

    return result


def purity(CT):
    return np.sum(np.max(CT, axis=1)) / np.sum(CT)


if __name__ == "__main__":
    iris = load_iris()
    data = iris.data
    cluster_num = 3

    k_means = cluster.KMeans(n_clusters=cluster_num)
    k_means.fit(data)
    predicted = k_means.predict(data)
    target = iris.target

    for d, p, t in zip(data, predicted, target):
        print(f"data: {d}   true_value: {t}  purity: {p}")
    CT = cross_table(target, predicted)

    purity_val = purity(CT)
    print(f"Purity: {purity_val:.4f}")
