from sklearn import cluster
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.metrics import v_measure_score
import json
from sklearn.metrics import v_measure_score
import time


def load_data():
    # load JSON 20newsgroup data
    with open("newsgroups.json") as fd:
        data = json.load(fd)

    target = data["target"]
    content = data["content"]
    target_names = data["target_names"]
    target_value_list = list(target.values())  # メッセージのカテゴリ ID
    content_value_list = list(content.values())  # メッセージテキスト本体
    target_namevalue_list = list(target_names.values())

    return target_value_list, content_value_list, target_namevalue_list


def cross_table(target, predict):
    result = [[0] * len(set(target)) for i in range(len(set(predict)))]

    for p, t in zip(predict, target):
        result[p][t] += 1

    return result


def purity(CT):
    return np.sum(np.max(CT, axis=1)) / np.sum(CT)


if __name__ == "__main__":
    target_value_list, content_value_list, target_namevalue_list = load_data()

    start_time = time.perf_counter()
    tfidf_transformer = TfidfVectorizer()
    vector = tfidf_transformer.fit_transform(content_value_list)

    k_means = cluster.KMeans(n_clusters=20)
    k_means.fit(vector)
    predicted = k_means.predict(vector)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(elapsed_time)

    for p, t in zip(predicted, target_value_list):
        print(f"   true_cluster: {t}  predicted_cluster: {p}")

    CT = cross_table(target_value_list, predicted)

    purity_val = purity(CT)
    print(f"CT: {CT}")
    print(f"Purity:    {purity_val}")
    print(f"V-measure: {v_measure_score(target_value_list, predicted)}")
