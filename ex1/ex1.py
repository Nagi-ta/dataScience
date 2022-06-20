from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()

plt.figure(figsize=(7, 7))
plt.grid()

colors = ["red", "blue", "green"]

for i, name in enumerate(iris.target_names):
    plt.scatter(
        x=iris.data[iris.target == i, 2],
        y=iris.data[iris.target == i, 3],
        label=name,
        c=colors[i],
    )

# plot
plt.title("Iris petalLength / petalWidth")
iris_dataset = iris["data"]
plt.xlim(
    min(row[2] for row in iris_dataset) - 0.1, max(row[2] for row in iris_dataset) + 0.1
)
plt.ylim(
    min(row[3] for row in iris_dataset) - 0.1, max(row[3] for row in iris_dataset) + 0.1
)
plt.legend(loc="best")
plt.xlabel(iris.feature_names[2], size=14)
plt.ylabel(iris.feature_names[3], size=14)

plt.savefig("Iris-petalLength-petalWidth-203331.pdf")
plt.show()
