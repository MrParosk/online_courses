import pandas as pd
from scipy.stats import multivariate_normal as mvn
import numpy as np
import matplotlib.pyplot as plt

class BayesGenerative:
	def __init__(self):
		self.gaussians = []

	def fit(self, X, y):
		self.classes = len(set(y))
		self.p_y = np.zeros([self.classes, 1])

		for k in range(0, self.classes):
			X_k = X[y == k]

			mean_ = X_k.mean(axis=0)
			cov_ = np.cov(X_k.T)

			self.gaussians.append({"mean": mean_, "cov": cov_})
			self.p_y[k, :] = X_k.shape[0]

		self.p_y = self.p_y / X.shape[0]


	def sample_given_y(self, c):
		g = self.gaussians[c]
		return mvn.rvs(mean=g["mean"], cov=g["cov"])

	def sample(self):
		y_ = np.random.choice(self.classes, p=self.p_y)
		return sample_given_y(y_)

if __name__ == "__main__":
	df = pd.read_csv("train.csv")
	y = df["label"].values
	X = df.drop("label", axis=1).values/255.0

	bg = BayesGenerative()
	bg.fit(X, y)

	for k in range(0, bg.classes):
		sample = bg.sample_given_y(k).reshape(28, 28)
		plt.imshow(sample, cmap='gray')
		plt.title("Generated sample for class {}".format(k))
		plt.show()
