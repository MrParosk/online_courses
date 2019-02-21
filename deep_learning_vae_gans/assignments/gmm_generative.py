import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture

class BayesGenerative:
	def __init__(self):
		self.gaussians = []

	def fit(self, X, y):
		self.classes = len(set(y))
		self.p_y = np.zeros([self.classes, 1])

		for k in range(0, self.classes):
			print("Fitting gmm for k={}".format(k))
			X_k = X[y == k]

			gmm = BayesianGaussianMixture(10)
			gmm.fit(X_k)
			self.gaussians.append(gmm)

			self.p_y[k, :] = X_k.shape[0]

		self.p_y = self.p_y/X.shape[0]

	def sample_given_y(self, c):
		gmm = self.gaussians[c]
		sample = gmm.sample()
		return sample[0].reshape(28, 28)

	def sample(self):
		y_ = np.random.choice(self.gaussians, p=self.p_y)
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
