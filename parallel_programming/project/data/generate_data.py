import numpy as np

if __name__ == "__main__":
	np.random.seed(42)

	num_samples = 1000
	num_features = 5

	theta = np.array([1.5, 2.2, 3.5, -2.3, -0.5]).reshape((num_features, 1))
	X = np.zeros((num_samples, num_features))
	X[:, 0] = 1.0
	X[:, 1:num_features] = np.random.uniform(-1, 1, size=(num_samples, num_features - 1))
	np.round(X, decimals=3)

	y = np.dot(X, theta) + np.random.normal(loc=0, scale=0.01, size=(num_samples, 1))

	np.savetxt("X.txt", X, fmt='%5.5f')
	np.savetxt("y.txt", y, fmt='%5.5f')
