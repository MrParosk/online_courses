import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

def network(x, shape):
	hidden_units=100
	h = tf.layers.dense(inputs=x, units=hidden_units)
	return tf.layers.dense(inputs=h, units=shape)

def loss_val(x, y):
	return tf.reduce_mean(tf.square(x-y))

if __name__ == "__main__":
	df = pd.read_csv("train.csv")
	X = df.drop("label", axis=1).values
	y = df["label"].values

	graph_ = tf.Graph()
	with graph_.as_default():
		x_ = tf.placeholder(shape=(None, X.shape[1]), dtype=tf.float32, name="x_")
		y_ = tf.placeholder(shape=(None, X.shape[1]), dtype=tf.float32, name="y_")
		encoded = network(x_, X.shape[1])
		loss = loss_val(y_, encoded)
		optimizer = tf.train.AdamOptimizer(0.05).minimize(loss)

	batch_size = 128
	epochs = 6
	steps = X.shape[0] // batch_size

	with tf.Session(graph=graph_) as sess:
		tf.set_random_seed(42)
		sess.run(tf.global_variables_initializer())
		for i in range(0, epochs):
			loss_sum = 0

			for j in range(0, steps):
				start = j*batch_size
				end = (j+1)*batch_size

				X_batch = X[start:end,:]
				_, L = sess.run([optimizer, loss], feed_dict={x_: X_batch, y_: X_batch})
				loss_sum += L

			print("Loss: {}".format(loss_sum/steps))

			if i % 2 == 0 and i != 0:
				index = 1
				X_plot = sess.run([encoded], feed_dict={x_: X[index,:].reshape(1, X.shape[1]) })
				X_plot = X_plot[0].reshape(28, 28)
				plt.imshow(X_plot, cmap='gray')
				plt.title("Generated sample for class {}".format(y[index]))
				plt.show()
