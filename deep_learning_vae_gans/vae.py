import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def encoder(x, hidden_units, n_z):
    h = tf.layers.dense(inputs=x, units=hidden_units, activation=tf.nn.relu)
    mu = tf.layers.dense(inputs=h, units=n_z, activation=None)
    sigma = tf.layers.dense(inputs=h, units=n_z, activation=tf.nn.softplus)
    sigma = sigma + 1e-6 # Adding small number to avoid numeric instability
    return mu, sigma

def decoder(z, hidden_units, shape):
    h = tf.layers.dense(inputs=z, units=hidden_units, activation=tf.nn.relu)
    output = tf.layers.dense(inputs=h, units=shape, activation=tf.nn.sigmoid)
    return output

def sample_z(mu, sigma):
    eps = tf.random_normal(shape=tf.shape(mu))
    z = mu + sigma*eps
    return z

def vae_loss(x_true, x_pred, mu, sigma):
    expected_log_likelihood = -tf.keras.losses.binary_crossentropy(x_true, x_pred)
    expected_log_likelihood = tf.reduce_mean(expected_log_likelihood)

    kl = -tf.log(sigma) + 0.5*(sigma**2 + mu**2) - 0.5
    kl = tf.reduce_mean(kl, axis=1)

    elbo = tf.reduce_mean(expected_log_likelihood - kl)
    return -elbo # Adding minus since we want to maximize ELBO

if __name__ == "__main__":
    df = pd.read_csv("train.csv")
    X = df.drop("label", axis=1).values/255.0
    X = (X > 0.5).astype(np.float32)

    y = df["label"].values

    hidden_units = 400
    n_z = 200

    graph_ = tf.Graph()
    with graph_.as_default():
        x = tf.placeholder(shape=(None, X.shape[1]), dtype=tf.float32)

        z_mu, z_sigma = encoder(x, hidden_units, n_z)
        z = sample_z(z_mu, z_sigma)
        output = decoder(z, hidden_units, X.shape[1])

        loss = vae_loss(x, output, z_mu, z_sigma)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss)

    batch_size = 128
    epochs = 30
    steps = X.shape[0] // batch_size
    losses = []

    with tf.Session(graph=graph_) as sess:
        tf.set_random_seed(42)
        sess.run(tf.global_variables_initializer())

        for i in range(0, epochs):
            loss_sum = 0
            for j in range(0, steps):
                start = j*batch_size
                end = (j+1)*batch_size

                X_batch = X[start:end, :]
                _, L = sess.run([optimizer, loss], feed_dict={x: X_batch})

                loss_sum += L
            losses.append(loss_sum/steps)

        plt.plot(range(1, epochs+1), losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

        row = 0; label = 0; break_label = 9
        while True:
            if y[row] != label:
                row += 1
                continue

            plt.subplot(1, 2, 1)
            plt.imshow(X[row, :].reshape(28, 28), cmap='gray')
            plt.title("Original sample for class {}".format(y[row]))

            plt.subplot(1, 2, 2)
            X_plot = sess.run([output], feed_dict={x: X[row, :].reshape(1, X.shape[1])})

            X_plot = X_plot[0].reshape(28, 28)
            plt.imshow(X_plot, cmap='gray')
            plt.title("Generated sample for class {}".format(y[row]))
            plt.show()

            label += 1

            if label == break_label+1:
                break
