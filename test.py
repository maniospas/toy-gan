import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


params = {"activation": tf.nn.leaky_relu, # recommended vs tf.nn.relu
          "optimizer": tf.optimizers.RMSprop(learning_rate=0.001), #tf.optimizers.Adam(learning_rate=0.001) does not converge
          "noise": 0, # better robustness than zero (learn to predict second mode) but a little flipflopping
          "random variables": 2,
          "batch_size": 512, # use a single batch
          "dropout": 0.5
        }

np.random.seed(1)
data = np.array([(x-0.5,0.7+(x-0.5)**2) for x in np.random.sample(200)]+[(x-1,2*(x-1)**3) for x in np.random.sample(200)])


generator = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation=params["activation"], input_shape=(params["random variables"],)),
    tf.keras.layers.Dense(16, activation=params["activation"]),
    tf.keras.layers.Dense(2)
])

discriminator = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation=params["activation"], input_shape=(data.shape[1],)),
    tf.keras.layers.Dropout(params["dropout"]),
    tf.keras.layers.Dense(16, activation=params["activation"]),
    tf.keras.layers.Dropout(params["dropout"]),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

# create dynamically updated figure
plt.ion()
figure = plt.figure()
axis = figure.add_subplot(1,1,1)

optimizer = params["optimizer"]
batch_size = params["batch_size"]
latent_random_variables = params["random variables"]
epsilon = params["noise"]
positive_labels = np.ones((batch_size, 1))
negative_labels = np.zeros((batch_size, 1))
for epoch in range(10001):
    if params["batch_size"] < data.shape[0]:
        np.random.shuffle(data)
    batch_pos = 0
    while batch_pos < data.shape[0]:
        batch_size = min(params["batch_size"], data.shape[0]-batch_pos)
        # train generator
        random = tf.random.normal(shape=(batch_size, latent_random_variables))
        with tf.GradientTape() as tape:
            generated = generator(random)
            prediction_negative = discriminator(generated+epsilon*tf.random.normal(shape=generated.shape))
            loss = -tf.losses.BinaryCrossentropy()(negative_labels[:batch_size], prediction_negative)
        optimizer.apply_gradients(zip(tape.gradient(loss, generator.trainable_variables), generator.trainable_variables))

        # train discriminator
        random = tf.random.normal(shape=(batch_size, latent_random_variables))
        with tf.GradientTape() as tape:
            generated = generator(random)
            batch_data = data[batch_pos:batch_pos+batch_size, :]
            prediction_negative = discriminator(generated+epsilon*tf.random.normal(shape=generated.shape))
            prediction_positive = discriminator(batch_data+epsilon*tf.random.normal(shape=batch_data.shape))
            loss = tf.losses.MeanSquaredError()(negative_labels[:batch_size], prediction_negative) \
                + tf.losses.MeanSquaredError()(positive_labels[:batch_size], prediction_positive)
        optimizer.apply_gradients(zip(tape.gradient(loss, discriminator.trainable_variables), discriminator.trainable_variables))
        batch_pos += batch_size

    if epoch % 10 == 0:
        generated = generator(tf.random.normal(shape=(data.shape[0], latent_random_variables)))
        axis.clear()
        axis.scatter(data[:, 0], data[:, 1])
        axis.scatter(generated[:, 0], generated[:, 1])
        plt.xlim([-1.5,1.5])
        plt.ylim([-1.5,1.5])
        plt.title('Epoch '+str(epoch))
        figure.canvas.draw()
        figure.canvas.flush_events()

plt.show()