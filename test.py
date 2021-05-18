import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


params = {"activation": tf.nn.leaky_relu, # recommended vs tf.nn.relu
          "optimizer": tf.optimizers.RMSprop(learning_rate=0.001), #tf.optimizers.Adam(learning_rate=0.001) does not converge
          "noise": 0.1, # better robustness
          "skip discriminator training": 1, # train discriminator every fixed number of epochs (1 does not skip training)
          "random variables": 2,
          "batch_size": float('inf'), # use a single batch
          "dropout": 0.5,
          "loss": "Entropy" # Wasserstein or Entropy
        }

np.random.seed(1)
data = np.array([(x-0.5,0.7+(x-0.5)**2) for x in np.random.sample(400)]+[(x-1,2*(x-1)**3) for x in np.random.sample(40)])

#plt.scatter(data[:, 0], data[:, 1])
#plt.show()

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
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid if params["loss"] == "Entropy" else lambda x: x)
])

# create dynamically updated figure
plt.ion()
figure = plt.figure()
axis = figure.add_subplot(1,1,1)

optimizer = params["optimizer"]
batch_size = params["batch_size"]
latent_random_variables = params["random variables"]
epsilon = params["noise"]
visualization_random = tf.random.normal(shape=(200, latent_random_variables)) # visualize the same 200 examples each time
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
            prediction_negative = discriminator(generated+epsilon*tf.random.normal(shape=generated.shape), training=False)
            if params["loss"] == "Wasserstein":
                loss = tf.reduce_sum(prediction_negative)
            else:
                loss = tf.reduce_sum(tf.math.log(1-prediction_negative))
            #loss = -tf.losses.BinaryCrossentropy()(negative_labels[:batch_size], prediction_negative)
        optimizer.apply_gradients(zip(tape.gradient(loss, generator.trainable_variables), generator.trainable_variables))

        if epoch % params["skip discriminator training"] == 0:
            # train discriminator
            random = tf.random.normal(shape=(batch_size, latent_random_variables))
            with tf.GradientTape() as tape:
                generated = generator(random)
                batch_data = data[batch_pos:batch_pos+batch_size, :]
                prediction_negative = discriminator(generated+epsilon*tf.random.normal(shape=generated.shape))
                prediction_positive = discriminator(batch_data+epsilon*tf.random.normal(shape=batch_data.shape))
                if params["loss"] == "Wasserstein":
                    loss = -tf.reduce_sum(prediction_negative)+tf.reduce_sum(prediction_positive)
                else:
                    loss = -tf.reduce_sum(tf.math.log(1-prediction_negative))-tf.reduce_sum(tf.math.log(prediction_positive))
                #loss = tf.losses.MeanSquaredError()(negative_labels[:batch_size], prediction_negative) \
                #    + tf.losses.MeanSquaredError()(positive_labels[:batch_size], prediction_positive)
            optimizer.apply_gradients(zip(tape.gradient(loss, discriminator.trainable_variables), discriminator.trainable_variables))
        batch_pos += batch_size

    if epoch % 10 == 0:
        generated = generator(visualization_random)
        axis.clear()
        axis.scatter(data[:, 0], data[:, 1])
        axis.scatter(generated[:, 0], generated[:, 1])
        plt.xlim([-1.5,1.5])
        plt.ylim([-1.5,1.5])
        plt.title('Epoch '+str(epoch))
        figure.canvas.draw()
        figure.canvas.flush_events()

plt.show()