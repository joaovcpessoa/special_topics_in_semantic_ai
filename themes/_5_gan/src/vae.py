import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import backend
import keras.datasets as kds

import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA, PCA

num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = kds.mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

plt.figure(figsize=(12, 8))
for i in range(24):
    ax = plt.subplot(4, 6, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0, stddev=0.1)
    return z_mean, K.exp(z_log_sigma) * epsilon

original_dim = 784 # 28x28
latent_dim = 2

inputs = keras.Input(shape=(original_dim,))
h = layers.Dense(64, activation='relu')(inputs)
z_mean = layers.Dense(latent_dim)(h)
z = layers.Lambda(sampling)([z_mean, z_log_sigma])
encoder01 = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(64, activation='relu')(latent_inputs)
outputs = layers.Dense(original_dim, activation='sigmoid')(x)
decoder01 = keras.Model(latent_inputs, outputs, name='decoder')

outputs =-decoder01(encoder01(inputs[2]))
vae01= keras.Model(inputs, outputs, name='vae_mlp')
vae.summary()
tf.keras.utils.plot_model(vae01, show_shapes=True)

reconstruction_loss = keras.losses.binary_crossentropy(inputs. outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
kl_loss *= -0,5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae01.add_loss(vae_loss)

vae1.compile(optimizer='adam')

history = vae01.fit(x_train, x_train, epochs=200, batch_size=256, validation_date=(x_test, x_test), verbose=1)

decoded_imgs01 = vae01.predic(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Reconstrução
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs01[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
    
plt.figure()
plt.plot(history.history['loss'], label = 'Train')
plt.plot(history.history['loss'], label = 'Validation')
plt.title('Cost function')
plt.ylabel('Error')
plt.xlabel('Epochs')
plt.legend()
plt.show()

x_test_encoded = encoder01.predict(x_test, batch_size=256)[2]

plt.figure(figsize=(8,8))
plt.scatter(x_test_encoded[:,0], x_test_encoded[:,1], c=y_test, cmap=plt.cm.jet)
plt.colorbar()
plt.show()

sampled = np.array([[-2,1], [2,1], [2,-2], [0,0]])
rec = decoder01.predict(sampled)
plt.figure(figsize=(10, 4))
for i in range(4):
    ax = plt.subplot(1, 4, i + 1)
    plt.imshow(rec[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

n = 15
figure = np.zeros((28 * n, 28 * n))
grid_x = np.linspace(np.min(x_test_encoded[:,0], np.max(x_test_encoded[:,0]), n))
grid_y = np.linspace(np.min(x_test_encoded[:,1], np.max(x_test_encoded[:,1]), n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder01.predict(z_sample, verbose=0)
        digit = x_decoded[0].reshape(28,28)
        figure[i * 28: (i+1)*28, j * 28: (j+1)*28] = digit
        

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()