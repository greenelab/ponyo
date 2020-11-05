"""
Author: Alexandra Lee
Date Created: 30 August 2019

Helper functions for running variational autoencoder `vae.py`.
"""

from keras.callbacks import Callback
from keras import metrics
from keras.layers import Layer
from keras import backend as K
import tensorflow as tf


# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see:
# https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


# Functions
#
# Based on publication by Way et. al. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5728678/)
# Github repo (https://github.com/greenelab/tybalt/blob/master/scripts/vae_pancancer.py)


# Function for reparameterization trick to make model differentiable


def sampling_maker(epsilon_std):
    def sampling(args):
        # Function with args required for Keras Lambda function
        z_mean, z_log_var = args

        # Draw epsilon of the same shape from a standard normal distribution
        epsilon = K.random_normal(shape=tf.shape(z_mean), mean=0.0, stddev=epsilon_std)

        # The latent vector is non-deterministic and differentiable
        # in respect to z_mean and z_log_var
        z = z_mean + K.exp(z_log_var / 2) * epsilon
        return z

    return sampling


class CustomVariationalLayer(Layer):
    """
    Define a custom layer that learns and performs the training
    """

    def __init__(self, original_dim, z_log_var_encoded, z_mean_encoded, beta, **kwargs):
        # https://keras.io/layers/writing-your-own-keras-layers/
        self.is_placeholder = True
        self.original_dim = original_dim
        self.z_log_var_encoded = z_log_var_encoded
        self.z_mean_encoded = z_mean_encoded
        self.beta = beta

        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x_input, x_decoded):
        reconstruction_loss = self.original_dim * metrics.binary_crossentropy(
            x_input, x_decoded
        )

        kl_loss = -0.5 * K.sum(
            1
            + self.z_log_var_encoded
            - K.square(self.z_mean_encoded)
            - K.exp(self.z_log_var_encoded),
            axis=-1,
        )

        return K.mean(reconstruction_loss + (K.get_value(self.beta) * kl_loss))

    def call(self, inputs):
        x = inputs[0]
        x_decoded = inputs[1]
        loss = self.vae_loss(x, x_decoded)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x


class WarmUpCallback(Callback):
    def __init__(self, beta, kappa):
        self.beta = beta
        self.kappa = kappa

    # Behavior on each epoch
    def on_epoch_end(self, epoch, logs={}):
        if K.get_value(self.beta) <= 1:
            K.set_value(self.beta, K.get_value(self.beta) + self.kappa)
