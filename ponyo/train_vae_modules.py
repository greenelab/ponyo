"""
Author: Alexandra Lee
Date Created: 11 March 2020

Scripts related to training the VAE including
1. Normalizing gene expression data
2. Wrapper function to input training parameters and run vae
training in `vae.tybalt_2layer_model`
"""

from ponyo import vae, utils
import os
import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing

import warnings


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


np.random.seed(123)


def normalize_expression_data(
    base_dir, config_file, raw_input_data_file, normalized_data_file
):
    """
    0-1 normalize the expression data.

    Arguments
    ----------
    base_dir: str
        Root directory containing analysis subdirectories

    config_file: str
        File containing user defined parameters

    raw_input_data_file: str
        File containing raw expression data

    normalize_data_file:
        Output file containing normalized expression data
    """
    # Read in config variables
    params = utils.read_config(config_file)

    # Read data
    data = pd.read_csv(raw_input_data_file, header=0, sep="\t", index_col=0)
    print(
        "input: dataset contains {} samples and {} genes".format(
            data.shape[0], data.shape[1]
        )
    )

    # 0-1 normalize per gene
    scaler = preprocessing.MinMaxScaler()
    data_scaled_df = scaler.fit_transform(data)
    data_scaled_df = pd.DataFrame(
        data_scaled_df, columns=data.columns, index=data.index
    )

    print(
        "Output: normalized dataset contains {} samples and {} genes".format(
            data_scaled_df.shape[0], data_scaled_df.shape[1]
        )
    )

    # Save scaler transform
    scaler_file = params["scaler_transform_file"]

    outfile = open(scaler_file, "wb")
    pickle.dump(scaler, outfile)
    outfile.close()

    # Save scaled data
    data_scaled_df.to_csv(normalized_data_file, sep="\t", compression="xz")


def train_vae(config_file, input_data_file):
    """
    Trains VAE model using parameters set in config file

    Arguments
    ----------
    config_file: str
        File containing user defined parameters

    input_data_file: str
        File path corresponding to input dataset to use

    """

    # Read in config variables
    params = utils.read_config(config_file)

    # Load parameters
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))
    dataset_name = params["dataset_name"]
    learning_rate = params["learning_rate"]
    batch_size = params["batch_size"]
    epochs = params["epochs"]
    kappa = params["kappa"]
    intermediate_dim = params["intermediate_dim"]
    latent_dim = params["latent_dim"]
    epsilon_std = params["epsilon_std"]
    train_architecture = params["NN_architecture"]
    validation_frac = params["validation_frac"]

    # Read data
    normalized_data = pd.read_csv(input_data_file, header=0, sep="\t", index_col=0)

    print(
        "input dataset contains {} samples and {} genes".format(
            normalized_data.shape[0], normalized_data.shape[1]
        )
    )

    # Train (VAE)
    vae.tybalt_2layer_model(
        learning_rate,
        batch_size,
        epochs,
        kappa,
        intermediate_dim,
        latent_dim,
        epsilon_std,
        normalized_data,
        base_dir,
        dataset_name,
        train_architecture,
        validation_frac,
    )
