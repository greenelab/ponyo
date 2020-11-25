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


def normalize_expression_data(
    base_dir, config_filename, raw_input_data_filename, normalized_data_filename
):
    """
    0-1 normalize the expression data.

    Arguments
    ----------
    base_dir: str
        Root directory containing analysis subdirectories

    config_filename: str
        File containing user defined parameters

    raw_input_data_filename: str
        File containing raw expression data

    normalize_data_filename:
        Output file containing normalized expression data
    """
    # Read in config variables
    params = utils.read_config(config_filename)

    # Read data
    data = pd.read_csv(raw_input_data_filename, header=0, sep="\t", index_col=0)
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
    scaler_filename = params["scaler_transform_file"]

    outfile = open(scaler_filename, "wb")
    pickle.dump(scaler, outfile)
    outfile.close()

    # Save scaled data
    data_scaled_df.to_csv(normalized_data_filename, sep="\t", compression="xz")


def train_vae(config_filename, input_data_filename):
    """
    Trains VAE model using parameters set in config file

    Arguments
    ----------
    config_filename: str
        File containing user defined parameters

    input_data_filename: str
        File path corresponding to input dataset to use

    """

    # Read in config variables
    params = utils.read_config(config_filename)

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
    normalized_data = pd.read_csv(input_data_filename, header=0, sep="\t", index_col=0)

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
