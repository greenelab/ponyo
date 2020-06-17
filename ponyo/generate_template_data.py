"""
Author: Alexandra Lee
Date Created: 15 June 2020

Script to generate gene expression compendia using experiment-preserving 
approach using a specific template experiment.
This script is used to generate a null set of gene expression data with
some context, defined by the template experiment.
"""

import os
import ast
import pandas as pd
import numpy as np
import random
import glob
import pickle
from keras.models import load_model
from sklearn import preprocessing

from ponyo import generate_labeled_data

import warnings

warnings.filterwarnings(action="ignore")

from numpy.random import seed

randomState = 123


def shift_template_experiment(
    normalized_data_file,
    selected_experiment_id,
    NN_architecture,
    dataset_name,
    scaler,
    local_dir,
    base_dir,
    run,
):
    """
    Generate simulated data using the selected_experiment_id as a template
    experiment using the same workflow as simulate_compendia in generate_data_parallel.py

    This will return a file with a single simulated experiment following the workflow mentioned.
    This function can be run multiple times to generate multiple simulated experiments from a 
    single selected_experiment_id.

    Arguments
    ----------
    normalized_data_file: str
        File containing normalized gene expression data

        ------------------------------| PA0001 | PA0002 |...
        05_PA14000-4-2_5-10-07_S2.CEL | 0.8533 | 0.7252 |...
        54375-4-05.CEL                | 0.7789 | 0.7678 |...
        ...                           | ...    | ...    |...

    selected_experiment_id: str
        Experiment id selected as template 

    NN_architecture: str
        Name of neural network architecture to use.
        Format 'NN_<intermediate layer>_<latent layer>'

    dataset_name: str
        Name for analysis directory. Either "Human" or "Pseudomonas"

    scaler: minmax model
        Model used to transform data into a different range

    local_dir: str
        Parent directory on local machine to store intermediate results

    base_dir: str
        Root directory containing analysis subdirectories

    run: int
        Simulation run

    Returns
    --------
    simulated_data_file: str
        File containing simulated gene expression data

    """

    # Files
    NN_dir = os.path.join(base_dir, dataset_name, "models", NN_architecture)
    latent_dim = NN_architecture.split("_")[-1]

    model_encoder_file = glob.glob(os.path.join(NN_dir, "*_encoder_model.h5"))[0]

    weights_encoder_file = glob.glob(os.path.join(NN_dir, "*_encoder_weights.h5"))[0]

    model_decoder_file = glob.glob(os.path.join(NN_dir, "*_decoder_model.h5"))[0]

    weights_decoder_file = glob.glob(os.path.join(NN_dir, "*_decoder_weights.h5"))[0]

    # Load saved models
    loaded_model = load_model(model_encoder_file)
    loaded_decode_model = load_model(model_decoder_file)

    loaded_model.load_weights(weights_encoder_file)
    loaded_decode_model.load_weights(weights_decoder_file)

    # Read data
    normalized_data = pd.read_csv(normalized_data_file, header=0, sep="\t", index_col=0)

    # Get corresponding sample ids
    sample_ids = generate_labeled_data.get_sample_ids(
        selected_experiment_id, dataset_name
    )

    # Gene expression data for selected samples
    selected_data_df = normalized_data.loc[sample_ids]

    # Encode selected experiment into latent space
    data_encoded = loaded_model.predict_on_batch(selected_data_df)
    data_encoded_df = pd.DataFrame(data_encoded, index=selected_data_df.index)

    # Get centroid of original data
    centroid = data_encoded_df.mean(axis=0)

    # Add individual vectors(centroid, sample point) to new_centroid

    # Encode original gene expression data into latent space
    data_encoded_all = loaded_model.predict_on_batch(normalized_data)
    data_encoded_all_df = pd.DataFrame(data_encoded_all, index=normalized_data.index)

    data_encoded_all_df.head()

    # Find a new location in the latent space by sampling from the latent space
    encoded_means = data_encoded_all_df.mean(axis=0)
    encoded_stds = data_encoded_all_df.std(axis=0)

    latent_dim = int(latent_dim)
    new_centroid = np.zeros(latent_dim)

    for j in range(latent_dim):
        new_centroid[j] = np.random.normal(encoded_means[j], encoded_stds[j])

    shift_vec_df = new_centroid - centroid
    # print(shift_vec_df)

    simulated_data_encoded_df = data_encoded_df.apply(
        lambda x: x + shift_vec_df, axis=1
    )

    # Decode simulated data into raw gene space
    simulated_data_decoded = loaded_decode_model.predict_on_batch(
        simulated_data_encoded_df
    )

    simulated_data_decoded_df = pd.DataFrame(
        simulated_data_decoded,
        index=simulated_data_encoded_df.index,
        columns=selected_data_df.columns,
    )

    simulated_data_scaled = scaler.inverse_transform(simulated_data_decoded_df)

    simulated_data_scaled_df = pd.DataFrame(
        simulated_data_scaled,
        columns=simulated_data_decoded_df.columns,
        index=simulated_data_decoded_df.index,
    )

    # Save
    out_file = os.path.join(
        local_dir,
        "pseudo_experiment",
        "selected_simulated_data_" + selected_experiment_id + "_" + str(run) + ".txt",
    )

    simulated_data_scaled_df.to_csv(out_file, float_format="%.3f", sep="\t")
