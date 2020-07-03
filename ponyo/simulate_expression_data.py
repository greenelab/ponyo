"""
Author: Alexandra Lee
Date Created: 30 August 2019

These scripts generate simulated compendia using the low-dimensional
representation of the gene expressiond data, created by training the
VAE on gene expression data.
"""

import os
import pandas as pd
import numpy as np
import glob
import warnings
from keras.models import load_model
from sklearn import preprocessing


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

np.random.seed(123)


def get_sample_ids(experiment_id, dataset_name, sample_id_colname):
    """
    Returns sample ids (found in gene expression df) associated with
    a given list of experiment ids (found in the metadata)

    Arguments
    ----------
    experiment_ids_file: str
        File containing all cleaned experiment ids

    dataset_name: str
        Name for analysis directory. Either "Human" or "Pseudomonas"

    sample_id_colname: str
        Column header that contains sample id that maps expression data
        and metadata

    """
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))

    if "pseudomonas" in dataset_name.lower():
        # metadata file
        mapping_file = os.path.join(
            base_dir, dataset_name, "data", "metadata", "sample_annotations.tsv"
        )

        # Read in metadata
        metadata = pd.read_csv(mapping_file, header=0, sep="\t", index_col=0)

        selected_metadata = metadata.loc[experiment_id]
        sample_ids = list(selected_metadata[sample_id_colname])

    else:
        # metadata file
        mapping_file = os.path.join(
            base_dir, dataset_name, "data", "metadata", "recount2_metadata.tsv"
        )

        # Read in metadata
        metadata = pd.read_csv(mapping_file, header=0, sep="\t", index_col=0)

        selected_metadata = metadata.loc[experiment_id]
        sample_ids = list(selected_metadata[sample_id_colname])

    return sample_ids


def simulate_compendium(
    num_simulated_experiments,
    normalized_data_file,
    NN_architecture,
    dataset_name,
    analysis_name,
    experiment_ids_file,
    sample_id_colname,
    local_dir,
    base_dir,
):
    """
    Generate simulated data by randomly sampling some number of experiments
    and linearly shifting the gene expression in the VAE latent space,
    preserving the relationship between samples within an experiment.

    Workflow:
    1. Randomly select 1 experiment and get the gene expression data for that
    experiment (here we are assuming that there is only biological variation
    within this experiment)
    2. Encode this experiment into a latent space using the trained VAE model
    3. Encode the entire dataset from the <normalized_data_file>
        3a. Select a random point in the encoded space. For each encoded feature, sample
        from a distribution using the mean and standard deviation for that feature
    4. Calculate the shift_vec_df = centroid(encoded experiment) - random encoded experiment
    5. Shift all the samples from the experiment by the shift_vec_df
    6. Decode the samples
    7. Repeat steps 1-6 for <num_simulated_experiments>

    This will generate a simulated compendium of different gene expression experiments that
    are of a similar type to the original data but with different perturbations

    Arguments
    ----------
    number_simulated_experiments: int
        Number of experiments to simulate

    normalized_data_file: str
        File containing normalized gene expression data

        ------------------------------| PA0001 | PA0002 |...
        05_PA14000-4-2_5-10-07_S2.CEL | 0.8533 | 0.7252 |...
        54375-4-05.CEL                | 0.7789 | 0.7678 |...
        ...                           | ...    | ...    |...

    NN_architecture: str
        Name of neural network architecture to use.
        Format 'NN_<intermediate layer>_<latent layer>'

    dataset_name: str
        Name for analysis directory. Either "Human" or "Pseudomonas"

    analysis_name: str
        Parent directory where simulated data with experiments/partitionings will be stored.
        Format of the directory name is <dataset_name>_<sample/experiment>_lvl_sim

    experiment_ids_file: str
        File containing all cleaned experiment ids

    sample_id_colname: str
        Column header that contains sample id that maps expression data and metadata

    local_dir: str
        Parent directory on local machine to store intermediate results

    base_dir: str
        Root directory containing analysis subdirectories

    Returns
    --------
    simulated dataframe

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
    experiment_ids = pd.read_csv(experiment_ids_file, header=0, sep="\t", index_col=0)

    normalized_data = pd.read_csv(normalized_data_file, header=0, sep="\t", index_col=0)

    print(
        "Normalized gene expression data contains {} samples and {} genes".format(
            normalized_data.shape[0], normalized_data.shape[1]
        )
    )

    # Simulate data

    simulated_data_df = pd.DataFrame()

    for i in range(num_simulated_experiments):

        selected_experiment_id = np.random.choice(
            experiment_ids["experiment_id"], size=1
        )[0]

        # Get corresponding sample ids
        sample_ids = get_sample_ids(
            selected_experiment_id, dataset_name, sample_id_colname
        )

        # Remove any missing sample ids
        sample_ids = list(filter(str.strip, sample_ids))

        # Remove any sample_ids that are not found in gene expression data
        # There are some experiments where most samples have gene expression but a
        # few do not
        sample_ids = [
            sample for sample in sample_ids if sample in normalized_data.index
        ]

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
        data_encoded_all_df = pd.DataFrame(
            data_encoded_all, index=normalized_data.index
        )

        data_encoded_all_df.head()

        # Find a new location in the latent space by sampling from the latent space
        encoded_means = data_encoded_all_df.mean(axis=0)
        encoded_stds = data_encoded_all_df.std(axis=0)

        latent_dim = int(latent_dim)
        new_centroid = np.zeros(latent_dim)

        for j in range(latent_dim):
            new_centroid[j] = np.random.normal(encoded_means[j], encoded_stds[j])

        shift_vec_df = new_centroid - centroid

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

        # Add experiment label
        simulated_data_decoded_df["experiment_id"] = (
            selected_experiment_id + "_" + str(i)
        )

        # Concatenate dataframe per experiment together
        simulated_data_df = pd.concat([simulated_data_df, simulated_data_decoded_df])

    # re-normalize per gene 0-1
    simulated_data_numeric_df = simulated_data_df.drop(
        columns=["experiment_id"], inplace=False
    )

    simulated_data_scaled = preprocessing.MinMaxScaler().fit_transform(
        simulated_data_numeric_df
    )

    simulated_data_scaled_df = pd.DataFrame(
        simulated_data_scaled,
        columns=simulated_data_numeric_df.columns,
        index=simulated_data_numeric_df.index,
    )

    simulated_data_scaled_df["experiment_id"] = simulated_data_df["experiment_id"]

    # If sampling with replacement, then there will be multiple sample ids that are the same
    # therefore we want to reset the index.
    simulated_data_scaled_df.reset_index(drop=True, inplace=True)

    print(
        "Return: simulated gene expression data containing {} samples and {} genes".format(
            simulated_data_scaled_df.shape[0], simulated_data_scaled_df.shape[1]
        )
    )

    return simulated_data_scaled_df


def simulate_data(
    normalized_data_file,
    NN_architecture,
    dataset_name,
    analysis_name,
    num_simulated_samples,
    local_dir,
    base_dir,
):
    """
    Generate simulated data by randomly sampling from VAE latent space.

    Workflow:
    1. Input gene expression data the entire compendium from the <normalized_data_file>
    2. Encode this input into a latent space using the trained VAE model
    3. Randomly sample <num_simulated_samples> samples from the latent space.
        For each encoded feature, sample from a distribution using the
        the mean and standard deviation for that feature
    4. Decode the samples

    This compendium is generated by randomly sampling samples from the
    latent space distribution of the compendium.  All samples are treated equal, where
    association with a specific experiment is ignored.


    Arguments
    ----------
    normalized_data_file: str
        File containing normalized gene expression data

        ------------------------------| PA0001 | PA0002 |...
        05_PA14000-4-2_5-10-07_S2.CEL | 0.8533 | 0.7252 |...
        54375-4-05.CEL                | 0.7789 | 0.7678 |...
        ...                           | ...    | ...    |...

    NN_architecture: str
        Name of neural network architecture to use.
        Format 'NN_<intermediate layer>_<latent layer>'

    dataset_name: str
        Name of analysis directory, Either "Human" or "Pseudomonas"

    analysis_name: str
        Parent directory where simulated data with experiments/partitionings will be stored.
        Format of the directory name is <dataset_name>_<sample/experiment>_lvl_sim

    number_simulated_samples: int
        Number of samples to simulate

    local_dir: str
        Parent directory on local machine to store intermediate results

    base_dir: str
        Root directory containing analysis subdirectories

    Returns
    --------
    simulated dataframe

    """

    # Files
    NN_dir = os.path.join(base_dir, dataset_name, "models", NN_architecture)
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

    print(
        "Normalized gene expression data contains {} samples and {} genes".format(
            normalized_data.shape[0], normalized_data.shape[1]
        )
    )

    # Simulate data

    # Encode into latent space
    data_encoded = loaded_model.predict_on_batch(normalized_data)
    data_encoded_df = pd.DataFrame(data_encoded, index=normalized_data.index)

    latent_dim = data_encoded_df.shape[1]

    # Get mean and standard deviation per encoded feature
    encoded_means = data_encoded_df.mean(axis=0)
    encoded_stds = data_encoded_df.std(axis=0)

    # Generate samples
    new_data = np.zeros([num_simulated_samples, latent_dim])
    for j in range(latent_dim):
        # Use mean and std for feature
        new_data[:, j] = np.random.normal(
            encoded_means[j], encoded_stds[j], num_simulated_samples
        )

        # Use standard normal
        # new_data[:,j] = np.random.normal(0, 1, num_simulated_samples)

    new_data_df = pd.DataFrame(data=new_data)

    # Decode samples
    new_data_decoded = loaded_decode_model.predict_on_batch(new_data_df)
    simulated_data = pd.DataFrame(data=new_data_decoded)

    print(
        "Return: simulated gene expression data containing {} samples and {} genes".format(
            simulated_data.shape[0], simulated_data.shape[1]
        )
    )

    return simulated_data


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
    loaded_model = load_model(model_encoder_file, compile=False)
    loaded_decode_model = load_model(model_decoder_file, compile=False)

    loaded_model.load_weights(weights_encoder_file)
    loaded_decode_model.load_weights(weights_decoder_file)

    # Read data
    normalized_data = pd.read_csv(normalized_data_file, header=0, sep="\t", index_col=0)

    # Get corresponding sample ids
    sample_ids = get_sample_ids(selected_experiment_id, dataset_name)

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

    out_encoded_file = os.path.join(
        local_dir,
        "pseudo_experiment",
        f"selected_simulated_encoded_data_{selected_experiment_id}_{run}.txt",
    )

    simulated_data_encoded_df.to_csv(out_encoded_file, float_format="%.3f", sep="\t")
