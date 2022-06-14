"""
Author: Alexandra Lee
Date Created: 30 August 2019

These scripts generate simulated compendia using the low-dimensional
representation of the gene expressiond data, created by training the
VAE on gene expression data.
"""

import os
import pickle
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


def get_sample_ids(
    metadata_filename, delimiter, experiment_colname, experiment_id, sample_id_colname
):
    """
    Returns sample ids (found in gene expression df) associated with
    a given list of experiment ids (found in the metadata)

    Arguments
    ----------
    metadata_filename: str
        Metadata file path. An example metadata file can be found
        here: https://github.com/greenelab/ponyo/blob/master/human_tests/data/metadata/recount2_metadata.tsv

    delimiter: str
        Delimiter for metadata file

    experiment_colname: str
        Column header that contains the experiment ids

    experiment_id: str
        Experiment id selected to retrieve sample ids for

    sample_id_colname: str
        Column header that contains sample id that maps expression data
        and metadata

    """

    # Read in metadata
    metadata = pd.read_csv(metadata_filename, header=0, sep=delimiter, index_col=None)

    # Set index column to experiment id column
    metadata.set_index(experiment_colname, inplace=True)

    # Select samples associated with experiment id
    selected_metadata = metadata.loc[experiment_id]
    sample_ids = list(selected_metadata[sample_id_colname])

    return sample_ids


def simulate_by_random_sampling(
    normalized_data_filename,
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
    normalized_data_filename: str
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
    model_encoder_filename = glob.glob(os.path.join(NN_dir, "*_encoder_model.h5"))[0]

    weights_encoder_filename = glob.glob(os.path.join(NN_dir, "*_encoder_weights.h5"))[
        0
    ]

    model_decoder_filename = glob.glob(os.path.join(NN_dir, "*_decoder_model.h5"))[0]

    weights_decoder_filename = glob.glob(os.path.join(NN_dir, "*_decoder_weights.h5"))[
        0
    ]

    # Load saved models
    loaded_model = load_model(model_encoder_filename)
    loaded_decode_model = load_model(model_decoder_filename)

    loaded_model.load_weights(weights_encoder_filename)
    loaded_decode_model.load_weights(weights_decoder_filename)

    # Read data
    normalized_data = pd.read_csv(
        normalized_data_filename, header=0, sep="\t", index_col=0
    )

    print(
        "Normalized gene expression data contains {} samples and {} genes".format(
            normalized_data.shape[0], normalized_data.shape[1]
        )
    )

    simulated_data = run_sample_simulation(
        loaded_model, loaded_decode_model, normalized_data, num_simulated_samples
    )

    return simulated_data


def run_sample_simulation(encoder, decoder, normalized_data, num_simulated_samples):
    """
    This function does the actual simulation work for simulate_by_random_sampling.
    To be more precise, it uses a VAE to simulate data based on the distribution of
    `normalized_data`.

    Arguments
    ----------
    encoder: keras.models.Model
        The encoder half of the VAE. `encoder` takes in a (samples x genes) dataframe of
        gene expression data and encodes it into a latent space

    decoder: keras.models.Model
        The decoder half of the VAE. `decoder` takes a dataframe of means and standard deviations
        and uses them to simulate gene expression data close to the distribution of normalized_data

    normalized_data: pd.DataFrame
        The data to be used to train the VAE

    num_simulated_samples: int
        The number of samples to simulate

    Returns
    --------
    simulated_data: pd.DataFrame
        The data simulated from the autoencoder

    """
    # Simulate data

    # Encode into latent space
    data_encoded = encoder.predict_on_batch(normalized_data)
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
    new_data_decoded = decoder.predict_on_batch(new_data_df)
    simulated_data = pd.DataFrame(data=new_data_decoded)

    print(
        "Return: simulated gene expression data containing {} samples and {} genes".format(
            simulated_data.shape[0], simulated_data.shape[1]
        )
    )

    return simulated_data


def simulate_by_latent_transformation(
    num_simulated_experiments,
    normalized_data_filename,
    NN_architecture,
    latent_dim,
    dataset_name,
    analysis_name,
    metadata_filename,
    metadata_delimiter,
    experiment_id_colname,
    sample_id_colname,
    experiment_ids_filename,
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
    num_simulated_experiments: int
        Number of experiments to simulate

    normalized_data_filename: str
        File containing normalized gene expression data

        ------------------------------| PA0001 | PA0002 |...
        05_PA14000-4-2_5-10-07_S2.CEL | 0.8533 | 0.7252 |...
        54375-4-05.CEL                | 0.7789 | 0.7678 |...
        ...                           | ...    | ...    |...

    NN_architecture: str
        Name of neural network architecture to use.
        Format 'NN_<intermediate layer>_<latent layer>'
    
    latent_dim: int
        The number of dimensions in the latent space

    dataset_name: str
        Name for analysis directory. Either "Human" or "Pseudomonas"

    analysis_name: str
        Parent directory where simulated data with experiments/partitionings will be stored.
        Format of the directory name is <dataset_name>_<sample/experiment>_lvl_sim

    metadata_filename: str
        Metadata file path. Note: The format of this metadata file
        requires the index column to contain experiment ids.

    metadata_delimiter: str
        Delimiter for metadata file

    experiment_colname: str
        Column header that contains the experiment ids

    sample_id_colname: str
        Column header that contains sample id that maps expression data
        and metadata

    experiment_ids_filename: str
        File containing all cleaned experiment ids to retrieve sample
        ids for

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

    model_encoder_filename = glob.glob(os.path.join(NN_dir, "*_encoder_model.h5"))[0]

    weights_encoder_filename = glob.glob(os.path.join(NN_dir, "*_encoder_weights.h5"))[
        0
    ]

    model_decoder_filename = glob.glob(os.path.join(NN_dir, "*_decoder_model.h5"))[0]

    weights_decoder_filename = glob.glob(os.path.join(NN_dir, "*_decoder_weights.h5"))[
        0
    ]

    # Load saved models
    loaded_model = load_model(model_encoder_filename)
    loaded_decode_model = load_model(model_decoder_filename)

    loaded_model.load_weights(weights_encoder_filename)
    loaded_decode_model.load_weights(weights_decoder_filename)

    # Read data
    experiment_ids = pd.read_csv(
        experiment_ids_filename, header=0, sep="\t", index_col=0
    )

    normalized_data = pd.read_csv(
        normalized_data_filename, header=0, sep="\t", index_col=0
    )

    print(
        "Normalized gene expression data contains {} samples and {} genes".format(
            normalized_data.shape[0], normalized_data.shape[1]
        )
    )

    # Simulate data
    simulation_results = run_latent_transformation_simulation(
        loaded_model,
        loaded_decode_model,
        normalized_data,
        experiment_ids,
        metadata_filename,
        metadata_delimiter,
        experiment_id_colname,
        sample_id_colname,
        num_simulated_experiments,
        latent_dim,
    )
    (
        simulated_data_scaled_df,
        simulated_data_encoded_df,
        data_encoded_df,
    ) = simulation_results

    # Save before and after experiment for visualization validation
    before_encoded_filename = os.path.join(local_dir, "simulated_before_encoded.txt")
    after_encoded_filename = os.path.join(local_dir, "simulated_after_encoded.txt")

    data_encoded_df.to_csv(before_encoded_filename, float_format="%.3f", sep="\t")
    simulated_data_encoded_df.to_csv(
        after_encoded_filename, float_format="%.3f", sep="\t"
    )

    return simulated_data_scaled_df


def run_latent_transformation_simulation(
    encoder,
    decoder,
    normalized_data,
    experiment_ids,
    metadata_filename,
    metadata_delimiter,
    experiment_id_colname,
    sample_id_colname,
    num_simulated_experiments,
    latent_dim,
):

    """
    This function handles the simulation logic used in `simulate_by_latent_transformation`

    Arguments
    ---------
    encoder: keras.models.Model
        The encoder half of the VAE. `encoder` takes in a (samples x genes) dataframe of
        gene expression data and encodes it into a latent space

    decoder: keras.models.Model
        The decoder half of the VAE. `decoder` takes a dataframe of means and standard deviations
        and uses them to simulate gene expression data close to the distribution of a
        a set of experiments from normalized_data

    normalized_data: pd.DataFrame
        The data to be used to train the VAE

    experiment_ids: pd.DataFrame
        The set of ids for experiments present in normalized_data

    metadata_filename: str
        Metadata file path. Note: The format of this metadata file
        requires the index column to contain experiment ids.

    metadata_delimiter: str
        Delimiter for metadata file

    experiment_colname: str
        Column header that contains the experiment ids

    sample_id_colname: str
        Column header that contains sample id that maps expression data
        and metadata

    num_simulated_experiments: int
        The number of experiments to simulate

    latent_dim: int
        The number of dimensions in the latent space

    Returns
    -------
    simulated_data_scaled_df: pd.DataFrame
        The simulated data rescaled to have a max of 1 and a min of zero

    simulated_data_encoded_df: pd.DataFrame
        The raw data created by taking an experiment from the data and shifting its
        centroid to elsewhere in the latent space

    data_encoded_df: pd.DataFrame
        The results of shifting `normalized_data` into the latent space specified by `encoder`
    """

    simulated_data_df = pd.DataFrame()

    for i in range(num_simulated_experiments):

        selected_experiment_id = np.random.choice(
            experiment_ids["experiment_id"], size=1
        )[0]

        # Get corresponding sample ids
        sample_ids = get_sample_ids(
            metadata_filename,
            metadata_delimiter,
            experiment_id_colname,
            selected_experiment_id,
            sample_id_colname,
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
        data_encoded = encoder.predict_on_batch(selected_data_df)
        data_encoded_df = pd.DataFrame(data_encoded, index=selected_data_df.index)

        # Get centroid of original data
        centroid = data_encoded_df.mean(axis=0)

        # Encode original gene expression data into latent space
        data_encoded_all = encoder.predict_on_batch(normalized_data)
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
        simulated_data_decoded = decoder.predict_on_batch(simulated_data_encoded_df)

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

    return simulated_data_scaled_df, simulated_data_encoded_df, data_encoded_df


def shift_template_experiment(
    normalized_data_filename,
    NN_architecture,
    latent_dim,
    dataset_name,
    scaler,
    metadata_filename,
    metadata_delimiter,
    experiment_id_colname,
    sample_id_colname,
    selected_experiment_id,
    local_dir,
    base_dir,
    num_runs,
):
    """
    Generate new simulated experiment using the selected_experiment_id as a template
    experiment using the same workflow as `simulate_by_latent_transform`

    This will return a file with a single simulated experiment following the workflow mentioned.
    This function can be run multiple times to generate multiple simulated experiments from a
    single selected_experiment_id.

    Arguments
    ----------
    normalized_data_filename: str
        File containing normalized gene expression data

        ------------------------------| PA0001 | PA0002 |...
        05_PA14000-4-2_5-10-07_S2.CEL | 0.8533 | 0.7252 |...
        54375-4-05.CEL                | 0.7789 | 0.7678 |...
        ...                           | ...    | ...    |...

    NN_architecture: str
        Name of neural network architecture to use.
        Format 'NN_<intermediate layer>_<latent layer>'

    latent_dim: int
        The number of dimensions in the latent space

    dataset_name: str
        Name for analysis directory. Either "Human" or "Pseudomonas"

    scaler: minmax model
        Model used to transform data into a different range

    metadata_filename: str
        Metadata file path. Note: The format of this metadata file
        requires the index column to contain experiment ids.

    metadata_delimiter: str
        Delimiter for metadata file

    experiment_colname: str
        Column header that contains the experiment ids

    sample_id_colname: str
        Column header that contains sample id that maps expression data
        and metadata

    selected_experiment_id: str
        Experiment id selected as template

    local_dir: str
        Parent directory on local machine to store intermediate results

    base_dir: str
        Root directory containing analysis subdirectories

    num_runs: int
        Number of experiments to simulate

    Returns
    --------
    simulated_data_filename: str
        File containing simulated gene expression data

    """

    # Files
    NN_dir = os.path.join(base_dir, dataset_name, "models", NN_architecture)

    model_encoder_filename = glob.glob(os.path.join(NN_dir, "*_encoder_model.h5"))[0]

    weights_encoder_filename = glob.glob(os.path.join(NN_dir, "*_encoder_weights.h5"))[
        0
    ]

    model_decoder_filename = glob.glob(os.path.join(NN_dir, "*_decoder_model.h5"))[0]

    weights_decoder_filename = glob.glob(os.path.join(NN_dir, "*_decoder_weights.h5"))[
        0
    ]

    # Load saved models
    loaded_model = load_model(model_encoder_filename, compile=False)
    loaded_decode_model = load_model(model_decoder_filename, compile=False)

    loaded_model.load_weights(weights_encoder_filename)
    loaded_decode_model.load_weights(weights_decoder_filename)

    # Read data
    normalized_data = pd.read_csv(
        normalized_data_filename, header=0, sep="\t", index_col=0
    )

    # Get corresponding sample ids
    sample_ids = get_sample_ids(
        metadata_filename,
        metadata_delimiter,
        experiment_id_colname,
        selected_experiment_id,
        sample_id_colname,
    )

    # Gene expression data for selected samples
    selected_data_df = normalized_data.loc[sample_ids]

    for run in range(num_runs):
        simulated_data_decoded_df, simulated_data_encoded_df = run_shift_template(
            loaded_model, loaded_decode_model, normalized_data, selected_data_df, latent_dim
        )

        # Un-normalize the data in order to run DE analysis downstream
        simulated_data_scaled = scaler.inverse_transform(simulated_data_decoded_df)

        simulated_data_scaled_df = pd.DataFrame(
            simulated_data_scaled,
            columns=simulated_data_decoded_df.columns,
            index=simulated_data_decoded_df.index,
        )

        # Save
        out_filename = os.path.join(
            local_dir,
            "pseudo_experiment",
            "selected_simulated_data_" + selected_experiment_id + "_" + str(run) + ".txt",
        )

        simulated_data_scaled_df.to_csv(out_filename, float_format="%.3f", sep="\t")

        out_encoded_filename = os.path.join(
            local_dir,
            "pseudo_experiment",
            f"selected_simulated_encoded_data_{selected_experiment_id}_{run}.txt",
        )

        simulated_data_encoded_df.to_csv(
            out_encoded_filename, float_format="%.3f", sep="\t"
        )

    # Save template data for visualization validation
    test_filename = os.path.join(
        local_dir,
        "pseudo_experiment",
        "template_normalized_data_" + selected_experiment_id + "_test.txt",
    )
    selected_data_df.to_csv(test_filename, float_format="%.3f", sep="\t")


def run_shift_template(encoder, decoder, normalized_data, selected_data_df, latent_dim):
    """
    This function does the template shifting used in `shift_template_experiment`.

    Arguments
    ---------
    encoder: keras.models.Model
        The encoder half of the VAE. `encoder` takes in a (samples x genes) dataframe of
        gene expression data and encodes it into a latent space

    decoder: keras.models.Model
        The decoder half of the VAE. `decoder` takes a dataframe of means and standard deviations
        and uses them to simulate gene expression data close to the distribution of normalized_data

    normalized_data: pd.DataFrame
        The data to be used to train the VAE

    selected_data_df: pd.DataFrame
        The samples to be shifted in the latent space

    latent_dim: int
        The dimension of the latent space the samples will be shifted in

    Returns
    -------
    simulated_data_decoded_df: pd.DataFrame
        The simulated data created by shifting the samples in the latent space

    simulated_data_encoded_df: pd.DataFrame
        The latent means and standard deviations in the latent space used to simulate the data
    """
    # Encode selected experiment into latent space
    data_encoded = encoder.predict_on_batch(selected_data_df)
    data_encoded_df = pd.DataFrame(data_encoded, index=selected_data_df.index)

    # Get centroid of original data
    centroid = data_encoded_df.mean(axis=0)

    # Add individual vectors(centroid, sample point) to new_centroid

    # Encode original gene expression data into latent space
    data_encoded_all = encoder.predict_on_batch(normalized_data)
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

    simulated_data_encoded_df = data_encoded_df.apply(
        lambda x: x + shift_vec_df, axis=1
    )

    # Decode simulated data into raw gene space
    simulated_data_decoded = decoder.predict_on_batch(simulated_data_encoded_df)

    simulated_data_decoded_df = pd.DataFrame(
        simulated_data_decoded,
        index=simulated_data_encoded_df.index,
        columns=selected_data_df.columns,
    )

    return simulated_data_decoded_df, simulated_data_encoded_df

def compare_match_features(template_filename, compendium_filename):
    """
    This function checks that the feature space matches between
    template experiment and the VAE model. 
    In other words, this function checks that the gene ids (column names)
    are the same in the template experiment and the VAE model. 
    
    If there are differences this function does the following:
    If a gene is present in template experiment but not in the VAE model, then drop gene
    If a gene is present in VAE model but not in the template experiment, 
    then add gene to template experiment with median gene expression value from the VAE model.
    
    template_filename: str
        File containing template gene expression data. Expect matrix of dimension: sample x gene
        
    compendium_filename: str
        File containing un-normalized compendium gene expression data. 
        
    """
    # Read template experiment
    template_experiment = pd.read_csv(
        template_filename, sep="\t", index_col=0, header=0
    )

    print(template_experiment.shape)

    # Read compendium
    compendium = pd.read_csv(compendium_filename, sep="\t", index_col=0, header=0)

    print(compendium.shape)

    # Check if genes are shared:
    template_genes = template_experiment.columns
    compendium_genes = compendium.columns

    # If a gene is present in template experiment but not in the VAE model, then drop gene
    # If a gene is present in VAE model but not in the template experiment,
    # then add gene to template experiment with median gene expression value
    only_template_genes = list(set(template_genes).difference(compendium_genes))
    only_compendium_genes = list(set(compendium_genes).difference(template_genes))

    tmp_template_experiment = template_experiment.drop(columns=only_template_genes)

    # Get median gene expression for only_compendium_genes
    # Use mapped_compendium for this to get counts
    median_gene_expression = compendium[only_compendium_genes].median().to_dict()
    tmp2_template_experiment = tmp_template_experiment.assign(**median_gene_expression)

    assert len(tmp2_template_experiment.columns) == len(compendium.columns)

    # sort template experiment columns to be in the same order as the compendium
    mapped_template_experiment = tmp2_template_experiment[compendium.columns]

    mapped_template_experiment.to_csv(template_filename, sep="\t")

    return mapped_template_experiment


def normalize_template_experiment(mapped_template_experiment, scaler_filename):
    """
    This function normalizes the template experiment to be within
    0-1 range, using the same scaler transform that was used to
    0-1 scale the training compendium.

    mapped_template_experiment: df
        Dataframe of template experiment after mapping gene ids        
    scaler_filename: str
        Filename containing picked scaler transform used to normalize compendium data
    """
    # Load pickled file
    with open(scaler_filename, "rb") as scaler_fh:
        scaler = pickle.load(scaler_fh)

    processed_template_experiment = scaler.transform(mapped_template_experiment)

    processed_template_experiment_df = pd.DataFrame(
        processed_template_experiment,
        columns=mapped_template_experiment.columns,
        index=mapped_template_experiment.index,
    )

    return processed_template_experiment_df


def process_template_experiment(
    template_filename,
    compendium_filename,
    scaler_filename,
    mapped_template_filename,
    processed_template_filename,
):
    """
    This function processes the template experiment to prepare for
    simulating new data. Specifically this function does the following:
    
    1. Compares and maps the template feature space to the compendium 
    feature space using `compare_match_features()`
    2. Normalizes the template experiment to be in the same scale
    as the compendium dataset using `normalize_template_experiment()`

    Arguments
    ----------
    template_filename: str
        File containing template gene expression data. Expect matrix of dimension: sample (rows) x gene (columns)
    compendium_filename: str
        File containing un-normalized compendium gene expression data. 
        Gene ids are either using PA#### (P. aeruginosa)
        or using HGNC symbols (Human)
    scaler_filename: str
        Filename containing pickled scaler transform used to normalize compendium data
    mapped_filename: str
        Filename containing the template data where genes are mapped to compendium data.
    processed_filename: str
        Filename containing the template normalized data. This data can now be
        encoded into the learned latent space.
    """

    # Compare and map genes from template experiment to
    # compendium dataset
    mapped_template_experiment = compare_match_features(
        template_filename, compendium_filename
    )

    normalized_template_experiment = normalize_template_experiment(
        mapped_template_experiment, scaler_filename
    )

    # Save
    mapped_template_experiment.to_csv(mapped_template_filename, sep="\t")
    normalized_template_experiment.to_csv(processed_template_filename, sep="\t")


def embed_shift_template_experiment(
    normalized_compendium_filename,
    normalized_template_filename,
    vae_model_dir,
    selected_experiment_id,
    scaler_filename,
    local_dir,
    latent_dim,
    num_runs,
):
    """
    Generate new simulated experiment using the selected_experiment_id as a template
    experiment and linearly shift template experiment to different locations of the
    latent space to create new experiment. This workflow is similar to `simulate_by_latent_transform`

    This will return a file with a single simulated experiment following the workflow mentioned.
    This function can be run multiple times to generate multiple simulated experiments from a
    single selected_experiment_id.

    Arguments
    ----------
    normalized_compendium filename: str
        File containing normalized gene expression data

        ------------------------------| PA0001 | PA0002 |...
        05_PA14000-4-2_5-10-07_S2.CEL | 0.8533 | 0.7252 |...
        54375-4-05.CEL                | 0.7789 | 0.7678 |...
        ...                           | ...    | ...    |...

    scaler: minmax model
        Model used to transform data into a different range

    local_dir: str
        Parent directory on local machine to store intermediate results

    base_dir: str
        Root directory containing analysis subdirectories

    num_runs: int
        Number of simulated experiments

    Returns
    --------
    simulated_data_filename: str
        File containing simulated gene expression data

    """

    # Files
    NN_dir = vae_model_dir

    model_encoder_filename = glob.glob(os.path.join(NN_dir, "*_encoder_model.h5"))[0]

    weights_encoder_filename = glob.glob(os.path.join(NN_dir, "*_encoder_weights.h5"))[
        0
    ]

    model_decoder_filename = glob.glob(os.path.join(NN_dir, "*_decoder_model.h5"))[0]

    weights_decoder_filename = glob.glob(os.path.join(NN_dir, "*_decoder_weights.h5"))[
        0
    ]

    # Read data
    normalized_compendium = pd.read_csv(
        normalized_compendium_filename, header=0, sep="\t", index_col=0
    )
    normalized_template = pd.read_csv(
        normalized_template_filename, header=0, sep="\t", index_col=0
    )

    # Load pickled file
    with open(scaler_filename, "rb") as scaler_fh:
        scaler = pickle.load(scaler_fh)

    # Load saved models
    loaded_model = load_model(model_encoder_filename, compile=False)
    loaded_decode_model = load_model(model_decoder_filename, compile=False)

    loaded_model.load_weights(weights_encoder_filename)
    loaded_decode_model.load_weights(weights_decoder_filename)

    # Gene expression data for template experiment
    selected_data_df = normalized_template

    for run in range(num_runs):
        simulated_data_decoded_df, simulated_data_encoded_df = run_embed_shift_template_experiment(
                loaded_model, loaded_decode_model, normalized_compendium, selected_data_df, latent_dim
            )

        # Un-normalize the data in order to run DE analysis downstream
        simulated_data_scaled = scaler.inverse_transform(simulated_data_decoded_df)

        simulated_data_scaled_df = pd.DataFrame(
            simulated_data_scaled,
            columns=simulated_data_decoded_df.columns,
            index=simulated_data_decoded_df.index,
        )

        # Save
        out_filename = os.path.join(
            local_dir,
            "pseudo_experiment",
            "selected_simulated_data_" + selected_experiment_id + "_" + str(run) + ".txt",
        )

        simulated_data_scaled_df.to_csv(out_filename, float_format="%.3f", sep="\t")

        out_encoded_filename = os.path.join(
            local_dir,
            "pseudo_experiment",
            f"selected_simulated_encoded_data_{selected_experiment_id}_{run}.txt",
        )

        simulated_data_encoded_df.to_csv(
            out_encoded_filename, float_format="%.3f", sep="\t"
        )
    
    # Save template data for visualization validation
        test_filename = os.path.join(
            local_dir,
            "pseudo_experiment",
            "template_normalized_data_" + selected_experiment_id + "_test.txt",
        )

        selected_data_df.to_csv(test_filename, float_format="%.3f", sep="\t")

def run_embed_shift_template_experiment(encoder, decoder, normalized_data, selected_data_df, latent_dim):
    """
    This function does the template shifting used in `embed_shift_template_experiment`.
    Arguments
    ---------
    encoder: keras.models.Model
        The encoder half of the VAE. `encoder` takes in a (samples x genes) dataframe of
        gene expression data and encodes it into a latent space
    decoder: keras.models.Model
        The decoder half of the VAE. `decoder` takes a dataframe of means and standard deviations
        and uses them to simulate gene expression data close to the distribution of normalized_data
    normalized_data: pd.DataFrame
        The data to be used to train the VAE
    selected_data_df: pd.DataFrame
        The samples to be shifted in the latent space
    latent_dim: int
        The dimension of the latent space the samples will be shifted in
    Returns
    -------
    simulated_data_decoded_df: pd.DataFrame
        The simulated data created by shifting the samples in the latent space
    simulated_data_encoded_df: pd.DataFrame
        The latent means and standard deviations in the latent space used to simulate the data
    """

    # Encode selected experiment into latent space
    data_encoded = encoder.predict_on_batch(selected_data_df)
    data_encoded_df = pd.DataFrame(data_encoded, index=selected_data_df.index)

    # Get centroid of original data
    centroid = data_encoded_df.mean(axis=0)

    # Add individual vectors(centroid, sample point) to new_centroid

    # Encode original gene expression data into latent space
    data_encoded_all = encoder.predict_on_batch(normalized_data)
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
    simulated_data_decoded = decoder.predict_on_batch(
        simulated_data_encoded_df
    )

    simulated_data_decoded_df = pd.DataFrame(
        simulated_data_decoded,
        index=simulated_data_encoded_df.index,
        columns=selected_data_df.columns,
    )

    return simulated_data_decoded_df, simulated_data_encoded_df
