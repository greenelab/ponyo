"""
Author: Alexandra Lee
Date Created: 30 August 2019

These scripts are the components used to run each simulation experiment,
found in `simulations.py`.
These scripts generate simulated compendia, add noise to simulated data,
apply noise correction to simulated data, permute simulated data.
"""

import os
import pandas as pd
import numpy as np
import random
import glob
import warnings
from keras.models import load_model
from sklearn import preprocessing

from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

limma = importr("limma")
sva = importr("sva")
pandas2ri.activate()

def fxn(): 
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():


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
        Column header that contains sample id that maps expression data and metadata

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
    and linearly shifting the gene expression in the VAE latent space, preserving
    the relationship between samples within an experiment.

    Workflow:
    1. Randomly select 1 experiment and get the gene expression data for that experiment
    (here we are assuming that there is only biological variation within this experiment)
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
        # There are some experiments where most samples have gene expression but a few do not
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


def permute_data(simulated_data):
    """
    Permute the simulated data

    Arguments
    ----------
    simulated_data: df
        Dataframe containing simulated gene expression data

    Returns
    --------
    permuted simulated dataframe. This data will be used as a 
    negative control in similarity analysis.
    """

    if "experiment_id" in list(simulated_data.columns):
        simulated_data_tmp = simulated_data.drop(columns="experiment_id", inplace=False)
    else:
        simulated_data_tmp = simulated_data.copy()

    # Shuffle values within each sample (row)
    # Each sample treated independently
    shuffled_simulated_arr = []
    num_samples = simulated_data.shape[0]

    for i in range(num_samples):
        row = list(simulated_data_tmp.values[i])
        shuffled_simulated_row = random.sample(row, len(row))
        shuffled_simulated_arr.append(shuffled_simulated_row)

    shuffled_simulated_data = pd.DataFrame(
        shuffled_simulated_arr,
        index=simulated_data_tmp.index,
        columns=simulated_data_tmp.columns,
    )

    return shuffled_simulated_data


def add_experiments_io(
    simulated_data, num_experiments, run, local_dir, dataset_name, analysis_name
):
    """
    Say we are interested in identifying genes that differentiate between
    disease vs normal states. However our dataset includes samples from
    different labs or protocols and there are variations
    in gene expression that are due to these other conditions
    that do not have to do with disease state.

    These non-relevant variations in the data are called technical variations
    that we want to model.  To model technical variation in our simulated data
    we will do the following:

    1. Partition our simulated data into <num_experiments> 
    2. For each partition we will shift all genes using a vector of values
    sampled from a gaussian distribution centered around 0. This noise represents
    noise shared acoss the samples in the partition.
    3. Repeat this for each partition
    4. Append all shifted partitions together

    Arguments
    ----------
    simulated_data: df
        Dataframe containing simulated gene expression data

    num_experiments: list
        List of different numbers of experiments to add to
        simulated data

    run: int
        Unique core identifier that is used to create unique filenames for intermediate files

    local_dir: str
        Parent directory on local machine to store intermediate results

    dataset_name: str
        Name of analysis directory. Either "Human" or "Pseudomonas"

    analysis_name: str
        Parent directory where simulated data with experiments/partitionings are be stored.
        Format of the directory name is <dataset>_<sample/experiment>_lvl_sim 

    Output
    --------
    Files of simulated data with different numbers of experiments added are save to file.
    Each file is named as "Experiment_<number of experiments added>"
    """
    analysis_dir = os.path.join(
        local_dir, "experiment_simulated", dataset_name + "_" + analysis_name
    )

    if os.path.exists(analysis_dir) == False:
        print("Creating new directory: \n {}".format(analysis_dir))
        os.makedirs(analysis_dir, exist_ok=True)

    # Add batch effects
    num_simulated_samples = simulated_data.shape[0]
    num_genes = simulated_data.shape[1]

    # Create an array of the simulated data indices
    simulated_ind = np.array(simulated_data.index)

    for i in num_experiments:
        print("Creating simulated data with {} experiments..".format(i))

        experiment_file = os.path.join(
            analysis_dir, "Experiment_" + str(i) + "_" + str(run) + ".txt.xz"
        )

        experiment_map_file = os.path.join(
            analysis_dir, "Experiment_map_" + str(i) + "_" + str(run) + ".txt.xz"
        )

        # Create dataframe with grouping
        experiment_data_map = simulated_data.copy()

        if i == 1:
            simulated_data.to_csv(experiment_file, sep="\t", compression="xz")

            # Add experiment id to map dataframe
            experiment_data_map["experiment"] = str(i)
            experiment_data_map_df = pd.DataFrame(
                data=experiment_data_map["experiment"], index=simulated_ind.sort()
            )

            experiment_data_map_df.to_csv(
                experiment_map_file, sep="\t", compression="xz"
            )

        else:
            experiment_data = simulated_data.copy()

            # Shuffle indices
            np.random.shuffle(simulated_ind)

            # Partition indices to batch
            # Note: 'array_split' will chunk data into almost equal sized chunks.
            # Returns arrays of size N % i and one array with the remainder
            partition = np.array_split(simulated_ind, i)

            for j in range(i):
                # Scalar to shift gene expressiond data
                stretch_factor = np.random.normal(0.0, 0.2, [1, num_genes])

                # Tile stretch_factor to be able to add to batches
                num_samples_per_experiment = len(partition[j])
                stretch_factor_tile = pd.DataFrame(
                    pd.np.tile(stretch_factor, (num_samples_per_experiment, 1)),
                    index=experiment_data.loc[partition[j].tolist()].index,
                    columns=experiment_data.loc[partition[j].tolist()].columns,
                )

                # Add experiments
                experiment_data.loc[partition[j].tolist()] = (
                    experiment_data.loc[partition[j].tolist()] + stretch_factor_tile
                )

                # Add experiment id to map dataframe
                experiment_data_map.loc[partition[j], "experiment"] = str(j)

            experiment_data_map_df = pd.DataFrame(
                data=experiment_data_map["experiment"], index=simulated_ind.sort()
            )

            # Save
            experiment_data.to_csv(
                experiment_file, float_format="%.3f", sep="\t", compression="xz"
            )

            experiment_data_map_df.to_csv(
                experiment_map_file, sep="\t", compression="xz"
            )


def add_experiments_grped_io(
    simulated_data, num_partitions, run, local_dir, dataset_name, analysis_name
):
    """
    Similar to `add_experiments_io` we will model technical variation in our
    simulated data. In this case, we will keep track of which samples
    are associated with an experiment.

    To do this we will:
    1. Partition our simulated data into <num_partitions>
        Here we are keeping track of experiment id and partitioning
        such that all samples from an experiment are in the same
        partition.

        Note: Partition sizes will be different since experiment
        sizes are different per experiment.
    2. For each partition we will shift all genes using a vector of values
    sampled from a gaussian distribution centered around 0.
    3. Repeat this for each partition
    4. Append all partitions together

    This function will return the files with compendia with different numbers
    of technical variation added with one file per compendia.

    Arguments
    ----------
    simulated_data_file: str
        File containing simulated gene expression data

    num_partitions: list
        List of different numbers of partitions to add
        technical variations to

    run: int
        Unique core identifier that is used to create unique filenames for intermediate files

    local_dir: str
        Parent directory on local machine to store intermediate results

    dataset_name: str
        Name of analysis directory. Either "Human" or "Pseudomonas"

    analysis_name: str
        Parent directory where simulated data with experiments/partitionings are be stored.
        Format of the directory name is <dataset>_<sample/experiment>_lvl_sim 


    Output
    --------
    Files of simulated data with different numbers of experiments added are saved to file.
    Each file is named as "Experiment_<number of experiments added>"
    """

    analysis_dir = os.path.join(
        local_dir, "partition_simulated", dataset_name + "_" + analysis_name
    )

    if os.path.exists(analysis_dir) == False:
        print("Creating new directory: \n {}".format(analysis_dir))
        os.makedirs(analysis_dir, exist_ok=True)

    # Add batch effects
    num_genes = simulated_data.shape[1] - 1

    # Create an array of the simulated data indices
    simulated_ind = np.array(simulated_data.index)

    for i in num_partitions:
        print("Creating simulated data with {} partitions..".format(i))

        partition_file = os.path.join(
            analysis_dir, "Partition_" + str(i) + "_" + str(run) + ".txt.xz"
        )

        partition_map_file = os.path.join(
            analysis_dir, "Partition_map_" + str(i) + "_" + str(run) + ".txt.xz"
        )

        # Create dataframe with grouping
        partition_data_map = simulated_data.copy()

        if i == 1:
            simulated_data_out = simulated_data.drop(columns="experiment_id")
            simulated_data_out.to_csv(partition_file, sep="\t", compression="xz")

            # Add experiment id to map dataframe
            partition_data_map["partition"] = str(i)

            partition_data_map_df = pd.DataFrame(
                data=partition_data_map["partition"], index=simulated_ind.sort()
            )

            partition_data_map_df.to_csv(partition_map_file, sep="\t", compression="xz")

        else:
            partition_data = simulated_data.copy()

            # Shuffle experiment ids
            experiment_ids = simulated_data["experiment_id"].unique()
            np.random.shuffle(experiment_ids)

            # Partition experiment ids
            # Note: 'array_split' will chunk data into almost equal sized chunks.
            # Returns arrays of size N % i and one array with the remainder
            partition = np.array_split(experiment_ids, i)

            for j in range(i):
                # Randomly select experiment ids
                selected_experiment_ids = partition[j]

                # Get sample ids associated with experiment ids
                sample_ids = list(
                    simulated_data[
                        simulated_data["experiment_id"].isin(partition[j])
                    ].index
                )

                # Scalar to shift gene expressiond data
                stretch_factor = np.random.normal(0.0, 0.2, [1, num_genes])

                # Tile stretch_factor to be able to add to batches
                num_samples_per_partition = len(sample_ids)

                if j == 0:
                    # Drop experiment_id label to do calculation
                    partition_data.drop(columns="experiment_id", inplace=True)

                stretch_factor_tile = pd.DataFrame(
                    pd.np.tile(stretch_factor, (num_samples_per_partition, 1)),
                    index=partition_data.loc[sample_ids].index,
                    columns=partition_data.loc[sample_ids].columns,
                )

                # Add noise to partition
                partition_data.loc[sample_ids] = (
                    partition_data.loc[sample_ids] + stretch_factor_tile
                )

                # Add partition id to map dataframe
                partition_data_map.loc[sample_ids, "partition"] = str(j)

            partition_data_map_df = pd.DataFrame(
                data=partition_data_map["partition"], index=simulated_ind.sort()
            )

            # Save
            partition_data.to_csv(
                partition_file, float_format="%.3f", sep="\t", compression="xz"
            )

            partition_data_map_df.to_csv(partition_map_file, sep="\t", compression="xz")


def apply_correction_io(
    local_dir, run, dataset_name, analysis_name, num_experiments, correction_method
):
    """
    This function uses the limma or sva R package to correct for the technical variation
    we added using <add_experiments_io> or <add_experiments_grped_io>

    This function will return the corrected gene expression files

    Arguments
    ----------
    local_dir: str
        Root directory where simulated data with experiments/partitionings are be stored

    run: int
        Unique core identifier that is used to create unique filenames for intermediate files

    dataset_name:
        Name of analysis directory. Either "Human" or "Pseudomonas"

    analysis_name: str
        Parent directory where simulated data with experiments/partitionings are be stored.
        Format of the directory name is <dataset_name>_<sample/experiment>_lvl_sim 

    num_experiments: list
        List of different numbers of experiments/partitions to add
        technical variations to

    correction_method: str
        Noise correction method. Either "limma" or "combat"


    Returns
    --------
    Files of simulated data with different numbers of experiments added and corrected are saved to file.
    Each file is named as "Experiment_<number of experiments added>".
    Note: After the data is corrected, the dimensions are now gene x sample
    """

    for i in range(len(num_experiments)):

        if "sample" in analysis_name:
            print("Correcting for {} experiments..".format(num_experiments[i]))

            experiment_file = os.path.join(
                local_dir,
                "experiment_simulated",
                dataset_name + "_" + analysis_name,
                "Experiment_" + str(num_experiments[i]) + "_" + str(run) + ".txt.xz",
            )

            experiment_map_file = os.path.join(
                local_dir,
                "experiment_simulated",
                dataset_name + "_" + analysis_name,
                "Experiment_map_"
                + str(num_experiments[i])
                + "_"
                + str(run)
                + ".txt.xz",
            )

            # Read in data
            # data transposed to form gene x sample for R package
            experiment_data = pd.read_csv(
                experiment_file, header=0, index_col=0, sep="\t"
            ).T

            experiment_map = pd.read_csv(
                experiment_map_file, header=0, index_col=0, sep="\t"
            )["experiment"]
        else:
            print("Correcting for {} Partition..".format(num_experiments[i]))

            experiment_file = os.path.join(
                local_dir,
                "partition_simulated",
                dataset_name + "_" + analysis_name,
                "Partition_" + str(num_experiments[i]) + "_" + str(run) + ".txt.xz",
            )

            experiment_map_file = os.path.join(
                local_dir,
                "partition_simulated",
                dataset_name + "_" + analysis_name,
                "Partition_map_" + str(num_experiments[i]) + "_" + str(run) + ".txt.xz",
            )

            # Read in data
            # data transposed to form gene x sample for R package
            experiment_data = pd.read_csv(
                experiment_file, header=0, index_col=0, sep="\t"
            ).T

            experiment_map = pd.read_csv(
                experiment_map_file, header=0, index_col=0, sep="\t"
            )["partition"]

        if i == 0:
            corrected_experiment_data_df = experiment_data.copy()

        else:
            # Correct for technical variation
            if correction_method == "limma":
                corrected_experiment_data = limma.removeBatchEffect(
                    experiment_data, batch=experiment_map
                )

                # Convert R object to pandas df
                # corrected_experiment_data_df = pandas2ri.ri2py_dataframe(
                #    corrected_experiment_data)
                corrected_experiment_data_df = pd.DataFrame(corrected_experiment_data)

            if correction_method == "combat":
                corrected_experiment_data = sva.ComBat(
                    experiment_data, batch=experiment_map
                )

                # Convert R object to pandas df
                # corrected_experiment_data_df = pandas2ri.ri2py_dataframe(
                #    corrected_experiment_data)
                corrected_experiment_data_df = pd.DataFrame(corrected_experiment_data)

                corrected_experiment_data_df.columns = experiment_data.columns

        if "sample" in analysis_name:
            # Write out corrected files
            experiment_corrected_file = os.path.join(
                local_dir,
                "experiment_simulated",
                dataset_name + "_" + analysis_name,
                "Experiment_corrected_"
                + str(num_experiments[i])
                + "_"
                + str(run)
                + ".txt.xz",
            )

            corrected_experiment_data_df.to_csv(
                experiment_corrected_file,
                float_format="%.3f",
                sep="\t",
                compression="xz",
            )

        else:
            # Write out corrected files
            experiment_corrected_file = os.path.join(
                local_dir,
                "partition_simulated",
                dataset_name + "_" + analysis_name,
                "Partition_corrected_"
                + str(num_experiments[i])
                + "_"
                + str(run)
                + ".txt.xz",
            )

            corrected_experiment_data_df.to_csv(
                experiment_corrected_file,
                float_format="%.3f",
                sep="\t",
                compression="xz",
            )
