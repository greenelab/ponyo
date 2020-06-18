"""
Author: Alexandra Lee
Date Created: 11 March 2020

Scripts called by analysis notebooks to run entire the entire analysis pipeline:
1. setup directories
2. Process data
3. Train VAE
4. Run simulation experiment, described in `simulations.py`
"""

import os
import sys
import ast
import pandas as pd
import numpy as np
import random
import math
from sklearn import preprocessing

from joblib import Parallel, delayed
import multiprocessing

import warnings

def fxn(): 
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():

from ponyo import vae, utils, simulations

from numpy.random import seed

randomState = 123
seed(randomState)


def setup_dir(config_file):
    """
    Create directories to store files created by VAE training and
    simulation analysis

    Arguments
    ----------
    config_file: str
        File containing user defined parameters
    """

    base_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))

    # Read in config variables
    params = utils.read_config(config_file)

    # Load parameters
    local_dir = params["local_dir"]
    dataset_name = params["dataset_name"]
    train_architecture = params["NN_architecture"]

    # Create VAE directories
    output_dirs = [
        os.path.join(base_dir, dataset_name, "models"),
        os.path.join(base_dir, dataset_name, "logs"),
    ]

    # Check if analysis output directory exist otherwise create
    for each_dir in output_dirs:
        if os.path.exists(each_dir) == False:
            print("creating new directory: {}".format(each_dir))
            os.makedirs(each_dir, exist_ok=True)

        # Check if NN architecture directory exist otherwise create
    for each_dir in output_dirs:
        new_dir = os.path.join(each_dir, train_architecture)
        if os.path.exists(new_dir) == False:
            print("creating new directory: {}".format(new_dir))
            os.makedirs(new_dir, exist_ok=True)

    # Create results directories
    output_dirs = [os.path.join(base_dir, dataset_name, "results")]

    # Check if analysis output directory exist otherwise create
    for each_dir in output_dirs:
        if os.path.exists(each_dir) == False:
            print("creating new directory: {}".format(each_dir))
            os.makedirs(each_dir, exist_ok=True)

    # Check if 'saved_variables' directory exist otherwise create
    for each_dir in output_dirs:
        new_dir = os.path.join(each_dir, "saved_variables")

        if os.path.exists(new_dir) == False:
            print("creating new directory: {}".format(new_dir))
            os.makedirs(new_dir, exist_ok=True)

    # Create local directories to store intermediate files
    output_dirs = [
        os.path.join(local_dir, "experiment_simulated"),
        os.path.join(local_dir, "partition_simulated"),
    ]

    # Check if analysis output directory exist otherwise create
    for each_dir in output_dirs:
        if os.path.exists(each_dir) == False:
            print("creating new directory: {}".format(each_dir))
            os.makedirs(each_dir, exist_ok=True)


def transpose_data(data_file, out_file):
    """
    Transpose and save expression data so that it is of the form sample x gene

    Arguments
    ----------
    data_file: str
        File containing gene expression

    out_file: str
        File containing transposed gene expression
    """
    # Read data
    data = pd.read_csv(data_file, header=0, sep="\t", index_col=0)

    data.T.to_csv(out_file, sep="\t", compression="xz")


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

    # Load parameters
    dataset_name = params["dataset_name"]

    # Read data
    data = pd.read_csv(raw_input_data_file, header=0, sep="\t", index_col=0)

    print(
        "input: dataset contains {} samples and {} genes".format(
            data.shape[0], data.shape[1]
        )
    )

    # 0-1 normalize per gene
    data_scaled_df = preprocessing.MinMaxScaler().fit_transform(data)
    data_scaled_df = pd.DataFrame(
        data_scaled_df, columns=data.columns, index=data.index
    )

    print(
        "Output: normalized dataset contains {} samples and {} genes".format(
            data_scaled_df.shape[0], data_scaled_df.shape[1]
        )
    )

    # Save scaled data
    data_scaled_df.to_csv(normalized_data_file, sep="\t", compression="xz")


def create_experiment_id_file(metadata_file, input_data_file, output_file, config_file):
    """
    Create file with experiment ids that are associated with expression data

    Arguments
    ----------
    metadata_file: str
        File containing metadata annotations per sample

    input_data_file: str
        File containing normalized expression data

    output_file: str
        File containing experiment ids with expression data and sample annotations

    config_file: str
        File containing user defined parameters

    """
    # Read in metadata
    metadata = pd.read_csv(metadata_file, header=0, sep="\t", index_col=0)

    # Read in expression data
    normalized_data = pd.read_csv(input_data_file, header=0, sep="\t", index_col=0)

    # Read in config variables
    params = utils.read_config(config_file)

    # Load parameters
    sample_id_colname = params["metadata_colname"]
    dataset_name = params["dataset_name"]

    # Get sample id that maps between metadata and expression files
    map_experiment_sample = metadata[[sample_id_colname]]

    # Get all unique experiment ids
    experiment_ids = np.unique(np.array(map_experiment_sample.index)).tolist()
    print("There are {} experiments in the compendium".format(len(experiment_ids)))

    # Get only sample ids that have expression data available
    sample_ids_with_gene_expression = list(normalized_data.index)

    # Get associated experiment ids with expression data
    experiment_ids_with_gene_expression = []

    for experiment_id in experiment_ids:

        if "human" in dataset_name.lower():
            # Some project id values are descriptions
            # We will skip these
            if len(experiment_id) == 9:
                selected_metadata = metadata.loc[experiment_id]
                sample_ids = list(selected_metadata[sample_id_colname])

                if any(x in sample_ids_with_gene_expression for x in sample_ids):
                    experiment_ids_with_gene_expression.append(experiment_id)
        else:
            selected_metadata = metadata.loc[experiment_id]
            sample_ids = list(selected_metadata[sample_id_colname])

            if any(x in sample_ids_with_gene_expression for x in sample_ids):
                experiment_ids_with_gene_expression.append(experiment_id)

    print(
        "There are {} experiments with gene expression data".format(
            len(experiment_ids_with_gene_expression)
        )
    )

    # Save file with experiment ids
    experiment_ids_with_gene_expression_df = pd.DataFrame(
        experiment_ids_with_gene_expression, columns=["experiment_id"]
    )
    experiment_ids_with_gene_expression_df.to_csv(output_file, sep="\t")
    print(
        "{} experiment ids saved to file".format(
            len(experiment_ids_with_gene_expression)
        )
    )


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
    )


def run_simulation(config_file, input_data_file, corrected, experiment_ids_file=None):
    """
    Runs simulation experiment without applying correction method

    Arguments
    ----------
    config_file: str
        File containing user defined parameters

    input_data_file: str
        File path corresponding to input dataset to use

    corrected: bool
        True if simulation is applying noise correction 

    experiment_ids_file: str
        File containing experiment ids with expression data associated generated from ```create_experiment_id_file```

    """

    # Read in config variables
    params = utils.read_config(config_file)

    # Load parameters
    dataset_name = params["dataset_name"]
    simulation_type = params["simulation_type"]
    NN_architecture = params["NN_architecture"]
    use_pca = params["use_pca"]
    num_PCs = params["num_PCs"]
    local_dir = params["local_dir"]
    correction_method = params["correction_method"]
    sample_id_colname = params["metadata_colname"]
    iterations = params["iterations"]
    num_cores = params["num_cores"]

    if "sample" in simulation_type:
        num_simulated_samples = params["num_simulated_samples"]
        lst_num_experiments = params["lst_num_experiments"]
    else:
        num_simulated_experiments = params["num_simulated_experiments"]
        lst_num_partitions = params["lst_num_partitions"]

    # Output files
    # base_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))
    base_dir = os.path.abspath(os.pardir)
    if corrected:
        similarity_uncorrected_file = os.path.join(
            base_dir,
            dataset_name,
            "results",
            "saved_variables",
            dataset_name
            + "_"
            + simulation_type
            + "_svcca_corrected_"
            + correction_method
            + ".pickle",
        )

        ci_uncorrected_file = os.path.join(
            base_dir,
            dataset_name,
            "results",
            "saved_variables",
            dataset_name
            + "_"
            + simulation_type
            + "_ci_corrected_"
            + correction_method
            + ".pickle",
        )

    else:
        similarity_uncorrected_file = os.path.join(
            base_dir,
            dataset_name,
            "results",
            "saved_variables",
            dataset_name
            + "_"
            + simulation_type
            + "_svcca_uncorrected_"
            + correction_method
            + ".pickle",
        )

        ci_uncorrected_file = os.path.join(
            base_dir,
            dataset_name,
            "results",
            "saved_variables",
            dataset_name
            + "_"
            + simulation_type
            + "_ci_uncorrected_"
            + correction_method
            + ".pickle",
        )

    similarity_permuted_file = os.path.join(
        base_dir,
        dataset_name,
        "results",
        "saved_variables",
        dataset_name + "_" + simulation_type + "_permuted",
    )

    # Run multiple simulations
    if "sample" in simulation_type:
        if corrected:
            file_prefix = "Experiment_corrected"
        else:
            file_prefix = "Experiment"
        results = Parallel(n_jobs=num_cores, verbose=100)(
            delayed(simulations.sample_level_simulation)(
                i,
                NN_architecture,
                dataset_name,
                simulation_type,
                num_simulated_samples,
                lst_num_experiments,
                corrected,
                correction_method,
                use_pca,
                num_PCs,
                file_prefix,
                input_data_file,
                local_dir,
                base_dir,
            )
            for i in iterations
        )

    else:
        if corrected:
            file_prefix = "Partition_corrected"
        else:
            file_prefix = "Partition"
        results = Parallel(n_jobs=num_cores, verbose=100)(
            delayed(simulations.experiment_level_simulation)(
                i,
                NN_architecture,
                dataset_name,
                simulation_type,
                num_simulated_experiments,
                lst_num_partitions,
                corrected,
                correction_method,
                use_pca,
                num_PCs,
                file_prefix,
                input_data_file,
                experiment_ids_file,
                sample_id_colname,
                local_dir,
                base_dir,
            )
            for i in iterations
        )

    # permuted score
    permuted_score = results[0][0]

    # Concatenate output dataframes
    all_svcca_scores = pd.DataFrame()

    for i in iterations:
        all_svcca_scores = pd.concat([all_svcca_scores, results[i][1]], axis=1)

    # Get mean svcca score for each row (number of experiments)
    mean_scores = all_svcca_scores.mean(axis=1).to_frame()
    mean_scores.columns = ["score"]
    print(mean_scores)

    # Get standard dev for each row (number of experiments)
    std_scores = (all_svcca_scores.std(axis=1) / math.sqrt(10)).to_frame()
    std_scores.columns = ["score"]
    print(std_scores)

    # Get confidence interval for each row (number of experiments)
    # z-score for 95% confidence interval
    err = std_scores * 1.96

    # Get boundaries of confidence interval
    ymax = mean_scores + err
    ymin = mean_scores - err

    ci = pd.concat([ymin, ymax], axis=1)
    ci.columns = ["ymin", "ymax"]
    print(ci)

    # Pickle dataframe of mean scores scores for first run, interval
    mean_scores.to_pickle(similarity_uncorrected_file)
    ci.to_pickle(ci_uncorrected_file)
    np.save(similarity_permuted_file, permuted_score)

