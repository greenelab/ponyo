import os
import pandas as pd
import numpy as np


def read_config(filename):
    """
    Read and parse configuration file containing stored user variables.

    These variables are then passed to the analysis notebooks
    and input to pipeline functions.
    """
    f = open(filename)
    config_dict = {}
    for lines in f:
        items = lines.split("\t", 1)
        config_dict[items[0]] = eval(items[1])
    return config_dict


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
    params = read_config(config_file)

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
        if not os.path.exists(each_dir):
            print("creating new directory: {}".format(each_dir))
            os.makedirs(each_dir, exist_ok=True)

        # Check if NN architecture directory exist otherwise create
    for each_dir in output_dirs:
        new_dir = os.path.join(each_dir, train_architecture)
        if not os.path.exists(new_dir):
            print("creating new directory: {}".format(new_dir))
            os.makedirs(new_dir, exist_ok=True)

    # Create results directories
    output_dirs = [os.path.join(base_dir, dataset_name, "results")]

    # Check if analysis output directory exist otherwise create
    for each_dir in output_dirs:
        if not os.path.exists(each_dir):
            print("creating new directory: {}".format(each_dir))
            os.makedirs(each_dir, exist_ok=True)

    # Check if 'saved_variables' directory exist otherwise create
    for each_dir in output_dirs:
        new_dir = os.path.join(each_dir, "saved_variables")

        if not os.path.exists(new_dir):
            print("creating new directory: {}".format(new_dir))
            os.makedirs(new_dir, exist_ok=True)

    # Create local directories to store intermediate files
    output_dirs = [
        os.path.join(local_dir, "experiment_simulated"),
        os.path.join(local_dir, "partition_simulated"),
    ]

    # Check if analysis output directory exist otherwise create
    for each_dir in output_dirs:
        if not os.path.exists(each_dir):
            print("creating new directory: {}".format(each_dir))
            os.makedirs(each_dir, exist_ok=True)


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
    params = read_config(config_file)

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
