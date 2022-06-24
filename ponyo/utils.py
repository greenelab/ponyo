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


def setup_dir(config_filename):
    """
    Create directories to store files created by VAE training and
    simulation analysis

    Arguments
    ----------
    config_filename: str
        File containing user defined parameters
    """

    # Read in config variables
    params = read_config(config_filename)

    # Load parameters    
    vae_model_dir = params["vae_model_dir"]
    local_dir = params["local_dir"]

    # Directories to create
    output_dirs = [
        vae_model_dir,
        local_dir,
        ]

    # Simulated data directory is only required for 
    # shift_template_experiment and embed_shift_template_experiment
    # simulation types
    # Here, check if this parameter is specified in the config 
    # file.

    # Training_stats_dir is only needed if training a new VAE model
    if "training_stats_dir" in params:
        training_stats_dir = params["training_stats_dir"]

        output_dirs.append(training_stats_dir)

    # simulated_data_dir is only needed for `shift_template_experiment`
    # and `embed_shift_template_experiment` simulation types
    if "simulated_data_dir" in params:
        simulated_data_dir = params["simulated_data_dir"]

        output_dirs.append(simulated_data_dir)

    # Check if the following directories exist
    # and if not to create them
    for each_dir in output_dirs:

        # Check if analysis output directory exist otherwise create
        if not os.path.exists(each_dir):
            print("creating new directory: {}".format(each_dir))
            os.makedirs(each_dir, exist_ok=True)




def create_experiment_id_file(
    metadata_filename, input_data_filename, output_filename, config_filename
):
    """
    Create file with experiment ids that are associated with expression data

    Arguments
    ----------
    metadata_filename: str
        File containing metadata annotations per sample

    input_data_filename: str
        File containing normalized expression data

    output_filename: str
        File containing experiment ids with expression data and sample annotations

    config_filename: str
        File containing user defined parameters

    """
    # Read in metadata
    metadata = pd.read_csv(metadata_filename, header=0, sep="\t", index_col=0)

    # Read in expression data
    normalized_data = pd.read_csv(input_data_filename, header=0, sep="\t", index_col=0)

    # Read in config variables
    params = read_config(config_filename)

    # Load parameters
    sample_id_colname = params["metadata_sample_colname"]
    is_recount2 = params["is_recount2"]

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

        if is_recount2:
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
    experiment_ids_with_gene_expression_df.to_csv(output_filename, sep="\t")
    print(
        "{} experiment ids saved to file".format(
            len(experiment_ids_with_gene_expression)
        )
    )
