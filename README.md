<img src="https://github.com/greenelab/ponyo/blob/master/logo.png" width=150 align=right> 

# ponyo 
[![Coverage Status](https://coveralls.io/repos/github/greenelab/ponyo/badge.svg?branch=master)](https://coveralls.io/github/greenelab/ponyo?branch=master)

**Alexandra J. Lee and Casey S. Greene 2020**

**University of Pennsylvania**

This repository is named after the the character [Ponyo](https://en.wikipedia.org/wiki/Ponyo), from Hayao Miyazaki's animated film *Ponyo*, as she uses her magic to simulate a human appearance after getting a sample of human blood. 
The method simulates new gene expression data by training a generative neural network on existing gene expression data to learn a representation of gene expression patterns.

## Installation

This package can be installed using pip:

```
pip install ponyo
```

## Types of simulations
There are 3 types of simulations that ponyo implements:

| Name | Description |
| :--- | :---------- |
| Simulation by random sampling| This approach simulates gene expression data by randomly sampling from the latent space distribution. The function to run this approach is divided into 2 components: `simulate_by_random_sampling` is a wrapper which loads VAE trained models from directory specified by `vae_model_dir` param and `run_sample_simulation` which runs the simulation. Note: `simulate_by_random_sampling` assumes the files are organized as described above. If this directory organization doesn't apply to you, then you can directly use `run_sample_simulation` by passing in your pre-trained VAE model files. An example of how to use this can be found [here](https://github.com/greenelab/ponyo/blob/master/examples/Example_simulate_by_random_sampling.ipynb). |
| Simulation by latent transformation| This approach simulates gene expression data by encoding experiments into the latent space and then shifting samples from that experiment in the latent space. Unlike the "Simulation by random sampling" approach, this method accounts for experiment level information by shifting samples from the same experiment together. The function to run this approach is divided into 2 components: `simulate_by_latent_transformation` is a wrapper which loads VAE trained models from directory specified by the `vae_model_dir` param and `run_latent_transformation_simulation` which runs the simulation. Note: `simulate_by_latent_transformation` assumes the files are organized as described above. If this directory organization doesn't apply to you, then you can directly use `run_latent_transformation_simulation` by passing in your VAE model trained using `run_tybalt_training` in [vae.py](https://github.com/greenelab/ponyo/blob/master/ponyo/vae.py). <br><br>There are 3 flavors of this approach: <br><br> 1. `simulate_by_latent_transform` inputs a dataset with multiple experiments (these are your template experiments) and then it outputs the same number of new simulated experiments that are created by shifting a randomly sampled template experiment. This simulation generates a collection of different types of experiments. An example of how to use this can be found [here](https://github.com/greenelab/ponyo/blob/master/examples/Example_simulate_by_latent_transform.ipynb). <br><br> 2. `shift_template_experiment` which inputs a single template experiment, which is an experiment included within the dataset used to train the VAE model. This simulation outputs multiple simulated experiments based on the one template by shifting that template experiment to different locations in the latent space. This simulation generates a collection of experiments of a similar design type. An example for how to use this can be found [here](https://github.com/greenelab/ponyo/blob/master/examples/Example_shift_template_experiment.ipynb). <br><br> 3. `embed_shift_template_experiment` which performs the same simulation approach as `shift_template_experiment` however this is using a template experiment that is not contained within the training dataset. An example for how to use this can be found [here](https://github.com/greenelab/ponyo/blob/master/examples/Example_embed_shift_template_experiment.ipynb).|


## How to use
Example notebooks using ponyo on test data can be found in [examples](https://github.com/greenelab/ponyo/tree/master/examples/)

Additionally, this method has been used in [simulate-expression-compendia](https://github.com/greenelab/simulate-expression-compendia) and [generic-expression-patterns](https://github.com/greenelab/generic-expression-patterns) repositories.

## Setting random seeds
To keep the VAE training deterministic, you will need to set multiple random seeds:
1. numpy random
2. python random
3. tensorflow random

For an example of this, see [example notebooks](https://github.com/greenelab/ponyo/tree/master/examples/)

## Configuration file

The tables lists the core parameters required to generate simulated data using modules from [ponyo](https://github.com/greenelab/ponyo).

Parameters required for **all simulation types**:

| Name | Description |
| :--- | :---------- |
| base_dir| str: Root directory containing analysis subdirectories. By default the path is one level up from where the scripts are run.|
| local_dir| str: Parent directory on local machine to store intermediate results|
| raw_compenium_filename| str: File storing raw gene expression data|
| normalized_compendium_filename| str: File storing normalized gene expression data. This file is generated by [normalize_expression_data()](https://github.com/greenelab/ponyo/blob/master/ponyo/train_vae_modules.py).|
| scaler_transform_filename| str: Python pickle file to store mapping from normalized to raw gene expression range. This file is generated by [normalize_expression_data()](https://github.com/greenelab/ponyo/blob/master/ponyo/train_vae_modules.py).|
| vae_model_dir | str:  The location where the VAE model files (.h5) are stored.|
| learning_rate| float: Step size used for gradient descent. In other words, it's how quickly the  methods is learning|
| batch_size | str: Training is performed in batches. So this determines the number of samples to consider at a given time|
| epochs | int: Number of times to train over the entire input dataset|
| kappa | float: How fast to linearly ramp up KL loss|
| intermediate_dim| int: Size of the hidden layer|
| latent_dim | int: Size of the bottleneck layer|
| epsilon_std | float: Standard deviation of Normal distribution to sample latent space|
| validation_frac | float: Fraction of input samples to use to validate for VAE training|
| training_stats_dir| str: Directory containing the VAE training log files.|
| num_simulated| int: The number of samples (for simulation by random sampling approach) or experiments (for all latent transformation approaches) to simulate. |

Additional parameters required for **all latent transformation approaches**:
| Name | Description |
| :--- | :---------- |
| project_id| int: The experiment id to use as a template experiment. This <project_id> corresponds to a group of samples that were used to test an single hypothesis. This parameter is needed if using either `shift_template_experiment` or  `embed_shift_template_experiment` approaches. If using `shift_template_experiment`, the id is pulled from the <metadata_experiment_colname> column of the <metadata_filename>. If using `embed_shift_template_experiment`, the id is used to name the simulated files generated.|
| is_recount2| bool: True is the compendium dataset being used is recount2. This will determine how experiment ids are parsed for latent transformation approaches.|

Additional parameters required for `simulate_by_latent_transform` and `shift_template_experiment` simulation types only:
| Name | Description |
| :--- | :---------- |
| metadata_filename| str: File containing metadata associated with data. This file maps samples to a given experiment.|
| metadata_delimiter| "," or "\t" to denote the delimiter used in the metadata file.|
| metadata_experiment_colname | str: Column header that contains experiment id that maps expression data and metadata. This parameter needed if using either latent transformation approaches.|
| metadata_sample_colname | str: Column header that contains sample id that maps expression data and metadata. This parameter needed if using either latent transformation approaches.|
| experiment_ids_filename| str: File containing list of experiment ids that have gene expression data available. |

Additional parameters required for `embed_shift_template_experiment` simulation type only:
| Name | Description |
| :--- | :---------- |
| raw_template_filename | str: Downloaded template gene expression data file. The input dataset should be a matrix that is sample x gene. The file should tab-delimited. The gene ids need to be consistent between the template and compendium datasets. The input dataset should be generated using the same platform as the model you plan to use (i.e. RNA-seq or array). The expression values are expected to have been uniformly processed and can be estimated counts (RNA-seq) or log2 expression (array).|
| mapped_template_filename | str: Template gene expression data filename. This file is generated by scale transforming the data using the scaler_filename. The gene ids of the template file and the compendium file are matched.|
| normalized_template_filename | str: Normalized template gene expression data filename.|

For guidance on setting VAE training prameters, see configurations used in [simulate-expression-compendia](https://github.com/greenelab/simulate-expression-compendia/configs) and [generic-expression-patterns](https://github.com/greenelab/generic-expression-patterns/configs) repositories


## Acknowledgements
We would like to thank Marvin Thielk for adding coverage to tests and Ben Heil for contributing code to add more flexibility.
