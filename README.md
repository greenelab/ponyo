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
| Simulation by random sampling| This approach simulates gene expression data by randomly sampling from the latent space distribution. The function to run this approach is divided into 2 components: `simulate_by_random_sampling` is a wrapper which loads VAE trained models from directory `<base_dir>/<dataset_name>/"models"/<NN_architecture>` and `run_sample_simulation` which runs the simulation. Note: `simulate_by_random_sampling` assumes the files are organized as described above. If this directory organization doesn't apply to you, then you can directly use `run_sample_simulation` by passing in your pre-trained VAE model files. An example of how to use this can be found [here](https://github.com/greenelab/ponyo/blob/master/human_tests/Human_random_sampling_simulation.ipynb). |
| Simulation by latent transformation| This approach simulates gene expression data by encoding experiments into the latent space and then shifting samples from that experiment in the latent space. Unlike the "Simulation by random sampling" approach, this method accounts for experiment level information by shifting samples from the same experiment together. The function to run this approach is divided into 2 components: `simulate_by_latent_transformation` is a wrapper which loads VAE trained models from directory `<base_dir>/<dataset name>/"models"/<NN_architecture>` and `run_latent_transformation_simulation` which runs the simulation. Note: `simulate_by_latent_transformation` assumes the files are organized as described above. If this directory organization doesn't apply to you, then you can directly use `run_latent_transformation_simulation` by passing in your VAE model trained using `run_tybalt_training` in [vae.py](https://github.com/greenelab/ponyo/blob/master/ponyo/vae.py). <br><br>There are 3 flavors of this approach: <br><br> 1. `simulate_by_latent_transform` inputs a dataset with multiple experiments (these are your template experiments) and then it outputs the same number of new simulated experiments that are created by shifting a randomly spampled template experiment. This simulation generates a collection of different types of experiments. An example of how to use this can be found [here](https://github.com/greenelab/ponyo/blob/master/human_tests/Human_latent_transform_simulation.ipynb). <br><br> 2. `shift_template_experiment` which inputs a single template experiment, which is an experiment included within the dataset used to train the VAE model. This simulation outputs multiple simulated experiments based on the one template by shifting that template experiment to different locations in the latent space. This simulation generates a collection of experiments of a similar design type. An example for how to use this can be found [here](https://github.com/greenelab/ponyo/blob/master/human_tests/Human_template_simulation.ipynb). <br><br> 3. `embed_shift_template_experiment` which performs the same simulation approach as `shift_template_experiment` however this is using a template experiment that is not contained within the training dataset. An example for how to use this can be found [here](https://github.com/greenelab/ponyo/blob/master/human_tests/Human_embed_shift_template_simulation.ipynb).
|


## How to use
Example notebooks using ponyo on test data can be found in [human_tests](https://github.com/greenelab/ponyo/tree/master/human_tests)

Additionally, this method has been used in [simulate-expression-compendia](https://github.com/greenelab/simulate-expression-compendia) and [generic-expression-patterns](https://github.com/greenelab/generic-expression-patterns) repositories.

## Setting random seeds
To keep the VAE training deterministic, you will need to set multiple random seeds:
1. numpy random
2. python random
3. tensorflow random

For an example of this, see [human_tests](https://github.com/greenelab/ponyo/tree/master/human_tests)

## Configuration file

The tables lists the core parameters required to generate simulated data using modules from [ponyo](https://github.com/greenelab/ponyo). Those marked with * indicate those parameters that will vary depending on the type of approach used.

| Name | Description |
| :--- | :---------- |
| base_dir| str: Root directory containing analysis subdirectories. By default the path is one level up from where the scripts are run.|
| local_dir| str: Parent directory on local machine to store intermediate results|
| dataset_name| str: Name for analysis directory containing notebooks using ponyo|
| raw_data_filename| str: File storing raw gene expression data|
| normalized_data_filename| str: File storing normalized gene expression data. This file is generated by [normalize_expression_data()](https://github.com/greenelab/ponyo/blob/master/ponyo/train_vae_modules.py).|
| metadata_filename*| str: File containing metadata associated with data. This file maps samples to a given experiment. This parameter needed if using the latent transformation approaches.|
| metadata_delimiter*| "," or "\t" to denote the delimiter used in the metadata file. This parameter needed if using either latent transformation approaches.|
| experiment_ids_filename*| str: File containing list of experiment ids that have gene expression data available. This parameter needed if using either latent transformation approaches.|
| scaler_transform_filename| str: Python pickle file to store mapping from normalized to raw gene expression range. This file is generated by [normalize_expression_data()](https://github.com/greenelab/ponyo/blob/master/ponyo/train_vae_modules.py).|
| simulation_type | str: Name of simulation approach directory to store results locally|
| NN_architecture | str: Name of neural network architecture to use. Format `NN_<intermediate layer>_<latent layer>`|
| learning_rate| float: Step size used for gradient descent. In other words, it's how quickly the  methods is learning|
| batch_size | str: Training is performed in batches. So this determines the number of samples to consider at a given time|
| epochs | int: Number of times to train over the entire input dataset|
| kappa | float: How fast to linearly ramp up KL loss|
| intermediate_dim| int: Size of the hidden layer|
| latent_dim | int: Size of the bottleneck layer|
| epsilon_std | float: Standard deviation of Normal distribution to sample latent space|
| validation_frac | float: Fraction of input samples to use to validate for VAE training|
| num_simulated_samples* | int: The number of samples to simulate. This parameter is needed if using the random sampling approach.|
| num_simulated_experiments*| int: The number of experiments to simulate. This parameter is needed if using the latent transformation approach - simulate_by_latent_transform.|
| num_simulated*| int: The number of experiments to simulate (i.e. the number of times to shift the starting template experiment). This parameter is needed if using the latent transformation approach - shift_template_experiment.|
| metadata_experiment_colname* | str: Column header that contains experiment id that maps expression data and metadata. This parameter needed if using either latent transformation approaches.|
| metadata_sample_colname* | str: Column header that contains sample id that maps expression data and metadata. This parameter needed if using either latent transformation approaches.|
| project_id*| int: The experiment id to use as a template experiment. This <project_id> corresponds to a group of samples that were used to test an single hypothesis. This parameter is needed if using either `shift_template_experiment` or  `embed_shift_template_experiment` approaches. If using `shift_template_experiment`, the id is pulled from the <metadata_experiment_colname> column of the <metadata_filename>. If using `embed_shift_template_experiment`, the id is used to name the simulated files generated.|

For guidance on setting VAE training prameters, see configurations used in [simulate-expression-compendia](https://github.com/greenelab/simulate-expression-compendia/configs) and [generic-expression-patterns](https://github.com/greenelab/generic-expression-patterns/configs) repositories


## Acknowledgements
We would like to thank Marvin Thielk for adding coverage to tests and Ben Heil for contributing code to add more flexibility.
