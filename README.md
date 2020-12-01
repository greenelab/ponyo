<img src="https://github.com/greenelab/ponyo/blob/master/logo.png" width=150 align=right> 

# ponyo 
[![Coverage Status](https://coveralls.io/repos/github/greenelab/ponyo/badge.svg?branch=master)](https://coveralls.io/github/greenelab/ponyo?branch=master)

**Alexandra Lee and Casey Greene 2020**

**University of Pennsylvania**

This repository is named after the the character [Ponyo](https://en.wikipedia.org/wiki/Ponyo), from Hayao Miyazaki's animated film *Ponyo*, as she uses her magic to simulate a human appearance after getting a sample of human blood. 
The method simulates a compendia of new gene expression data based on existing gene expression data to learn a representation of gene expression patterns.

## Installation

This package can be installed using pip:

```
pip install ponyo
```

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

The tables lists the core parameters required to generate simulated data using modules from [ponyo](https://github.com/greenelab/ponyo). Those marked with * indicate those parameters that will vary depending on the type of approach .

| Name | Description |
| :--- | :---------- |
| local_dir| str: Parent directory on local machine to store intermediate results|
| dataset_name| str: Name for analysis directory containing notebooks using ponyo|
| raw_data_filename| str: File storing raw gene expression data|
| normalized_data_filename| str: File storing normalized gene expression data|
| metadata_filename*| str: File containing metadata associated with data|
| experiment_ids_filename*| str: File containing list of experiment ids that have gene expression data available|
| scaler_transform_filename| str: File to store mapping from normalized to raw gene expression range|
| simulation_type | str: Name of simulation approach directory to store results locally|
| NN_architecture | str: Name of neural network architecture to use. Format 'NN_<intermediate layer>_<latent layer>'|
| learning_rate| float: Step size used for gradient descent. In other words, it's how quickly the  methods is learning|
| batch_size | str: Training is performed in batches. So this determines the number of samples to consider at a given time|
| epochs | int: Number of times to train over the entire input dataset|
| kappa | float: How fast to linearly ramp up KL loss|
| intermediate_dim| int: Size of the hidden layer|
| latent_dim | int: Size of the bottleneck layer|
| epsilon_std | float: Standard deviation of Normal distribution to sample latent space|
| validation_frac | float: Fraction of input samples to use to validate for VAE training|
| num_simulated_samples* | int: If using random sampling approach, simulate a compendia with these many samples|
| num_simulated_experiments*| int: If using latent-transformation approach, simulate a compendia with these many experiments|
| num_simulated*| int: If using template-based approach, simulate these many experiments|
| metadata_delimiter*| str: Delimiter to parse metadata file|
| metadata_experiment_colname* | str: Column header that contains experiment id that maps expression data and metadata|
| metadata_sample_colname* | str: Column header that contains sample id that maps expression data and metadata|
| project_id*| int: If using template-based approach, experiment id to use as template experiment|
