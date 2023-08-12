# About

This work contains scripts that were used in the paper: 
**Optimizing performance and energy across problem sizes through a search space exploration and machine learning**[1].

# Initialization

We start by describing how to setup the script environment.
Please execute all the commands from the root directly of the project.

## Python 3

Please install python 3.8.
Packages include: sklearn, pandas, matplotlib, pyeasyga, and numpy.

## Communication Metrics 

This section describes how to generate the communication metrics described in Section 5.3. Application characteristics (features).
They are necessary to train the ML models. 

```console
$ cd communication-matrix/ 
$ tar -xf input-ml-explo-matrices.tar.xz 
$ tar -xf input-ml-explo-trace.tar.xz
$ git clone https://github.com/llpilla/communication-statistics.git  communicationstatistics
$ python3 generate_comm_metrics.py 
```

These commands generate the directory communication-matrix containing the information later on used as features to train the ML models.

## Execution time & Energy consumption

The goal of this section is to expose the timing and energy measurements.
They can all be accessed at: https://zenodo.org/record/7590756

The directory contains in details:
* region: a file listing all the regions we study
* stability: directory containing the stability reports for all the performance and energy measurements
* values: directory containing all the performance and energy measurements

Both stability and values contain files named:
conf1_energy.csv  conf1_time_med.csv  conf1_time_sum.csv  conf2_energy.csv  conf2_time_med.csv  conf2_time_sum.csv

Where,
* confX refers to the size small or large
* energy show energy measurements
* time_med refers to the median across the meta-repetitions while time_sum us the sum across the meta-repetitions. 

Note that execution times and energy measurements can be used both as labels and features for the models.
Also some measurements indicated in settings were switched between sizes for consistency (i.e., codes with larger inputs were set in small runs).

# Results

## Evaluating the potential gains of the search optimization

To reproduce the results presented in Section 4, please execute:

```console
$ cd ml
$ python3 main.py 1 energy
```

The script returns the average gains of exploring the space along its size.
Then, it describes in details the gains per code.

It then performs the evaluation of sub spaces presented in Section 4.1 Figure 4.
For each TS, it provides the gains per code along with how the different sub spaces affect the gains.

Finally, it projects the discovered configurations from one TS to another one from Section 4.2, Figure5.

## Machine learning models

To train a model, please run the function call_classfy.
It takes as argument a list of features to be used.
The function will perform the 10-fold cross validation, train a decision tree, and return its average performance across all the validation folds.

# Remarks

If you have any question on this code, please contact Mihail Popov (mihail.popov@inria.fr).

Reference:
[1] Lana Scravaglieri, Mihail Popov, Laércio Lima Pilla, Amina Guermouche, Olivier Aumage, and Emmanuelle Saillard. "Optimizing Performance and Energy Across Problem Sizes Through a Search Space Exploration and Machine Learning." Journal of Parallel and Distributed Computing (2023): 104720.
