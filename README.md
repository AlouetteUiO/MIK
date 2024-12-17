# Methane emission rate Inference of ruminants in Kenya (MIK)

These scripts were used in the study "Inferring methane emissions from African livestock by fusing drone, tower, and satellite data" (manuscript to be submitted).

The dataset will be made available here: 10.5281/zenodo.14214699.

The scripts include:
- 00_preprocess: The data from the various platforms is combined in one dataframe. The scripts have to be run in sequential order.
- 01_mass_balance: Application of the mass balance method. The scripts have to be run in sequential order, after running 00_preprocess scripts.
- 02_bayesian_inference: Application of the Bayesian inference method. The scripts have to be run in sequential order, after running 00_preprocess scripts.

Please, adjust the file locations in the scripts. Output will be written to the 'data' folder (now empty). 
