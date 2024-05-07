### Config details 

1. Change file paths in globals.py
2. add Data folder and keep data locally
3. download data from https://ashitabh.egnyte.com/fl/qdZlXcXLxa

4. ./Data/individual_time_samples is the directory structure
5. Audio data is 16000 samples in each .pt file (The parkland readme might've a typo), freq 8000Hz so 2 seconds
6. The main file to use as a reference for analysis is Analytics/analyse_fft_pts
7. To run a file in analytics use: ```python -m Analytics.analyse_fft_pts``` ( the project is structured as a python package

### Pattern mining
1. Run: ```pip install pycspade```
2. Run: ```python3 Analytics/example_file.py``` from root directory

### Interpreting the patterns
The patterns are made for each label separately. The data corresponding to each label consists of time series, each with 7 steps and 63 features originally, corresponding to different frequencies. The itemsets used in the pattern mining are constructed by passing the features at each event in each sequence through a maxpool layer, which results in an itemset of size 9. The 80th percentile of the values in the resulting data for the label is used to threshold the values and itemsets are hence formed. 

The patterns consist of subsequences of the original sample sequences and can be interpreted as follows. If an value occurs in the sequence, then its amplitude would be higher than the 80th percentile and hence it will be a dominant frequency. Hence, the patterns indicate the succession of the dominant frequencies in the sequences for each label, that characterize the label.

### DualQuanv Training and Testing
#### Training
To conduct the DualQuanv training experiment, follow these steps:

1. Ensure Dependencies: Make sure you have the necessary dependencies installed, including PyTorch and other required libraries.
2. Prepare Datasets: Place the training, validation, and test datasets in the specified data_dir directory. Dataset files should be named train_tensor_dataset_stft_3vehicles_pd_0_5_125_0_5.pth, val_tensor_dataset_stft_3vehicles_pd_0_5_125_0_5.pth, and test_tensor_dataset_stft_3vehicles_pd_0_5_125_0_5.pth.
3. Set Hyperparameters: Open the dual_quant_experiment.py file and set the appropriate hyperparameters, such as bitwidth_non_pattern_bands, bitwidth_probs_pattern_band, lower_bound_ratio, upper_bound_ratio, and num_epochs.
4. Run Experiment: Execute the following command to start the DualQuanv training experiment: ```python DataMiningProject/DualQuanv/dual_quant_experiment.py```

#### Testing
To conduct the DualQuanv testing experiment, follow these steps:

1. Prepare Model Checkpoint: Ensure you have the trained DualQuanv model checkpoint available. The checkpoint file will be automatically named as dual_quant_model_4_8_12_16.pth (or similar based on used bitwidths).
2. Prepare Test Dataset: Place the test dataset file (test_tensor_dataset_stft_3vehicles_pd_0_5_125_0_5.pth) in the specified data_dir directory.
3. Set Bitwidths and Ratios: Open the test.py file and set the appropriate bitwidth allocation and lower/upper bound ratios for testing.
4. Run Experiment: Execute the following command to start the DualQuanv testing experiment: ```python DataMiningProject/DualQuanv/test.py```
