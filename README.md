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