### Config details 

1. Change file paths in globals.py
2. add Data folder and keep data locally
3. download data from https://ashitabh.egnyte.com/fl/qdZlXcXLxa

4. ./Data/individual_time_samples is the directory structure
5. Audio data is 16000 samples in each .pt file (The parkland readme might've a typo), freq 8000Hz so 2 seconds
6. The main file to use as a reference for analysis is Analytics/analyse_fft_pts
7. To run a file in analytics use: ```python -m Analytics.analyse_fft_pts``` ( the project is structured as a python package
