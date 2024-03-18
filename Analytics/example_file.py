import torch
import torch.nn as nn
from pycspade.helpers import spade, print_result
import sys

orig_stdout = sys.stdout

quantile = 0.8
support = 0.5

def data2features(stft_data):
    maxpool_layer = nn.MaxPool1d(kernel_size=7, stride=7)
    stft_data = maxpool_layer(stft_data.permute(0, 2, 1)).permute(0, 2, 1)
    qth_quantile = torch.quantile(stft_data, quantile)
    print(f"{quantile}th-quantile: ", qth_quantile)
    # thresholding
    stft_data = (stft_data > qth_quantile)
    return stft_data.permute(0, 2, 1), qth_quantile

def data2spadeformat(stft_data):
	data = []
	for i in range(stft_data.shape[0]):
		for j in range(stft_data.shape[1]):
			feats_list = []
			for k in range(stft_data.shape[2]):
				if stft_data[i][j][k]:
					feats_list.append(k+1)
			# if len(feats_list) == 0:
			# 	feats_list.append('0')
			dlist = [i+1, j+1]
			dlist.append(feats_list)
			data.append(dlist)
	return data
      
def spade_run(datad, quant, file):
    print('Running SPADE')
    # print('data type: ', type(datad[0]))
    result = spade(data=datad, support=support)
    print(f'completed, check {file}')
    f = open(file, 'w')
    sys.stdout = f
    print("Quantile: ", quant)	
    print_result(result)
    sys.stdout = orig_stdout
    f.close()

if __name__ == "__main__":
	print(" Data description: Each row is calculated using 0.5 seconds sample of audio.\
			Parameters of stft: nperseg = 125, noverlap = 62 (or in ratio 0.5), window = 'hann' ")


	print("Spectral Roll-off: The frequency below which 80% of the magnitude distribution is concentrated")

	dataset = torch.load("./Data/manual_feats_stft_y_label.pth")

	manual_features, stft , y_label = dataset.tensors

	print("Shape manual features (rmse,spectral centroid, spectral roll-off) : ", manual_features.shape)
	print("Shape stft : ", stft.shape)
	print("Shape y_label : ", y_label.shape)
	print('unique labels: ', y_label.unique())
	all_labels = list(y_label.unique())
	for l in all_labels:
		print('*'*100)
		print("Frequent patterns for label ", l)
		stft_label = stft[y_label == l]
		feats, quant = data2features(stft_label)
		file = f'./Results/spade_data/data_label_{l}.txt'
		data = data2spadeformat(feats)
		spade_run(data, quant, file=file)
		print('*'*100)
