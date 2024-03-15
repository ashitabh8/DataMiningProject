import torch


if __name__ == "__main__":
    print(" Data description: Each row is calculated using 0.5 seconds sample of audio.\
          Parameters of stft: nperseg = 125, noverlap = 62 (or in ratio 0.5), window = 'hann' ")


    print("Spectral Roll-off: The frequency below which 80% of the magnitude distribution is concentrated")

    dataset = torch.load("./Data/manual_feats_stft_y_label.pth")

    manual_features, stft , y_label = dataset.tensors

    print("Shape manual features (rmse,spectral centroid, spectral roll-off) : ", manual_features.shape)
    print("Shape stft : ", stft.shape)
    print("Shape y_label : ", y_label.shape)

