import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset

from model import QCNN_ReLU_BN, QCNN_DualQuant
from layers import DualConvNew, SwitchBN2d
# from layers import DualQuanConv
import numpy as np

from utils import set_lower_upper_ratios_for_all_layers, train_model, test_model, normalize_dataset

if __name__ == '__main__':
    data_dir = '/Users/utkarsh/cs512/data'
    train_file = data_dir + '/train_tensor_dataset_stft_3vehicles_pd_0_5_125_0_5.pth'
    test_file = data_dir + '/test_tensor_dataset_stft_3vehicles_pd_0_5_125_0_5.pth'
    val_file = data_dir + '/val_tensor_dataset_stft_3vehicles_pd_0_5_125_0_5.pth'

    train_dataset = torch.load(train_file)
    test_dataset = torch.load(test_file)
    val_dataset = torch.load(val_file)

    train_dataset = normalize_dataset(train_dataset)
    test_dataset = normalize_dataset(test_dataset)
    val_dataset = normalize_dataset(val_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=2)

    bitwidth_non_pattern_bands = {2: 0.5, 4: 0.3, 8: 0.2, 16: 0.0} 
    bitwidth_probs_pattern_band = {2: 0.0, 4: 0.2, 8: 0.3, 16: 0.5}

    running_model = QCNN_ReLU_BN(num_classes=3)
    bitwidth_opts = bitwidth_non_pattern_bands.keys()
    bw_str = '_'.join([str(bw) for bw in bitwidth_opts])
    model_name_file = f'.dual_quant_model_{bw_str}.pth'

    lower_bound_ratio = 0.70
    upper_bound_ratio = 0.95

    print(f"Model saving name: {model_name_file}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(running_model.parameters(), lr=0.001, weight_decay=0.0001)
    num_epochs = 75

    set_lower_upper_ratios_for_all_layers(running_model, lower_bound_ratio, upper_bound_ratio)

    # random_configs = get_random_configs(running_model, bitwidth_probs_pattern_band, bitwidth_non_pattern_bands)
    #
    # print(f"Randomly selected configurations: {random_configs}")
    # exit()


    best_model = train_model(running_model, train_loader, val_loader,bitwidth_probs_pattern_band,bitwidth_non_pattern_bands, criterion, optimizer, 5,
                             num_epochs, save_best=True, best_name=model_name_file)

    print("Loading the best model")
    testing_model = QCNN_ReLU_BN(num_classes=3)
    testing_model.load_state_dict(torch.load(model_name_file))

    print("Testing the model")
    test_model(testing_model, test_loader, criterion)
