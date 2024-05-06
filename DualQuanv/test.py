import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset

from model import QCNN_ReLU_BN 
from layers import DualConvNew, SwitchBN2d
# from layers import DualQuanConv
import numpy as np

from utils import set_lower_upper_ratios_for_all_layers, train_model, test_model, normalize_dataset, set_bitwidths, get_random_configs

if __name__ == "__main__":
    data_dir = '/Users/utkarsh/cs512/data'
    test_file = data_dir + '/test_tensor_dataset_stft_3vehicles_pd_0_5_125_0_5.pth'

    test_dataset = torch.load(test_file)

    test_dataset = normalize_dataset(test_dataset)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)

    bitwidth_non_pattern_bands = {2: 0.1, 4: 0.1, 8: 0.4, 16: 0.4} 
    bitwidth_probs_pattern_band = {2: 0.5, 4: 0.3, 8: 0.1, 16: 0.1}
    lower_bound_ratio = 0.70
    upper_bound_ratio = 0.90

    # (pattern band bw, non-pattern band bw)
    # bitwidth_allocation = [(4,8), (8,4), (16,8), (4,8), (8,8)]
    file_name = '0.70_0.90_dual_quant_model_2_4_8_16.pth'
    running_model = QCNN_ReLU_BN(num_classes=3)

    print(f"Loading the best model")
    running_model.load_state_dict(torch.load(file_name))
    
    criterion = nn.CrossEntropyLoss()
    
    # Generate random bitwidth configurations using get_random_configs
    bitwidth_allocation = get_random_configs(running_model, bitwidth_probs_pattern_band, bitwidth_non_pattern_bands)

    print(f"Generated bitwidth allocation: {bitwidth_allocation}")

    set_bitwidths(running_model, bitwidth_allocation)
    set_lower_upper_ratios_for_all_layers(running_model, lower_bound_ratio, upper_bound_ratio)

    test_model(running_model, test_loader, criterion)




