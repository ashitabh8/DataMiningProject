'''
We will use CNN with no dense layer to classify the audio data.
'''
from Utils import data_processing, helpers, basic_vehicle_object_setup
from Utils.Vehicle import Vehicle
import torch
# from sklearn.mixture import GaussianMixture 
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
from tqdm import tqdm
from Utils import train_test_funcs
from Models.deep_cnn import ConvNeuralNetBasic, CNN_nodeep, CNN_nodeep_v2
from Utils.memory_tracking_funcs import model_parameters_memory_size


import math

def calculate_number_of_windows(N, nperseg, noverlap = 0.5):
    # Ensure nperseg is greater than noverlap to avoid division by zero or negative window step
    assert nperseg > noverlap, "nperseg must be greater than noverlap"
    overlap_size = nperseg*noverlap

    # Calculate the number of windows
    num_windows = 1 + math.floor((N - nperseg) / (nperseg - overlap_size))
    
    return num_windows

def processing_steps(vehicle_obj):
    # processing_fns = [("downsample_torch_audio", helpers.downsample_torch_audio, {'orig_freq': 8000, 'target_freq': 1000, 'vehicle_obj': vehicle_obj}), 
    #                   ("a_law_quantize", helpers.a_law_quantize, {'bitwidth': 8})]
    processing_fns = [("downsample_torch_audio", helpers.downsample_torch_audio, {'orig_freq': 8000, 'target_freq': 1000, 'vehicle_obj': vehicle_obj})]

    vehicle_obj.apply_processing_to_data(processing_fns)

def apply_hann_window(signal: np.ndarray):
    N = signal.shape[-1]
    # breakpoint()
    hann_window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N - 1))
    return signal * hann_window

def add_stft_usingfft(vehicle_obj: Vehicle, nperseg, noverlap = 0.5):
    step_size = int(nperseg * noverlap)
    num_windows = calculate_number_of_windows(vehicle_obj.record_file_data[0]['audio'].shape[1], nperseg, noverlap)
    # print(f"Number of windows: {num_windows}")
    N = nperseg
    # f = np.linspace(0, vehicle_obj.original_sampling_audio_rate / 2, N // 2, endpoint=True)
    # F = np.fft.fftfreq(N, 1 / vehicle_obj.original_sampling_audio_rate)[:N // 2]
    F = np.arange(0, nperseg//2 + 1) * vehicle_obj.original_sampling_audio_rate / nperseg
    F = torch.tensor(F)

    num_files = len(vehicle_obj)
    complete_data = torch.zeros(num_files, nperseg//2 + 1, num_windows)
    for record in range(num_files):
        # breakpoint()
        audio_signal = vehicle_obj.record_file_data[record]['audio']
        audio_signal = apply_hann_window(audio_signal)
        assert audio_signal.shape[1] % nperseg == 0, "The audio signal length is not a multiple of nperseg"

        Zxx = torch.zeros((num_windows, nperseg//2 + 1))
        current_window = 0
        for i in range(num_windows):
            start_idx = i * step_size
            end_idx = start_idx + nperseg
            assert end_idx <= audio_signal.shape[1], "The end index is greater than the audio signal length"
            audio_segment = audio_signal[:, start_idx:end_idx]
            Fxx = torch.fft.fft(audio_segment)
            Fxx_half = Fxx[:, :nperseg//2 + 1]
            Fxx_half = torch.abs(Fxx_half)
            Zxx[current_window, :] = Fxx_half
            current_window += 1

        Zxx = Zxx.T

        vehicle_obj.record_file_data[record]['stft'] = Zxx
        # complete_data[record, :, :] = Zxx
    # return complete_data

def get_train_test_dataloader(vehicle_objs, one_hot_encodings, remove_silence = True,
                              test_size = 0.2, batch_size = 32):
    num_rows = vehicle_objs[0].record_file_data[0]['stft'].shape[0]
    num_columns = vehicle_objs[0].record_file_data[0]['stft'].shape[1]

    total_num_samples = 0
    for vehicle in vehicle_objs:
        for idx in range(len(vehicle)):
            if remove_silence:
                if not vehicle.record_file_data[idx]['silence']:
                    total_num_samples += 1

    complete_data = torch.zeros(total_num_samples, 1, num_rows, num_columns)
    target_data = torch.zeros(total_num_samples, len(one_hot_encodings))

    curr_idx = 0
    for vehicle in vehicle_objs:
        for idx in range(len(vehicle)):
            if remove_silence:
                if not vehicle.record_file_data[idx]['silence']:
                    complete_data[curr_idx,0, :, :] = vehicle.record_file_data[idx]['stft']
                    target_data[curr_idx, :] = torch.tensor(one_hot_encodings[vehicle.vehicle_name])
                    curr_idx += 1

    class_labels = torch.argmax(target_data, dim=1)
    unique_labels, frequency = torch.unique(class_labels, return_counts=True)
    print(f"Unique labels: {unique_labels}, Frequency: {frequency}")
    print(f"Complete data shape: {complete_data.shape}, Target data shape: {target_data.shape}")
    # Split the complete data into training+validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        complete_data, target_data, 
        test_size=test_size, random_state=42, stratify=class_labels
    )

# Further split the training+validation set into actual training and validation sets
# Assuming you want to use 20% of the training data for validation
    val_size = 0.2
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=val_size, random_state=42, stratify=y_train_val
    )

# Create TensorDatasets for training, validation, and test sets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

# Create DataLoaders for the datasets
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader





    



if __name__ == "__main__":

    # names = ['tesla_rs3', 'motor_rs3', 'mustang0528']
    # names = ['tesla_rs3', 'motor_rs3']
    # names = ['tesla_rs3', 'Polaris0150pm_rs1']
    # names = ['tesla_rs3', 'Polaris0150pm_rs1', 'motor_rs3']
    names = ['tesla_rs3', 'Silverado0255pm_rs1', 'motor_rs3']
    vehicle_objs = basic_vehicle_object_setup.get_vehicle_objects(names, processing_duration=0.5)


    # breakpoint()

    for vehicle in vehicle_objs:
        processing_steps(vehicle)

    for vehicle in vehicle_objs:
        vehicle.add_label_silence(rmse_threshold = 84)

    one_hot_encodings = helpers.get_one_hot_encoding(names)

    for vehicle in vehicle_objs:
        add_stft_usingfft(vehicle, 125, 0.5)

    # complete_data_stft_vehicles = [ stft_usingfft(vehicle, 250, 0.5) for vehicle in vehicle_objs]
    print(f"One hot encodings: {one_hot_encodings}")


    train_loader,val_loader, test_loader = get_train_test_dataloader(vehicle_objs, one_hot_encodings)

    model = CNN_nodeep_v2(num_classes = len(one_hot_encodings))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_test_funcs.train_model(model, train_loader, val_loader,criterion, optimizer, num_epochs=100)

    memory_size = model_parameters_memory_size(model, in_bytes=False)

    print(f"Memory size of the model: {memory_size} KB")


    # for inputs, labels in tqdm(train_loader, desc='Batches', leave=False):
    #     inputs, labels = inputs, labels
    #     breakpoint()

