'''
In this experiment, we will only use the means of the features to train the model.
'''
from Utils import data_processing, helpers, basic_vehicle_object_setup
from Utils.Vehicle import Vehicle
import torch
from sklearn.mixture import GaussianMixture 
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
from tqdm import tqdm



def processing_steps(vehicle_obj):
    # processing_fns = [("downsample_torch_audio", helpers.downsample_torch_audio, {'orig_freq': 8000, 'target_freq': 1000, 'vehicle_obj': vehicle_obj}), 
    #                   ("a_law_quantize", helpers.a_law_quantize, {'bitwidth': 8})]
    processing_fns = [("downsample_torch_audio", helpers.downsample_torch_audio, {'orig_freq': 8000, 'target_freq': 1000, 'vehicle_obj': vehicle_obj})]

    vehicle_obj.apply_processing_to_data(processing_fns)


def setup_features(vehicle_objs):
    vehicle_1_audio_segment_length = vehicle_objs[0].record_file_data[0]['audio'].shape[-1]
    nperseg_input = int(vehicle_1_audio_segment_length/4)
    overlap_input = int(nperseg_input/2)

    for vehicle in vehicle_objs:

        vehicle.add_stft_values(nperseg = nperseg_input ,noverlap = overlap_input)
        vehicle.add_rmse_values()
        # breakpoint()
        vehicle.add_label_silence(rmse_threshold = 80)
        # breakpoint()

def normalize_tensor(tensor):
    min = tensor.min()
    max = tensor.max()

    return (tensor - min) / (max - min)

def setup_data_in_list(vehicle_objs, gmm_num_components = 7):
    vehicle_names = []
    for vehicle in vehicle_objs:
        vehicle_names.append(vehicle.vehicle_name)
    # Create a mapping from vehicle names to indices
    name_to_index = {name: index for index, name in enumerate(vehicle_names)}

    # Initialize an empty list to hold the one-hot encoded vectors
    one_hot_encoded = {}
    # Iterate over each vehicle name in the list
    for name in vehicle_names:
        # Create a vector of zeros with the same length as the number of unique vehicle names
        encoding = [0] * len(vehicle_names)
        # Set the position corresponding to the current vehicle name to 1
        encoding[name_to_index[name]] = 1
        # Append the one-hot encoded vector to the list
        one_hot_encoded[name] = encoding


    
    dataset_each_vehicle = {}
    num_columns_stft = vehicle_objs[0].record_file_data[0]['stft'][2].shape[1]
    # column_names = ["stft_" + str(i) for i in range(num_columns)]
    column_names = []
    for i in range(num_columns_stft):
        column_names.append("gmm_means_" + str(i))
        column_names.append("gmm_covariances_" + str(i))
    column_names.append("rmse")
    column_names.append("silence")
    column_names.append("one_hot_label")
    gmm_model = GaussianMixture(n_components=gmm_num_components, covariance_type='full', random_state=0)
    for vehicle in vehicle_objs:
        data = []


        for i in tqdm(range(len(vehicle)), desc='Processing Vehicles'):
            if vehicle.record_file_data[i]['silence'] == 1:
                continue

            stft_columns = ()
            
            # Assuming num_columns_stft is defined earlier in your code
            for col in range(num_columns_stft):
                col_tensor = torch.tensor(vehicle.record_file_data[i]['stft'][2][:, col]).unsqueeze(1)
                col_tensor = torch.abs(col_tensor)
                col_tensor = normalize_tensor(col_tensor)

                freq_tensor = torch.tensor(vehicle.record_file_data[i]['stft'][0]).unsqueeze(1)
                freq_tensor = normalize_tensor(freq_tensor)

                combined_tensor = torch.cat((col_tensor, freq_tensor), dim=1)
                gmm_model.fit(combined_tensor)
                gmm_means = gmm_model.means_
                gmm_means = gmm_means[gmm_means[:, 1].argsort()]
                gmm_means = gmm_means.reshape(-1)
                stft_columns += (gmm_means,)

            data.append(stft_columns)
        # for i in range(len(vehicle)):
        #     # num_columns = vehicle.record_file_data[i]['stft'][2].shape[1]
        #     # Initialize an empty tuple to collect single-columned tensors
        #     if vehicle.record_file_data[i]['silence'] == 1:
        #         continue
        #     stft_columns = ()
        #     
        #     # Iterate over each column in stft[2]
        #     for col in range(num_columns_stft):
        #         # Convert the column to a single-columned tensor and add it to the tuple
        #         # breakpoint()
        #         col_tensor = torch.tensor(vehicle.record_file_data[i]['stft'][2][:, col]).unsqueeze(1)
        #         col_tensor = torch.abs(col_tensor)
        #         col_tensor = normalize_tensor(col_tensor)
        #
        #         freq_tensor = torch.tensor(vehicle.record_file_data[i]['stft'][0]).unsqueeze(1)
        #         freq_tensor = normalize_tensor(freq_tensor)
        #
        #         combined_tensor = torch.cat((col_tensor, freq_tensor), dim=1)
        #         # if vehicle.vehicle_name == 'mustang0528_rs3':
        #         #     breakpoint()
        #         gmm_model.fit(combined_tensor)
        #         gmm_means = gmm_model.means_
        #         gmm_means = gmm_means[gmm_means[:,1].argsort()]
        #         gmm_means = gmm_means.reshape(-1)
        #         # breakpoint()
        #         stft_columns += (gmm_means,)
        #
        #     data.append(stft_columns)
        dataset_each_vehicle[vehicle.vehicle_name] = {'data': data, 'label': one_hot_encoded[vehicle.vehicle_name]}


    total_num_rows = 0
    for key,value in dataset_each_vehicle.items():
        total_num_rows += len(value['data'])
    total_num_columns = gmm_num_components * 2 * num_columns_stft
    print("Total number of rows: ", total_num_rows)
    print("Total number of columns: ", total_num_columns)
    
    final_data = torch.zeros((total_num_rows, total_num_columns))
    final_target_data = torch.zeros((total_num_rows, len(one_hot_encoded[vehicle_names[0]])))
    

    row_index = 0
    for vehicle in vehicle_objs:
        for data in dataset_each_vehicle[vehicle.vehicle_name]['data']:
            for i in range(num_columns_stft):
                one_d_means = data[i]
                # breakpoint()
                final_data[row_index, i*gmm_num_components*2:(i+1)*gmm_num_components*2] = torch.tensor(one_d_means)
            final_target_data[row_index, :] = torch.tensor(one_hot_encoded[vehicle.vehicle_name])
            row_index += 1

    print(f"Row index: {row_index}")
    assert row_index == total_num_rows, "Row index does not match total number of rows"
    return final_data, final_target_data
    


class SimpleNN(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 500)  # 20 features as input
        self.bn1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 800)
        self.bn2 = nn.BatchNorm1d(800)
        self.fc3 = nn.Linear(800, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc6 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.fc7 = nn.Linear(64, 32)
        self.bn7 = nn.BatchNorm1d(32)
        self.fc8 = nn.Linear(32, 8)
        self.bn8 = nn.BatchNorm1d(8)
        self.last_layer = nn.Linear(8, 2)  # 2 outputs for the two targets
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn5(self.fc5(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn6(self.fc6(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn7(self.fc7(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn8(self.fc8(x)))
        x = self.dropout(x)
        x = self.last_layer(x)  # No batchnorm or dropout after the last layer
        # x = nn.functional.softmax(x, dim=1)
        return x

def get_train_test_loader(final_data, target_data, test_size = 0.2):
    class_labels = np.argmax(target_data, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(final_data, target_data, test_size=test_size, random_state=42, stratify=class_labels)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)
    return train_loader, test_loader


# def train_basic(model, criterion, optimizer, train_loader, num_epochs = 100):
#     num_epochs = num_epochs
#     for epoch in range(num_epochs):
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             breakpoint()
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             # breakpoint()
#             loss = criterion(outputs, torch.max(labels, 1)[1])
#             # breakpoint()
#             loss.backward()
#             optimizer.step()
#         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')


def train_basic(model, criterion, optimizer, train_loader, num_epochs=100):
    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        # Initialize a variable to accumulate the loss over the epoch
        # epoch_loss = 0.0
        
        # Wrap the train_loader with tqdm for a progress bar
        for inputs, labels in tqdm(train_loader, desc='Batches', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            # Assuming labels are one-hot encoded and need to be converted to class indices
            labels_indices = torch.max(labels, 1)[1]
            
            loss = criterion(outputs, labels_indices)
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            # epoch_loss += loss.item()

        # Print average loss for the epoch
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')



if __name__ == "__main__":

    names = ['tesla_rs3', 'mustang0528_rs3']
    vehicle_objs = basic_vehicle_object_setup.get_vehicle_objects(names, processing_duration=0.5)

    for vehicle in vehicle_objs:
        processing_steps(vehicle)

    setup_features(vehicle_objs)

    for vehicle in vehicle_objs:
        vehicle.print_state_processing()
    # if pickle_data:
    final_data, final_target_data = setup_data_in_list(vehicle_objs, gmm_num_components = 4)
    X = torch.Tensor(final_data)
    y = torch.Tensor(final_target_data)
    # Check for GPU availability and set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

# Split the dataset into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
#
# # Create TensorDatasets and DataLoaders
#     train_dataset = TensorDataset(X_train, y_train)
#     test_dataset = TensorDataset(X_test, y_test)
#
#     train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
#     test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)
    train_loader, test_loader = get_train_test_loader(final_data, final_target_data, test_size = 0.2)


    # Initialize the network, loss function, and optimizer
    model = SimpleNN(final_data.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_basic(model, criterion, optimizer,train_loader, num_epochs = 100)


    # Train the network








