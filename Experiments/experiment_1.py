from Utils import data_processing, helpers, basic_vehicle_object_setup
from Utils.Vehicle import Vehicle
import torch
from sklearn.mixture import GaussianMixture 
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pickle

def processing_steps(vehicle_obj):
    # processing_fns = [("downsample_torch_audio", helpers.downsample_torch_audio, {'orig_freq': 8000, 'target_freq': 1000, 'vehicle_obj': vehicle_obj}), 
    #                   ("a_law_quantize", helpers.a_law_quantize, {'bitwidth': 8})]
    processing_fns = [("downsample_torch_audio", helpers.downsample_torch_audio, {'orig_freq': 8000, 'target_freq': 1000, 'vehicle_obj': vehicle_obj})]

    vehicle_obj.apply_processing_to_data(processing_fns)


def setup_features(vehicle_objs):
    vehicle_1_audio_segment_length = vehicle_objs[0].record_file_data[0]['audio'].shape[-1]
    nperseg_input = int(vehicle_1_audio_segment_length/2)
    overlap_input = int(nperseg_input/2)

    for vehicle in vehicle_objs:

        vehicle.add_stft_values(nperseg = nperseg_input ,noverlap = overlap_input)
        vehicle.add_rmse_values()
        # breakpoint()
        vehicle.add_label_silence(rmse_threshold = 84)
        # breakpoint()

def normalize_tensor(tensor):
    min = tensor.min()
    max = tensor.max()

    return (tensor - min) / (max - min)


def setup_data_in_list(vehicle_objs, gmm_num_components = 4):

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
        for i in range(len(vehicle)):
            # num_columns = vehicle.record_file_data[i]['stft'][2].shape[1]
            # Initialize an empty tuple to collect single-columned tensors
            stft_columns = ()
            
            # Iterate over each column in stft[2]
            for col in range(num_columns_stft):
                # Convert the column to a single-columned tensor and add it to the tuple
                # breakpoint()
                col_tensor = torch.tensor(vehicle.record_file_data[i]['stft'][2][:, col]).unsqueeze(1)
                col_tensor = torch.abs(col_tensor)
                col_tensor = normalize_tensor(col_tensor)

                freq_tensor = torch.tensor(vehicle.record_file_data[i]['stft'][0]).unsqueeze(1)
                freq_tensor = normalize_tensor(freq_tensor)

                combined_tensor = torch.cat((col_tensor, freq_tensor), dim=1)
                # if vehicle.vehicle_name == 'mustang0528_rs3':
                #     breakpoint()
                breakpoint()
                gmm_model.fit(combined_tensor)
                gmm_means = gmm_model.means_
                # convert numpy array to tensor
                gmm_means = torch.from_numpy(gmm_means)
                gmm_covariances = gmm_model.covariances_


                # breakpoint()
                stft_columns += (gmm_means, gmm_covariances)

            data.append((*stft_columns, vehicle.record_file_data[i]['rmse']
                         , vehicle.record_file_data[i]['silence'], one_hot_encoded[vehicle.vehicle_name]))
            # breakpoint()
        dataset_each_vehicle[vehicle.vehicle_name] = data
    # breakpoint()

    first_data = dataset_each_vehicle[vehicle_names[0]]
    total_num_rows = 0
    for vehicle in vehicle_objs:
        total_num_rows += len(dataset_each_vehicle[vehicle.vehicle_name])

    final_num_columns = (first_data[0][0].shape[0] * first_data[0][0].shape[1])*num_columns_stft + 1 + 1 + 1 +1

    # create tensor zeros
    final_data = torch.zeros((total_num_rows, final_num_columns))
    final_target_data = torch.zeros((total_num_rows, len(one_hot_encoded[vehicle_names[0]])))


    # fill in tensor
    row_index = 0
    for vehicle in vehicle_objs:
        for data in dataset_each_vehicle[vehicle.vehicle_name]:
            columns_for_means_and_covariances = 0
            offset = 0
            for i in range(num_columns_stft):
                # first element gmm_means reshape to 1d
                breakpoint()
                one_d_means = data[2*i].reshape(-1)
                # final_data[row_index, i*one_d_means.shape[0]:(i+1)*one_d_means.shape[0]] = one_d_means
                # # first element gmm_covariances reshape from 1d to number 
                # final_data[row_index, (i+1)*one_d_means.shape[0]] = data[2*i+1]


                final_data[row_index, offset:offset+one_d_means.shape[0]] = one_d_means
                offset += one_d_means.shape[0]  # Update the offset after adding 'one_d_means'

                # Add the single element and update the offset by 1 to account for it
                final_data[row_index, offset] = data[2*i+1]
                offset += 1
                columns_for_means_and_covariances += len(one_d_means) + 1
            # breakpoint()

            # add rmse
            # breakpoint()
            final_data[row_index, columns_for_means_and_covariances] = data[2*num_columns_stft]

            if data[-2] == True:
                final_data[row_index, columns_for_means_and_covariances+1] = 1
            else:
                final_data[row_index, columns_for_means_and_covariances+1] = 0
            # breakpoint()
            final_target_data[row_index, :] = torch.tensor(one_hot_encoded[vehicle.vehicle_name])
            row_index += 1

    # perform GMM on stft
    # breakpoint()
    # num_time_components_in_stft = vehicle_objs[0].record_file_data[0]['stft'][2].shape[1]
    # num_freq_components_in_stft = vehicle_objs[0].record_file_data[0]['stft'][2].shape[0]

    # print(f"num_time_components_in_stft: {num_time_components_in_stft}")
    # print(f"num_freq_components_in_stft: {num_freq_components_in_stft}")


    return final_data, final_target_data, column_names
    # breakpoint()


# Define the neural network
# class SimpleNN(nn.Module):
#     def __init__(self):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(20, 64)  # 20 features as input
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, 32)
#         self.fc4 = nn.Linear(32,16)
#         self.fc5 = nn.Linear(16, 2)  # 2 outputs for the two targets
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
#         x = torch.relu(self.fc4(x))
#         x = self.fc5(x)
#         return x

class SimpleNN(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # 20 features as input
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 2)  # 2 outputs for the two targets
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
        x = self.fc5(x)  # No batchnorm or dropout after the last layer
        return x


if __name__ == "__main__":
    names = ['tesla_rs3', 'mustang0528_rs3']
    vehicle_objs = basic_vehicle_object_setup.get_vehicle_objects(names, processing_duration=1)

    for vehicle in vehicle_objs:
        processing_steps(vehicle)

    setup_features(vehicle_objs)

    for vehicle in vehicle_objs:
        vehicle.print_state_processing()
    # if pickle_data:
    final_data, final_target_data, column_names = setup_data_in_list(vehicle_objs, gmm_num_components = 7)
    print(f"final_data shape: {final_data.shape}")
    print(f"final_target_data shape: {final_target_data.shape}")
    print(f"column_names: {column_names}")
    #
    # # Save the data to pickle files
    with open('final_data.pkl', 'wb') as f:
        pickle.dump(final_data, f)

    with open('final_target_data.pkl', 'wb') as f:
        pickle.dump(final_target_data, f)
    #
    # breakpoint()
    #
    # exit()
    # with open('final_data.pkl', 'rb') as f:
    #     final_data = pickle.load(f)
    #
    # with open('final_target_data.pkl', 'rb') as f:
    #     final_target_data = pickle.load(f)

    num_columns_final_data = final_data.shape[1]
    print(f"num_columns_final_data: {num_columns_final_data}")

    # remove all rows with silence == 1 from final_data and final_target_data

    # final_data = final_data[final_data[:, -1] == 0]
    # final_target_data = final_target_data[final_data[:, -1] == 0]
    # Create a mask for rows in final_data that meet your condition
    # breakpoint()
    mask = final_data[:, -1] == 0

# Apply this mask to final_data to filter rows
    final_data_filtered = final_data[mask]

    # normalize the 19th column
    # breakpoint()
    final_data_filtered[:, -2] = normalize_tensor(final_data_filtered[:, -2])

# Apply the same mask to final_target_data to ensure consistency
    final_target_data_filtered = final_target_data[mask]



    # Convert data to PyTorch tensors
    X = torch.Tensor(final_data)
    y = torch.Tensor(final_target_data)
    # Check for GPU availability and set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

# Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)


    # Initialize the network, loss function, and optimizer
    model = SimpleNN(num_columns_final_data).to(device)
    criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
    optimizer = optim.Adam(model.parameters(), lr=0.0001)


    # Train the network
    num_epochs = 100
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, torch.max(labels, 1)[1])
            # breakpoint()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')


        # print(vehicle.get_energy_stats())
