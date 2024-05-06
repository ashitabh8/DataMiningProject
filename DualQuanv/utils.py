import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset

from model import QCNN_ReLU_BN 
from layers import DualConvNew, SwitchBN2d
# from layers import DualQuanConv
import numpy as np

def get_num_quant_layers(model):
    num_quant_layers = 0
    for layer in model.children():
        if isinstance(layer, DualConvNew):
            num_quant_layers += 1
    return num_quant_layers

def set_bitwidths(model, config):

    num_quant_layers = get_num_quant_layers(model)

    assert len(config) == num_quant_layers, "Number of configurations does not match the number of quantized layers"

    config_idx = 0

    for layer in model.children():
        if isinstance(layer, DualConvNew):
            setattr(layer, 'bitwidth_pattern_band', config[config_idx][0])
            setattr(layer, 'bitwidth_non_pattern_band', config[config_idx][1])

            config_idx += 1

def get_random_configs(model, bitwidth_pattern, bitwidth_non_pattern_bands):
    # Expected output: [(pattern_bw, non_pattern_bw), ...]
    #bitwidth_pattern ex: 
    # bitwidth_non_pattern_bands = {4:0.5, 8:0.25, 12: 0.125, 16:0.125}
    #
    # bitwidth_probs_pattern_band  = {4:0.125, 8:0.125, 12: 0.25, 16:0.5}
    num_quant_layers = get_num_quant_layers(model)

     # Extract the bitwidths and their corresponding probabilities for the pattern band
    pattern_bitwidths = list(bitwidth_pattern.keys())
    pattern_probs = list(bitwidth_pattern.values())

    # Extract the bitwidths and their corresponding probabilities for the non-pattern band
    non_pattern_bitwidths = list(bitwidth_non_pattern_bands.keys())
    non_pattern_probs = list(bitwidth_non_pattern_bands.values())

    # Randomly select the bitwidths for the pattern band
    pattern_bitwidths = np.random.choice(pattern_bitwidths, num_quant_layers, p=pattern_probs)

    # Randomly select the bitwidths for the non-pattern band
    non_pattern_bitwidths = np.random.choice(non_pattern_bitwidths, num_quant_layers, p=non_pattern_probs)

    return list(zip(pattern_bitwidths, non_pattern_bitwidths))


def set_lower_upper_ratios_for_all_layers(model, lower_bound_ratio, upper_bound_ratio):
    for layer in model.children():
        if isinstance(layer, DualConvNew):
            layer.set_lower_upper_band_ratio(lower_bound_ratio, upper_bound_ratio)


def train_model(model, train_loader, val_loader,bitwidth_probs_pattern, bitwidth_non_pattern_bands, criterion, optimizer, num_configs_per_epoch, num_epochs=10, save_best=False, best_name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_validation_acc = 0.0
    training_acc_for_best_validation = 0.0

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        curr_bitwidths = [get_random_configs(model, bitwidth_probs_pattern, bitwidth_non_pattern_bands) for _ in range(num_configs_per_epoch)]

        # Iterate over data.
        for inputs, labels in train_loader:
            loss_per_config = []
            correct_preds_in_batch = []
            
            inputs, labels = inputs.to(device).float(), labels.to(device).float()

            # Zero the parameter gradients
            optimizer.zero_grad()

            for config in curr_bitwidths:
                set_bitwidths(model, config)
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    loss_per_config.append(loss.item())

                    correct_preds_current = (torch.sum(preds == labels.data)).to(torch.device("cpu")).item()
                    correct_preds_in_batch.append(correct_preds_current)
            optimizer.step()

            avg_loss = np.mean(loss_per_config)
            # correct_preds_in_batch.to(torch.device("cpu"))
            # breakpoint()
            avg_correct_preds = np.mean(correct_preds_in_batch)

            running_loss += avg_loss
            running_corrects += avg_correct_preds

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1} - Training loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.4f}')

        # Validation phase
        model.eval()  # Set model to evaluate mode

        val_loss = []
        val_corrects = []
        print(f" Size val_loader: {len(val_loader)}")

        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            loss_per_config = []
            correct_preds_in_batch = []
            
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            for config in curr_bitwidths:
                set_bitwidths(model, config)
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels.long())
                    loss_per_config.append(loss.item())

                    correct_preds_cirrent = (torch.sum(preds == labels.data)).to(torch.device("cpu")).item()
                    correct_preds_in_batch.append(correct_preds_cirrent)
            avg_loss = sum(loss_per_config) / (inputs.size(0) * val_loader.batch_size)
            avg_correct_preds_val = sum(correct_preds_in_batch) / (inputs.size(0) * val_loader.batch_size)

            val_loss.append(avg_loss)
            val_corrects.append(avg_correct_preds_val)


        print(f"len of val_corrects: {len(val_corrects)}")
        val_acc = np.mean(val_corrects)
        val_loss_total = np.mean(val_loss)
        print(f'Epoch {epoch}/{num_epochs - 1} - Validation loss: {val_loss_total:.4f}, accuracy: {val_acc:.4f}')
        if val_acc > best_validation_acc:
            if save_best:
                assert best_name is not None, "Please provide a name for the best model"
                torch.save(model.state_dict(), best_name)
            best_validation_acc = val_acc
            training_acc_for_best_validation = epoch_acc

    print(f"Best validation accuracy: {best_validation_acc:.4f}, Training accuracy for best validation: {training_acc_for_best_validation:.4f}")
    print('Training complete')
    return model

def test_model(model, test_dataset_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in test_dataset_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_predictions += inputs.size(0)

        avg_test_loss = test_loss / len(test_dataset_loader.dataset)
        print('Test Loss: {:.6f}\n'.format(avg_test_loss))

        test_accuracy = correct_predictions / total_predictions
        print('Test Accuracy: {:.2f}%\n'.format(test_accuracy * 100))

def normalize_dataset(dataset):
    inputs = [data[0] for data, _ in dataset]
    inputs_tensor = torch.stack(inputs)
    mean = inputs_tensor.mean(dim=0, keepdim=True)
    std = inputs_tensor.std(dim=0, keepdim=True)
    std[std == 0] = 1
    normalized_inputs_tensor = (inputs_tensor - mean) / std
    normalized_inputs_tensor = normalized_inputs_tensor.unsqueeze(1)

    labels = [label.item() for _, label in dataset]
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    normalized_dataset = TensorDataset(normalized_inputs_tensor, labels_tensor)
    return normalized_dataset
