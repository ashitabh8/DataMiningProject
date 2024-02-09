import libros, debug_info= True)
from globals import DATA_DIR_PATH, DATA_PREPROCESSING_CONFIGS_PATH
import os, argparse

import json



def get_file_paths(vehicle_type = 'tesla'):
    file_names = os.listdir(DATA_DIR_PATH)
    vehicle_data_file_names = []

    for file_name in file_names:
        # Check if file_name contains vehicle_type
        if vehicle_type in file_name:
            # If yes, add file_name to dataset_dict[vehicle_type]
            vehicle_data_file_names.append(file_name)
    return vehicle_data_file_names



def import_data(data_processing_config):
    print("Type: ", type(data_processing_config))
    print("Data processing config: ", data_processing_config)
    # Read file data_processing_config


    with open(data_processing_config, 'r') as f:
        # Read the file
        data = f.read()
        # Convert string to dictionary
        data_dict = json.loads(data)

        # Ensure "vehicle_type" is a list of strings if it exists
        if 'vehicle_type' in data_dict:
            if not isinstance(data_dict['vehicle_type'], list):
                data_dict['vehicle_type'] = [data_dict['vehicle_type']]

        # The data_dict now has "vehicle_type" as a list of strings
    print("Data dictionary: ", data_dict)
    vehicle_dict = data_dict['vehicle_type']

    # Ensure vehicle_dict is a list of strings
    # Ensure vehicle_dict is not empty
    assert isinstance(vehicle_dict, list), "vehicle_dict should be a list"
    assert all(isinstance(item, str) for item in vehicle_dict), "All items in vehicle_dict should be strings"
    assert len(vehicle_dict) > 0, "vehicle_dict should not be empty"

    # get all files in DATA_DIR_PATH
    file_names = os.listdir(DATA_DIR_PATH)
    
    dataset_dict = {}


    for vehicle_type in vehicle_dict:
        dataset_dict[vehicle_type] = []
        for file_name in file_names:
            # Check if file_name contains vehicle_type
            if vehicle_type in file_name:
                # If yes, add file_name to dataset_dict[vehicle_type]
                dataset_dict[vehicle_type].append(file_name)



def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process data using a configuration file.")
    parser.add_argument('--data_processing_config', required=True, help='JSON configuration file name')

    # Parse arguments
    args = parser.parse_args()
    config_file_name = args.data_processing_config
    print("Config file name: ", config_file_name)
    full_path = DATA_PREPROCESSING_CONFIGS_PATH + config_file_name
    print("Full path: ", full_path)

    # Import data using the provided configuration file
    import_data(full_path)


if __name__ == "__main__":
    main()
    # import_data()
    # Import data from Data/individual_sample_times

    




