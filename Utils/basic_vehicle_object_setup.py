from Utils import data_processing
from Utils.Vehicle import Vehicle


# def get_vehicle_object(name, processing_duration = 0.25):
#     vehicle = Vehicle(name)
#     data_points = data_processing.get_file_paths(name)
#     vehicle.add_data_files(data_points)
#     vehicle.build_record_file_data(processing_duration=processing_duration,duration_in_file=2, debug_info= False)
#     return vehicle

def get_vehicle_objects(names, processing_duration=0.25):
    vehicles = []  # Initialize an empty list to hold vehicle objects
    for name in names:  # Iterate over each name in the list
        vehicle = Vehicle(name)  # Create a new Vehicle object
        data_points = data_processing.get_file_paths(name)  # Get file paths for the vehicle
        vehicle.add_data_files(data_points)  # Add data files to the vehicle
        vehicle.build_record_file_data(processing_duration=processing_duration, duration_in_file=2, debug_info=False)  # Build record file data
        vehicles.append(vehicle)  # Add the vehicle object to the list of vehicles
    return vehicles


