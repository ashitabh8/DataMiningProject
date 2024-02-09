from Utils import data_processing, helpers
from Utils.Vehicle import Vehicle
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
import numpy as np
from scipy.signal import spectrogram


'''
get average audio energy for each record file
duration: duration of each split in seconds
'''
def get_silence_samples_from_data(vehicle: Vehicle, duration = 0.25):
    record_keys = vehicle.get_keys()
    average_audio_energy = {}
    
    for key in record_keys:
        average_audio_energy[key] = vehicle.average_audio_energy(key, duration_split=duration)

    return average_audio_energy


def get_cumalitive_average_energy(vehicle: Vehicle, duration = 0.25):
    record_keys = vehicle.get_keys()
    average_audio_energy = {}
    for key in record_keys:
        average_audio_energy[key] = vehicle.average_audio_energy(key, duration_split=duration)
    return average_audio_energy


'''
averace_audio_energy: dict of average audio energy for each record file
each record is tuple of average audio energy for each split
'''
def plot_average_audio_energy(average_audio_energy, vehicle_name):
    # Create an empty figure
    fig = go.Figure()
    X_labels = []
    Y_labels = []

    for record, energy_entries in average_audio_energy.items():
        for i in range(len(energy_entries)):
            # print(f"index {i}")
            X_labels.append(f"{record}_split_{i}")
            Y_labels.append(energy_entries[i])


    # Add a trace to the plot
    fig.add_trace(go.Scatter(x=X_labels, y=Y_labels, mode='markers'))


    # Update the layout of the figure
    fig.update_layout(
        title=f"Average Audio Energy (Root Mean Square) per Record Split for {vehicle_name}",
        xaxis_title="Split Number",
        yaxis_title="Average Audio Energy",
        template="plotly_dark"  # You can change the template as per your preference
    )

    # Show the plot
    fig.show()

def build_and_plot_average_audio_energy_plot(*vehicle_names):
    # list of args
    vehicle_objs = []
    all_vehicle_average_audio_energy = {}
    for vehicle in vehicle_names:
        vehicle_objs.append(Vehicle(vehicle))
        data_points = data_processing.get_file_paths(vehicle)
        vehicle_objs[-1].add_data_files(data_points)
        vehicle_objs[-1].build_record_file_data()
        all_vehicle_average_audio_energy[vehicle] = get_cumalitive_average_energy(vehicle_objs[-1])

    
    num_vehicles = len(vehicle_names)
    fig = make_subplots(rows=num_vehicles, cols=1, subplot_titles=vehicle_names)

    for idx, vehicle in enumerate(vehicle_names, start=1):
        average_audio_energy = all_vehicle_average_audio_energy[vehicle]
        X_labels = []
        Y_labels = []

        for record, energy_entries in average_audio_energy.items():
            for i, energy_entry in enumerate(energy_entries):
                X_labels.append(f"{record}_split_{i+1}")
                Y_labels.append(energy_entry)

        # Add trace to the corresponding subplot
        fig.add_trace(go.Scatter(x=X_labels, y=Y_labels, mode='markers', name=vehicle), row=idx, col=1)

    # Update the layout of the figure
    fig.update_layout(
        title="Average Audio Energy per Vehicle and Record Split",
        xaxis_title="Record and Split Number",
        yaxis_title="Average Audio Energy",
        template="plotly_dark",  # You can change the template as per your preference
        height=600 * num_vehicles,  # Adjust the height based on the number of vehicles
        showlegend=False  # To avoid repeated legends, as vehicle names are in titles
    )


    fig.show()


def process_data_and_plot(vehicle_obj: Vehicle, processing_function_pipline = []):

    vehicle_obj.apply_processing_to_data(processing_function_pipline)
    vehicle.print_state_processing()
    vehicle.plot_rmse_energy()



    pass




if __name__ == "__main__":
    # build_and_plot_average_audio_energy_plot('tesla_rs3', 'mustang0528_rs3', 'motor_rs3')
    vehicle = Vehicle('tesla_rs3')
    data_points = data_processing.get_file_paths('tesla')
    vehicle.add_data_files(data_points)
    vehicle.build_record_file_data()
    processing_fns = [("downsample_torch_audio", helpers.downsample_torch_audio, {'orig_freq': 8000, 'target_freq': 1000})]
    process_data_and_plot(vehicle, processing_fns)
    




    # mustang_vehicle = Vehicle('mustang0528_rs3')
    # mustang_data_points = data_processing.get_file_paths('mustang0528_rs3')
    # mustang_vehicle.add_data_files(mustang_data_points)
    # print("Number of Mustang data points: ", len(mustang_data_points))
    # print("Number of Tesla data points: ", len(data_points))
    # print("First 5 data points: ", data_points[:5])
    # print("First 5 mustang data points: ", mustang_data_points[:5])
    # vehicle.add_data_files(data_points)
    # vehicle.build_record_file_data()
    # mustang_vehicle.build_record_file_data()
    # # print("Length of mustang vehicle object: ", len(mustang_vehicle))
    # average_audio_energy: dict = get_silence_samples_from_data(vehicle)
    # average_audio_energy_mustang :dict = get_silence_samples_from_data(mustang_vehicle)
    # # plot_average_audio_energy(average_audio_energy, vehicle_name='tesla_rs3')
    # plot_average_audio_energy(average_audio_energy_mustang, vehicle_name='mustang0528_rs3')
