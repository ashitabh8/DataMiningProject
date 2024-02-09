from Utils import data_processing, helpers
from Utils.Vehicle import Vehicle
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
import numpy as np
from scipy.signal import spectrogram





if __name__ == "__main__":
    # vehicle = Vehicle('mustang0528_rs3')
    vehicle = Vehicle('motor_rs3')
    data_points = data_processing.get_file_paths('motor_rs3')
    vehicle.add_data_files(data_points)
    vehicle.build_record_file_data()
    processing_fns = [("downsample_torch_audio", helpers.downsample_torch_audio, {'orig_freq': 8000, 'target_freq': 1000,'vehicle_obj':vehicle })]

    vehicle.apply_processing_to_data(processing_fns)
    print(f"updated sampling rate audio: {vehicle.original_sampling_audio_rate}")

    vehicle.plot_rmse_energy_both_normalized(duration = 0.1, overlay=True)




