from Utils import data_processing, helpers
from Utils import basic_vehicle_object_setup
from Utils.Vehicle import Vehicle

import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
import numpy as np
from scipy.signal import spectrogram
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture





if __name__ == "__main__":
    print("Running energy_comparision_vehicles.py")
    # tesla_obj = basic_vehicle_object_setup.get_vehicle_object("tesla_rs3", processing_duration=0.25)
    # mustang_obj = basic_vehicle_object_setup.get_vehicle_object("mustang0528_rs3", processing_duration=0.25)

    vehicle_objs = basic_vehicle_object_setup.get_vehicle_objects(['tesla_rs3', 'mustang0528_rs3'], processing_duration=0.5)
    processing_fns_tesla = [("downsample_torch_audio", helpers.downsample_torch_audio, {'orig_freq': 8000, 'target_freq': 1000,'vehicle_obj':vehicle_objs[0] }), ("a_law_quantize", helpers.a_law_quantize, {'bitwidth': 8})]
    processing_fns_mustang = [("downsample_torch_audio", helpers.downsample_torch_audio, {'orig_freq': 8000, 'target_freq': 1000,'vehicle_obj':vehicle_objs[1] }), ("a_law_quantize", helpers.a_law_quantize, {'bitwidth': 8})]
    vehicle_objs[0].apply_processing_to_data(processing_fns_tesla)
    vehicle_objs[1].apply_processing_to_data(processing_fns_mustang)
    vehicle_objs[0].show_energy_stats()
    vehicle_objs[1].show_energy_stats()




