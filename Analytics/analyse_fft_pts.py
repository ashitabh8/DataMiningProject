from Utils import data_processing, helpers, basic_vehicle_object_setup
from Utils.Vehicle import Vehicle
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.offline import init_notebook_mode, iplot
from plotly.subplots import make_subplots
import numpy as np
from scipy.signal import spectrogram
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import scipy.fftpack
import torch




def plot_fft(F, Zxx):
    # Initialize plotly notebook mode
    # init_notebook_mode(connected=True)

    # Calculate the magnitude of the FFT results
    # Zxx_magnitude = np.abs(Zxx)

    # Create a line plot of the FFT magnitude vs frequency
    trace = go.Scatter(
        x=F,
        y=Zxx,
        mode='lines',
        name='FFT Magnitude'
    )

    layout = go.Layout(
        title='FFT Magnitude Spectrum',
        xaxis=dict(title='Frequency (Hz)'),
        yaxis=dict(title='Magnitude'),
        hovermode='closest'
    )

    fig = go.Figure(data=[trace], layout=layout)

    # Plot the figure offline
    fig.show()

def plot_many_ffts(F, Zxx_list, labels):
    # Initialize plotly notebook mode (uncomment this if using in a notebook environment)
    # init_notebook_mode(connected=True)

    # Check if the length of Zxx_list and labels are the same
    assert len(Zxx_list) == len(labels), "The length of Zxx_list and labels must be the same"

    # Create a list to hold all the traces
    traces = []

    # Iterate over the Zxx_list and labels to create a trace for each FFT magnitude
    for Zxx, label in zip(Zxx_list, labels):
        # Create a line plot for the current FFT magnitude
        trace = go.Scatter(
            x=F,
            y=Zxx,
            mode='lines',
            name=label  # Use the corresponding label for the legend
        )
        traces.append(trace)

    # Define the layout
    layout = go.Layout(
        title='FFT Magnitude Spectrum',
        xaxis=dict(title='Frequency (Hz)'),
        yaxis=dict(title='Magnitude'),
        hovermode='closest'
    )

    # Create the figure with all the traces
    fig = go.Figure(data=traces, layout=layout)

    # Plot the figure
    fig.show()

# def plot_many_ffts(F, Zxx_list, labels):
#     pass

def dct_1d(signal):
    # Assuming signal is a PyTorch tensor
    return torch.tensor(scipy.fftpack.dct(signal.numpy(), type=2, norm='ortho'))


def normalize_tensor(tensor):
    min = tensor.min()
    max = tensor.max()

    return (tensor - min) / (max - min)

if __name__ == '__main__':
    # Load the data
    names = ['tesla_rs3', 'motor_rs3']
    
    # this function runs all the boiler plate code to import the data and create the vehicle objects
    # processing_duration is the time split each data point will carry
    # in each data file (.pt) there are 2 seconds audio ( 16000 samples, frequency = 8000 Hz)
    vehicle_objs = basic_vehicle_object_setup.get_vehicle_objects(names, processing_duration=0.25)

    # Vehicle Objects that casrry the data and helper functions
    tesla_obj:Vehicle = vehicle_objs[0]
    mustang_obj: Vehicle = vehicle_objs[1]

    breakpoint()

    # Downsample the data, using helper functions in Utils/helpers.py
    # you can pass a list of functions to apply to the data either inplace or return the final result as follows
    # [ (<fun_name: str>, <fun: callable>, <kwargs: dict>), ...]
    processing_fns = [("downsample_torch_audio", helpers.downsample_torch_audio, {'orig_freq': 8000, 'target_freq': 1000,'vehicle_obj':tesla_obj })
                      , ("a_law_quantize", helpers.a_law_quantize, {'bitwidth': 8})]

    processing_only_downsample = [("downsample_torch_audio", helpers.downsample_torch_audio, {'orig_freq': 8000, 'target_freq': 1000,'vehicle_obj':tesla_obj })]



    tesla_obj.apply_processing_to_data(processing_only_downsample)
    mustang_obj.apply_processing_to_data(processing_only_downsample)
    # breakpoint()


    # Manually going through the data
    # Access data through record_file_data attribute
    # which is a dictionary of all data where you can save the processed versions or any relevant data about each point

    trial_pt_num = 0
    pt_tesla = tesla_obj.record_file_data[trial_pt_num]

    print(f"Keys in the data: {pt_tesla.keys()}")
    # breakpoint()

    # for record_file_data[0:4] will be the file num 0, [5:9] will be file num 1 and so on
    print(f"File path of point: {pt_tesla['file_path']}")

    # Example of adding info about each pt in record_file_data
    tesla_obj.add_rmse_values()
    mustang_obj.add_rmse_values()
    print(f"Keys (with rmse) in the data: {pt_tesla.keys()}")

    # Labelling silence based on rmse values, now each pt has a label of silence based on the rmse value
    tesla_obj.add_label_silence(rmse_threshold=80)
    mustang_obj.add_label_silence(rmse_threshold=80)


    all_non_silence_pts = tesla_obj.get_all_non_silence_pts()
    all_non_silence_pts_mustang = mustang_obj.get_all_non_silence_pts()

    print(f"Number of non-silence points (Tesla) : {len(all_non_silence_pts)}")
    print(f"Number of non-silence points (Motor): {len(all_non_silence_pts_mustang)}")

    # calculating fft of a single point
    # the data is already downsampled and quantized
    # absolute = True, returns the absolute value of the fft else will return complex values
    analysing_fft_pt = 12
    f, Zxx = tesla_obj.calculate_fft_single_point(all_non_silence_pts[analysing_fft_pt], absolute= True)
    f_mustang, Zxx_mustang = mustang_obj.calculate_fft_single_point(all_non_silence_pts[analysing_fft_pt], absolute= True)
    Zxx_log = helpers.log_scaling(Zxx)
    Zxx_log_mustang = helpers.log_scaling(Zxx_mustang)
    f_normalized, Zxx_normalized = normalize_tensor(f), normalize_tensor(Zxx)

    # print(f"Shape of the fft: {Zxx.shape}")
    # print(f"Frequency bins: {f.shape}")
    #
    # print(f"First 5 frequency bins: {f[:5]}")
    # print(f"First 5 fft values: {Zxx[:5]}")

    # Uncomment to plot the fft
    # plot_fft(f, Zxx_log)
    # plot_many_ffts(f, [Zxx, Zxx_mustang], ['Tesla', 'Motor'])
    # plot_many_ffts(f, [Zxx_log, Zxx_log_mustang], ['Tesla', 'Mustang'])

    # Uncomment to plot the normalized fft
    # plot_fft(f_normalized, Zxx_normalized)

    # dct_transformed = dct_1d(tesla_obj.record_file_data[all_non_silence_pts[17]]['audio'])
    # print(f"Shape of the DCT: {dct_transformed.shape}")

    # Uncomment for FFT values for all non-silence points or all points
    tesla_obj.add_fft_values()
    breakpoint()



    # tesla_obj.add_gmms_for_fft(num_components= 4, covariance_type='full', random_state=42, only_non_silence=True)


    






