from Utils import data_processing, helpers
from Utils.Vehicle import Vehicle
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
import numpy as np
from scipy.signal import spectrogram
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture



def get_all_Zxx_of_vehicle(vehicle_obj: Vehicle, nperseg = 256, noverlap = 128):
    """
    Returns a list of all Zxx of a vehicle
    """
    Zxx_list = []
    for i in range(len(vehicle_obj.get_keys())):
        f, t, Zxx = vehicle_obj.calculate_stft_single_point(i, nperseg =nperseg, noverlap=noverlap)
        # breakpoint()
        Zxx_list.append(np.abs(Zxx))
    # Convert list of Zxx to numpy array
    Zxx_list = np.array(Zxx_list)
    return Zxx_list

def get_gmm_of_vehicle(Zxx, num_components:int):
    """
    Returns a Gaussian Mixture Model of a vehicle
    """
    # Create and fit the GMM
    gmm = GaussianMixture(n_components=num_components, random_state=0)
    gmm.fit(Zxx)
    return gmm


def plot_covariance_ellipse(mean, covariance):
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # Major and minor axes
    major_axis = np.sqrt(eigenvalues[0]) * 2
    minor_axis = np.sqrt(eigenvalues[1]) * 2

    # Angle for ellipse rotation
    angle = np.arctan2(eigenvectors[0,1], eigenvectors[0,0])

    # Parametric equation for ellipse
    t = np.linspace(0, 2*np.pi, 100)
    ellipse_x = mean[1] + major_axis * np.cos(t) * np.cos(angle) - minor_axis * np.sin(t) * np.sin(angle)
    ellipse_y = mean[0] + major_axis * np.cos(t) * np.sin(angle) + minor_axis * np.sin(t) * np.cos(angle)

    valid_indices = ellipse_y >= 0
    ellipse_x = ellipse_x[valid_indices]
    ellipse_y = ellipse_y[valid_indices]
    return go.Scatter(x=ellipse_x, y=ellipse_y, mode='lines', line=dict(color='green'), name='GMM Covariance')




def exp_1(Zxx_list):
    first_column_Zxx_0 = Zxx_list[4][:,0]
    len_column = len(first_column_Zxx_0)
    first_column_Zxx_0 = first_column_Zxx_0.reshape(-1,1)
    breakpoint()

    # Add column of 1,2,3,...,len_column
    first_column_Zxx_0 = np.hstack((first_column_Zxx_0, np.arange(1,len_column+1).reshape(-1,1)))
    breakpoint()

    # Scale the data to be between 0 and 1 each column separately

    # Scale the first column
    first_column_Zxx_0[:,0] = (first_column_Zxx_0[:,0] - np.min(first_column_Zxx_0[:,0])) / (np.max(first_column_Zxx_0[:,0]) - np.min(first_column_Zxx_0[:,0]))

    # Scale the second column
    first_column_Zxx_0[:,1] = (first_column_Zxx_0[:,1] - np.min(first_column_Zxx_0[:,1])) / (np.max(first_column_Zxx_0[:,1]) - np.min(first_column_Zxx_0[:,1]))
    
    print("Shape of first row: ", first_column_Zxx_0.shape)
    print("First row: ", first_column_Zxx_0[0,:])
    # output: (129,2)

    reg_var = 1e-7
    gmm = GaussianMixture(n_components=4, random_state=0, reg_covar=reg_var)
    gmm.fit(first_column_Zxx_0)


    print("Mean shape:", gmm.means_.shape)
    print("Covariance shape:", gmm.covariances_.shape)
    # Create scatter plot for original data
    scatter = go.Scatter(x=first_column_Zxx_0[:, 1], y=first_column_Zxx_0[:, 0], mode='markers', marker=dict(color='blue', symbol='x'), name='Data')
    # Create traces for each GMM component
    gmm_traces = []
    for mean, covar in zip(gmm.means_, gmm.covariances_):
        # Plotting the means as points
        gmm_traces.append(go.Scatter(x=[mean[1]], y=[mean[0]], mode='markers', marker=dict(color='red', size=10), name='GMM Mean'))
        # Plotting Ellipses
        ellipses = plot_covariance_ellipse(mean, covar)
        gmm_traces.append(ellipses)
        
       

    layout = go.Layout(title="GMM on Frequency-Amplitude Data", xaxis_title="Frequency Bin", yaxis_title="Amplitude", showlegend=True)
    fig = go.Figure(data=[scatter] + gmm_traces, layout=layout)
    fig.show()


def exp_bmm(Zxx_list):
    first_column_Zxx_0 = Zxx_list[0][:,0]
    len_column = len(first_column_Zxx_0)
    first_column_Zxx_0 = first_column_Zxx_0.reshape(-1,1)

    # Add column of 1,2,3,...,len_column
    first_column_Zxx_0 = np.hstack((first_column_Zxx_0, np.arange(1,len_column+1).reshape(-1,1)))
    # Scale the first column
    first_column_Zxx_0[:,0] = (first_column_Zxx_0[:,0] - np.min(first_column_Zxx_0[:,0])) / (np.max(first_column_Zxx_0[:,0]) - np.min(first_column_Zxx_0[:,0]))

    # Scale the second column
    first_column_Zxx_0[:,1] = (first_column_Zxx_0[:,1] - np.min(first_column_Zxx_0[:,1])) / (np.max(first_column_Zxx_0[:,1]) - np.min(first_column_Zxx_0[:,1]))

    # Using Bayesian Gaussian Mixture Model
    bgmm = BayesianGaussianMixture(n_components=3, random_state=0)
    bgmm.fit(first_column_Zxx_0)

    scatter = go.Scatter(x=first_column_Zxx_0[:, 1], y=first_column_Zxx_0[:, 0], mode='markers', marker=dict(color='blue', symbol='x'), name='Data')
    gmm_traces = [scatter]

    # Only plot ellipses for components with significant weight
    for mean, covar, weight in zip(bgmm.means_, bgmm.covariances_, bgmm.weights_):
        if weight > 0.01:  # Assuming a component is significant if its weight is above 1%
            # Plotting the means as points
            gmm_traces.append(go.Scatter(x=[mean[1]], y=[mean[0]], mode='markers', marker=dict(color='red', size=10), name='BGMM Mean'))
            # Plotting covariance ellipses
            ellipse = plot_covariance_ellipse(mean, covar)
            gmm_traces.append(ellipse)

    layout = go.Layout(title="BGMM on Frequency-Amplitude Data", xaxis_title="Frequency Bin", yaxis_title="Amplitude", showlegend=True)
    fig = go.Figure(data=gmm_traces, layout=layout)
    fig.show()






if __name__ == "__main__":
    vehicle = Vehicle('mustang0528_rs3')
    data_points = data_processing.get_file_paths('mustang0528_rs3')
    vehicle.add_data_files(data_points)
    vehicle.build_record_file_data(processing_duration=0.1,duration_in_file=2, debug_info= False)

    # print("Num entries: ", len(vehicle))
    # processing_fns = [("downsample_torch_audio", helpers.downsample_torch_audio, {'orig_freq': 8000, 'target_freq': 1000,'vehicle_obj':vehicle })]
    processing_fns = [("downsample_torch_audio", helpers.downsample_torch_audio, {'orig_freq': 8000, 'target_freq': 1000,'vehicle_obj':vehicle }), ("a_law_quantize", helpers.a_law_quantize, {'bitwidth': 8})]
    #
    vehicle.apply_processing_to_data(processing_fns)

    # print(f"Vehicle record file data shape: {vehicle.record_file_data[0]['audio'].shape
    # breakpoint()
    print(f"updated sampling rate audio: {vehicle.original_sampling_audio_rate}")
    vehicle.print_state_processing()

    print(f"Vehicle audio data shape: {vehicle.record_file_data[0]['audio'].shape}")
    #
    f, t, Zxx = vehicle.calculate_stft_single_point(0, nperseg =100, noverlap=0)
    print("f shape: ", f.shape)
    print("t shape: ", t.shape)
    print("Zxx shape: ", Zxx.shape)
    # Zxx_list = get_all_Zxx_of_vehicle(vehicle, nperseg =100, noverlap=0)
    # #
    # print("Shape of Zxx_list: ", Zxx_list.shape)
    # print(f"Shape of Zxx_list[0]: {Zxx_list[0].shape}")
    # exp_1(Zxx_list)
    # exp_bmm(Zxx_list)

    # gmm = get_gmm_of_vehicle(Zxx_list[0], 4)
    #
    # # print(f"Means: {gmm.means_}")
    # print(f"Means shape: {gmm.means_.shape}")
    # print(f"Spherical covariance shape: {gmm.covariances_.shape}")

    # print("Shape: ", Zxx.shape)
