import torch
import numpy as np
from scipy.signal import resample,stft
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots

# NOTE: FIRST ARG should be audio tensor for preprocessing functions

def downsample_torch_audio(audio_tensor,orig_freq, target_freq, vehicle_obj = None):
    """
    Downsamples a 1D torch tensor from 8000 Hz to a target frequency.

    Parameters:
    audio_tensor (torch.Tensor): The input audio tensor sampled at orig_freq.
    target_freq (int): The target sampling frequency.
    vehicle_obj to update the internal state of the object to reflect new frequency

    Returns:
    torch.Tensor: The downsampled audio tensor.
    """
    original_freq = orig_freq  # Original frequency
    num_samples = int(len(audio_tensor) * (target_freq / original_freq))

    # Convert to NumPy, downsample, then convert back to torch tensor
    audio_np = audio_tensor.numpy()
    downsampled_audio_np = resample(audio_np, num_samples)
    downsampled_audio_tensor = torch.from_numpy(downsampled_audio_np)
    if vehicle_obj is not None:
        vehicle_obj.original_sampling_audio_rate = target_freq
        # print(f"vehicle_obj is not None, updated original_sampling_rate: {vehicle_obj.original_sampling_audio_rate} ")
    downsampled_audio_tensor = downsampled_audio_tensor.unsqueeze(0)
    return downsampled_audio_tensor

def normalize_tensor(tensor):
    min = tensor.min()
    max = tensor.max()

    return (tensor - min) / (max - min)

def log_scaling(tensor):
    return torch.log1p(tensor)

def dither_audio(audio_tensor, noise_level=1e-1):
    """
    Applies dithering to an audio signal represented as a 1D torch tensor.
    Dithering is a technique used to improve the quality of audio signals in
    digital systems. It works by adding a small amount of noise to the signal
    before quantization.

    Parameters:
    audio_tensor (torch.Tensor): The input audio tensor.
    noise_level (float): The amplitude of the noise to be added.

    Returns:
    torch.Tensor: The dithered audio tensor.
    """
    # Generate random noise

    # Find the peak level of the audio signal to scale the noise
    peak_level = audio_tensor.abs().max()
    noise_level = peak_level * noise_level

    # Generate random noise
    noise = torch.rand_like(audio_tensor) * noise_level

    # Subtract 0.5 to center the noise around zero
    noise = noise - 0.5 * noise_level

    # Add noise to the audio
    dithered_audio = audio_tensor + noise

    return dithered_audio

def a_law_quantize(audio_tensor, bitwidth=8, A=87.6):
    """
    Apply A-law quantization to an audio signal represented as a PyTorch tensor.

    Parameters:
    audio_tensor (torch.Tensor): The input audio tensor.
    bitwidth (int): The bitwidth for quantization.
    A (float): The A-law compression parameter.

    Returns:
    torch.Tensor: The A-law quantized audio tensor.
    """
    # Convert A to a tensor
    A_tensor = torch.tensor(A)

    # Ensure the audio tensor is in the range [-1, 1]
    max_val = torch.max(torch.abs(audio_tensor))
    if max_val > 1:
        audio_tensor = audio_tensor / max_val

    # Apply the A-law transformation element-wise
    sign = torch.sign(audio_tensor)
    abs_x = torch.abs(audio_tensor)
    mask = abs_x < 1 / A_tensor
    compressed_audio = torch.where(mask, 
                                   sign * (A_tensor * abs_x) / (1 + torch.log(A_tensor)), 
                                   sign * (1 + torch.log(A_tensor * abs_x)) / (1 + torch.log(A_tensor)))

    # Quantize the signal based on the specified bitwidth
    num_levels = 2 ** bitwidth
    quantized_tensor = torch.round((compressed_audio + 1) * (num_levels / 2 - 1))
    quantized_tensor = quantized_tensor - 2 ** (bitwidth - 1)
    # breakpoint()

    return quantized_tensor


def calculate_stft(signal, fs=1.0, window='hann', nperseg=256, noverlap=None):
    """
    Calculate the Short Time Fourier Transform of a signal.

    Parameters:
    signal (array_like): Input signal.
    fs (float, optional): Sampling frequency of the signal. Default is 1.0.
    window (str or tuple or array_like, optional): Desired window to use. Default is 'hann'.
    nperseg (int, optional): Length of each segment. Default is 256.
    noverlap (int, optional): Number of points to overlap between segments. If None, noverlap = nperseg // 2. Default is None.

    Returns:
    f (ndarray): Array of sample frequencies.
    t (ndarray): Array of segment times.
    Zxx (ndarray): STFT of signal.
    """

    f, t, Zxx = signal.stft(signal, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    return f, t, Zxx





    return audio_tensor

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

def plot_gmm(gmm_means, gmm_covariances, data_points):
    # print("Mean shape:", gmm.means_.shape)
    # print("Covariance shape:", gmm.covariances_.shape)
    # Create scatter plot for original data
    scatter = go.Scatter(x=data_points[:, 1], y=data_points[:, 0], mode='markers', marker=dict(color='blue', symbol='x'), name='Data')
    # Create traces for each GMM component
    gmm_traces = []
    for mean, covar in zip(gmm_means,gmm_covariances):
        # Plotting the means as points
        gmm_traces.append(go.Scatter(x=[mean[1]], y=[mean[0]], mode='markers', marker=dict(color='red', size=10), name='GMM Mean'))
        # Plotting Ellipses
        ellipses = plot_covariance_ellipse(mean, covar)
        gmm_traces.append(ellipses)
        
    layout = go.Layout(title="GMM on Frequency-Amplitude Data", xaxis_title="Frequency Bin", yaxis_title="Amplitude", showlegend=True)
    fig = go.Figure(data=[scatter] + gmm_traces, layout=layout)
    fig.show()

def get_one_hot_encoding(names):
    name_to_index = {name: index for index, name in enumerate(names)}

    # Initialize an empty list to hold the one-hot encoded vectors
    one_hot_encoded = {}
    # Iterate over each vehicle name in the list
    for name in names:
        # Create a vector of zeros with the same length as the number of unique vehicle names
        encoding = [0] * len(names)
        # Set the position corresponding to the current vehicle name to 1
        encoding[name_to_index[name]] = 1
        # Append the one-hot encoded vector to the list
        one_hot_encoded[name] = encoding
    return one_hot_encoded

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

processing_function_mapping = {'downsample_torch_audio':downsample_torch_audio,
                                 'dither_audio':dither_audio,
                                 'a_law_quantize':a_law_quantize}
