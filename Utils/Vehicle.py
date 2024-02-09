# Make a class named vehicle, and add a method named __init__ to it:
import torch
from globals import *
from .helpers import downsample_torch_audio
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
import copy
from scipy import signal
import numpy as np
from scipy import fft
from sklearn.mixture import GaussianMixture 
from multiprocessing import Pool, cpu_count
import multiprocessing
from tqdm import tqdm


def worker_function(args):
    index, num_components, covariance_type, random_state, record_data = args
    f, Zxx = record_data['fft']
    combined_tensor = torch.cat((torch.tensor(f).unsqueeze(0), Zxx.unsqueeze(0)), dim=0)
    gmm = GaussianMixture(n_components=num_components, covariance_type=covariance_type, random_state=random_state)
    gmm.fit(combined_tensor.T.numpy())
    return index, gmm


# def process_batch(pool, batch, num_components, covariance_type, random_state):
#     args_list = [(index, num_components, covariance_type, random_state, data) for index, data in batch]
#     return pool.map(worker_function, args_list)

def process_batch(batch, num_components, covariance_type, random_state, cpu_count):
    with Pool(processes=cpu_count) as pool:
        args_list = [(index, num_components, covariance_type, random_state, data) for index, data in batch]
        results = pool.map(worker_function, args_list)
        pool.close()
        pool.join()
    return results


'''
Manually applying hann for no windowing
'''
def apply_hann_window(signal: np.ndarray):
    N = signal.shape[-1]
    hann_window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N - 1))
    return signal * hann_window

def get_index_from_file_path(file_path):
    base_name = file_path.split('.pt')[0]

    # Split the filename by '_' and extract the last element
    parts = base_name.split('_')
    
    # The index we're interested in is the last numeric part in the filename
    # Loop through the parts in reverse and return the first numeric part found
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    raise ValueError(f"Could not find index in file path {file_path}")

def hz_to_mel(hz):
    """Convert a frequency in Hertz to Mel scale."""
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    """Convert a Mel scale frequency to Hertz."""
    return 700 * (10**(mel / 2595) - 1)

def mel_filterbank(num_filters, fft_bins, sample_rate):
    """Generate a Mel frequency filterbank."""
    max_mel = hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(hz_to_mel(0), max_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((fft_bins + 1) * hz_points / sample_rate).astype(int)
    filters = np.zeros((num_filters, fft_bins))

    for i in range(1, num_filters + 1):
        filters[i - 1, bin_points[i - 1]:bin_points[i]] = (np.arange(bin_points[i - 1], bin_points[i]) - bin_points[i - 1]) / (bin_points[i] - bin_points[i - 1])
        filters[i - 1, bin_points[i]:bin_points[i + 1]] = (bin_points[i + 1] - np.arange(bin_points[i], bin_points[i + 1])) / (bin_points[i + 1] - bin_points[i])

    return filters

'''
TODO: Add this to the class
'''
def compute_mel_spectrogram(audio_signal, sample_rate, n_mels=128, n_fft=2048, hop_length=512):
    """Compute the Mel spectrogram of an audio signal."""
    # Compute the STFT
    frequencies, times, spectrogram = signal.stft(audio_signal, fs=sample_rate, window='hann', nperseg=n_fft, noverlap=n_fft - hop_length)
    
    # Compute Mel filterbank
    mel_filters = mel_filterbank(n_mels, n_fft // 2 + 1, sample_rate)
    
    # Apply the filterbank to the power spectrum (squared magnitude of the STFT)
    mel_power = np.dot(mel_filters, np.abs(spectrogram)**2)
    
    # Convert to log scale
    mel_spectrogram = 10 * np.log10(mel_power + 1e-6)
    
    return mel_spectrogram, frequencies, times


class Vehicle:
    def __init__(self, vehicle_name, duration = 2):
        self.vehicle_name = vehicle_name
        self.data_file_paths = []
        self.file_path_to_index = {}
        self.is_data_in_tensor = False
        self.data_tensor_audio = None
        self.data_tensor_seismic = None
        self.next_index_to_add = 0
        self.original_sampling_audio_rate = 8000

        self.original_sampling_seismic_rate = 100
        self.processing_functions = []
        self.duration = duration


    def add_data_files(self, data_dir_path):
        if isinstance(data_dir_path, list):
            # Add globals.DATA_DIR_PATH as prefix to all the paths
            data_dir_path = [DATA_DIR_PATH + path for path in data_dir_path]
            # print("first part sort key: ", data_dir_path[0].split(selfii.vehicle_name + '_')[-1])
            # print("sort key: ", int(data_dir_path[0].split(self.vehicle_name + '_')[-1].split('.pt')[0]))
            data_dir_path.sort(key=lambda x: get_index_from_file_path(x))
            for path in data_dir_path:
                if path not in self.file_path_to_index:
                    self.data_file_paths.append(path)
                    self.file_path_to_index[path] = self.next_index_to_add
                    self.next_index_to_add += 1
            return 1
        if isinstance(data_dir_path, str):
            if data_dir_path not in self.file_path_to_index:
                self.data_file_paths.append(data_dir_path)
                self.file_path_to_index[data_dir_path] = self.next_index_to_add
                self.next_index_to_add += 1
        raise ValueError("data_dir_path must be a list of strings or a string")
        # self.data_file_paths.sort(key=lambda x: int(x.split(self.vehicle_name + '_')[-1].split('.pt')[0]))
    
    '''
    Return a plotly figure object, with title, x-axis label, y-axis label, and frequency and self.vehicle_name in the caption
    Use self.record_file_data[index]['audio'] to get the audio data
    if freq is less than 8000, then downsample the audio data

    '''
    def get_plot_audio_data(self,pt_num,freq, quantization_bitwidth = None):
        '''
        pt_num : int Point number to plot
        '''
        audio_data = self.record_file_data[pt_num]['audio'][0]
        # print("First 10 audio data: ", audio_data[:5])

        # downsample the audio data if freq is less than 8000
        assert freq <= self.original_sampling_audio_rate, "freq must be less than or equal to 8000"
        if freq < self.original_sampling_audio_rate:
            audio_data = downsample_torch_audio(audio_data, self.original_sampling_audio_rate, freq)
        # Create the plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=audio_data))

        # Setting the title, x-axis and y-axis labels
        fig.update_layout(
            title='Audio Data Visualization',
            xaxis_title='Time',
            yaxis_title='Amplitude',
            # Assuming self.vehicle_name is a property of your class
            annotations=[{
                'text': f"Frequency: {freq}, Vehicle Name: {self.vehicle_name}",
                'showarrow': False,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0
            }]
        )

        # plot(fig, filename='./tmp/temp-plot.html', auto_open=True)

        return fig

    '''
    Calculate and Plot the RMSE between the audio and seismic data, for all the data files in the vehicle object
    '''
    def plot_rmse_energy(self,duration: float = 2, category = 'audio'):
        record_keys = self.get_keys()
        average_audio_energy = {}
        for key in record_keys:
            average_audio_energy[key] = self.average_audio_energy(key, duration_split=duration)

        X_labels = []
        Y_labels = []

        for record, energy_entries in average_audio_energy.items():
            for i in range(len(energy_entries)):
                X_labels.append(f"{record}_split_{i}")
                Y_labels.append(energy_entries[i])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X_labels, y=Y_labels, mode='markers'))


        # Update the layout of the figure
        fig.update_layout(
            title=f"Average Audio Energy (Root Mean Square) per Record Split for {self.vehicle_name}",
            xaxis_title="Split Number",
            yaxis_title="Average Audio Energy",
            template="plotly_dark"  # You can change the template as per your preference
        )
        annotation_text = ''
        annotation_text += f"Durations: {duration}<br>"
        annotation_text += f"Processing Steps: <br>"
        for function_name, function, function_args in self.processing_functions:
            annotation_text += f"{function_name}: {function_args}<br>"

        fig.add_annotation(
            text=annotation_text,
            align='left',
            showarrow=False,
            xref='paper',
            yref='paper',
            x=0.9,
            y=0.95,  # Adjust this value to move the caption up or down
            font=dict(size=18, color="white"),  # Adjust font and color as needed
            bgcolor='rgba(0,0,0,0)'  # Set background color (transparent here)
        )

        fig.show()

    def plot_rmse_energy_both(self, duration: float = 2):
        record_keys = self.get_keys()
        average_audio_energy = {}
        for key in record_keys:
            average_audio_energy[key] = self.average_audio_energy(key, duration_split=duration)

        X_labels = []
        Y_labels = []

        for record, energy_entries in average_audio_energy.items():
            for i in range(len(energy_entries)):
                X_labels.append(f"{record}_split_{i}")
                Y_labels.append(energy_entries[i])

        fig = make_subplots(rows=2, cols=1, subplot_titles=("Audio", "Seismic"))
        fig.add_trace(go.Scatter(x=X_labels, y=Y_labels, mode='markers'), row=1, col=1)

        average_seismic_energy = {}
        for key in record_keys:
            average_seismic_energy[key] = self.average_seismic_energy(key, duration_split=duration)

        X_labels = []
        Y_labels = []

        for record, energy_entries in average_seismic_energy.items():
            for i in range(len(energy_entries)):
                X_labels.append(f"{record}_split_{i}")
                Y_labels.append(energy_entries[i])

        fig.add_trace(go.Scatter(x=X_labels, y=Y_labels, mode='markers'), row=2, col=1)

        # Update the layout of the figure
        fig.update_layout(
            title=f"Average Audio and Siesmic Energy (Root Mean Square) per Record Split for {self.vehicle_name}",
            xaxis_title="Split Number",
            yaxis_title="Average Audio Energy",
            template="plotly_dark"  # You can change the template as per your preference
        )

        fig.show()

    def plot_rmse_energy_both_normalized(self, duration: float = 2, overlay = False):
        record_keys = self.get_keys()
        
        # Collecting audio and seismic energy data
        average_audio_energy = {}
        for key in record_keys:
            average_audio_energy[key] = self.average_audio_energy(key, duration_split=duration)

        audio_X_labels = []
        audio_Y_values = []

        for record, energy_entries in average_audio_energy.items():
            for i, energy in enumerate(energy_entries):
                audio_X_labels.append(f"{record}_split_{i}")
                audio_Y_values.append(energy)

        average_seismic_energy = {}
        for key in record_keys:
            average_seismic_energy[key] = self.average_seismic_energy(key, duration_split=duration)

        seismic_X_labels = []
        seismic_Y_values = []

        for record, energy_entries in average_seismic_energy.items():
            for i, energy in enumerate(energy_entries):
                seismic_X_labels.append(f"{record}_split_{i}")
                seismic_Y_values.append(energy)

        audio_min = min(audio_Y_values)
        audio_max = max(audio_Y_values)
        normalized_audio_Y = [(y - audio_min) / (audio_max - audio_min) for y in audio_Y_values]

        # Normalizing seismic dataset separately
        seismic_min = min(seismic_Y_values)
        seismic_max = max(seismic_Y_values)
        normalized_seismic_Y = [(y - seismic_min) / (seismic_max - seismic_min) for y in seismic_Y_values]


        # Creating a combined plot
        if not overlay:
            fig = make_subplots(rows=2, cols=1, subplot_titles=("Audio", "Seismic"))
            fig.add_trace(go.Scatter(x=audio_X_labels, y=normalized_audio_Y, mode='markers', name='Normalized Audio'), row=1, col=1)
            fig.add_trace(go.Scatter(x=seismic_X_labels, y=normalized_seismic_Y, mode='markers', name='Normalized Seismic'), row=2, col=1)

            # Update the layout of the figure
            fig.update_layout(
                title=f"Normalized Average Audio and Seismic Energy per Record Split for {self.vehicle_name}",
                xaxis_title="Record Split",
                yaxis_title="Normalized Energy",
                template="plotly_dark"  # You can change the template as per your preference
            )

            fig.show()
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=audio_X_labels, y=normalized_audio_Y, mode='markers', name='Normalized Audio'))
            fig.add_trace(go.Scatter(x=seismic_X_labels, y=normalized_seismic_Y, mode='markers', name='Normalized Seismic'))

            # Update the layout of the figure
            fig.update_layout(
                title=f"Normalized Average Audio and Seismic Energy per Record Split for {self.vehicle_name}",
                xaxis_title="Record Split",
                yaxis_title="Normalized Energy",
                template="plotly_dark"  # You can change the template as per your preference
            )

            fig.show()

    '''
    [FINISH THIS]
    Generalised PLotting Manager for this class
    type: str (default = 'columnwise') Type of plot to make, can make them 'overlay'
    processing_functions: list of lists of tuples (default = []) List of processing functions to apply to the data
    '''
    def make_plot(self, processing_functions, titles = None, type = 'columnwise'):
        assert type in ['columnwise', 'overlay'], "type must be either 'columnwise' or 'overlay'"

        # fig = go.Figure()
        if type == 'columnwise':
            num_of_plots = len(processing_functions)
            fig = make_subplots(rows = num_of_plots, cols = 1, subplot_titles = titles)
            
            
        pass

    
    def check_if_pt_num_exists_in_data_file_paths(self, pt_num):
        pass

    def get_1d_data(self, dataset):
        return dataset.reshape(dataset.shape[0], -1)

    def get_num_columns(self):
        sample = torch.load(self.data_file_paths[0])
        audio_data = sample["data"]["shake"]["audio"]
        seismic_data = sample["data"]["shake"]["seismic"]
        audio_data = self.get_1d_data(audio_data)
        seismic_data = self.get_1d_data(seismic_data)
        audio_data_shape = audio_data.shape
        seismic_data_shape = seismic_data.shape
        return (audio_data_shape[1], seismic_data_shape[1])


    def calculate_stft_single_point(self, pt_num, category = 'audio', window = 'hann', nperseg:int = 256, noverlap:int = 0):
        assert category in ['audio', 'seismic'], "category must be either 'audio' or 'seismic'"
        data: np.ndarray = np.array(0)
        freq:float = 0.0
        if category == 'audio':
            data = self.record_file_data[pt_num]['audio'][0]
            freq = self.original_sampling_audio_rate
            # print(f"Using freq: {freq}")
        elif category == 'seismic':
            data = self.record_file_data[pt_num]['seismic'][0]
            freq = self.original_sampling_seismic_rate
        # breakpoint()
        # print(f"Shape: self.record_file_data = {self.record_file_data[0]['audio'].shape}")
        # print(f"(Shape: data var = {data.shape}")
        if data.shape[-1] == nperseg and window == 'hann':
            data = apply_hann_window(data)
            # data = data.numpy()
            # breakpoint()
            # print(f"Shape of data after manual hann {data.shape}")
            # print(f"Freq: {freq}")
            f,t,Zxx = signal.stft(data, fs=freq, window='boxcar', nperseg=nperseg, noverlap=noverlap)
            return f,t,Zxx
        else:
            f, t, Zxx = signal.stft(data, fs=freq, window=window, nperseg=nperseg, noverlap=noverlap)
            return f, t, Zxx

    def calculate_fft_single_point(self, pt_num, category = 'audio', absolute = True):
        assert category in ['audio', 'seismic'], "category must be either 'audio' or 'seismic'"
        data: np.ndarray = np.array([])
        freq: float = 0.0

        # Select the data and frequency based on the category
        if category == 'audio':
            data = self.record_file_data[pt_num]['audio'][0]
            freq = self.original_sampling_audio_rate
        elif category == 'seismic':
            data = self.record_file_data[pt_num]['seismic'][0]
            freq = self.original_sampling_seismic_rate

        # Perform FFT on the data
        Zxx = torch.fft.fft(data)
        if absolute:
            Zxx = torch.abs(Zxx)

        # Generate the frequency bins based on the sampling rate and data length
        # This is important to interpret the FFT results correctly
        N = len(data)  # Number of data points
        # Frequency bins (in Hertz), only the positive part up to Nyquist frequency
        f = np.linspace(0, freq / 2, N // 2, endpoint=False)

        # Since FFT output is symmetrical, only take the first half of it for positive frequencies
        Zxx_half = Zxx[:N // 2]

        return f, Zxx_half

    def add_fft_values(self, category = 'audio'):
        num_samples = len(self.record_file_data)
        for i in range(num_samples):
            f, Zxx = self.calculate_fft_single_point(i, category)
            self.record_file_data[i]['fft'] = (f, Zxx)

    # def add_gmms_for_fft(self, num_components=3, covariance_type='full', random_state=0, only_non_silence=True, category='audio', cpu_count=4):
    #     batch_size = 10
    #     if only_non_silence:
    #         pts_to_calculate = self.get_all_non_silence_pts()
    #     else:
    #         pts_to_calculate = list(range(len(self.record_file_data)))
    #     
    #     # If cpu_count is not specified, use the maximum number of CPUs available
    #     if cpu_count is None:
    #         cpu_count = multiprocessing.cpu_count()
    #     
    #     batches = [pts_to_calculate[i:i + batch_size] for i in range(0, len(pts_to_calculate), batch_size)]
    #     
    #     batch_num = 0
    #     for batch in batches:
    #         print(f"Processing batch {batch_num + 1} of {len(batches)}")
    #         # Prepare batch data
    #         batch_data = [(index, self.record_file_data[index]) for index in batch]
    #         results = process_batch(batch_data, num_components, covariance_type, random_state, cpu_count)
    #         
    #         # Store the results back in the record_file_data
    #         for index, gmm in results:
    #             self.record_file_data[index]['gmm'] = gmm
    #         batch_num += 1


    # def add_gmms_for_fft(self, num_components=3, covariance_type='full', random_state=0, only_non_silence=True, category='audio', cpu_count=8):
    #     if only_non_silence:
    #         pts_to_calculate = self.get_all_non_silence_pts()
    #     else:
    #         pts_to_calculate = list(range(len(self.record_file_data)))
    #     
    #     # Prepare the arguments for each worker
    #     args_list = [(index, num_components, covariance_type, random_state, self.record_file_data[index]) for index in pts_to_calculate]
    #     
    #     # If cpu_count is not specified, use the maximum number of CPUs available
    #     if cpu_count is None:
    #         cpu_count = multiprocessing.cpu_count()
    #     
    #     # Create a Pool of workers and map the worker function over the points
    #     with Pool(processes=cpu_count) as pool:
    #         results = pool.map(worker_function, args_list)
    #
    #     # Store the results back in the record_file_data
    #     for index, gmm in results:
    #         self.record_file_data[index]['gmm'] = gmm

    def add_gmms_for_fft(self, num_components = 3, 
                         covariance_type = 'full',random_state = 0,
                         only_non_silence= True, category = 'audio'):
        pts_to_calculate = []
        if only_non_silence:
            pts_to_calculate = self.get_all_non_silence_pts()
        else:
            pts_to_calculate = list(range(len(self.record_file_data)))

        for index in tqdm(pts_to_calculate, desc="Calculating GMMs for FFT"):
            f, Zxx = self.record_file_data[index]['fft']
            combined_tensor = torch.cat((torch.tensor(f).unsqueeze(0), Zxx.unsqueeze(0)), dim=0)
            # breakpoint()
            self.record_file_data[index]['gmm'] = GaussianMixture(n_components = num_components,
                                                                  covariance_type =covariance_type, 
                                                                  random_state = random_state)
            self.record_file_data[index]['gmm'].fit(combined_tensor.T)

    def get_all_non_silence_pts(self):
        assert 'silence' in self.record_file_data[0], "silence key does not exist in record_file_data"

        non_silence_pts = []
        for i in range(len(self.record_file_data)):
            if not self.record_file_data[i]['silence']:
                non_silence_pts.append(i)
        return non_silence_pts

    def show_energy_stats(self, category = 'audio'):
        if category == 'audio':
            num_records = len(self.record_file_data)
            print(f"Vehicle name: {self.vehicle_name}")
            print(f"Number of records: {num_records}")
            print(f"Duration: {self.duration}")
            print(f"Frequency: {self.original_sampling_audio_rate}")
            all_rmse_values = []
            for i in range(num_records):
                all_rmse_values.append(self.average_audio_energy(i))

            all_rmse_values = torch.tensor(all_rmse_values)
            print(f"Min RMSE: {torch.min(all_rmse_values)}")
            print(f"Max RMSE: {torch.max(all_rmse_values)}")
            print(f"Mean RMSE: {torch.mean(all_rmse_values)}")
            print(f"Median RMSE: {torch.median(all_rmse_values)}")
            print(f"First Quartile RMSE: {torch.quantile(all_rmse_values, 0.25)}")
            print(f"Third Quartile RMSE: {torch.quantile(all_rmse_values, 0.75)}")


    # def build_plot(self, pt_num = 0, file_name = None):
    #     '''
    #     pt_num : int (default = 0) Point number to plot
    #     file_name : str (default = None) File name to save the plot
    #     '''
    #     assert check_if_pt_num_exists_in_data_file_paths(pt_num), "pt_num does not exist in data_file_paths"
    #     
    #     pass
    def average_audio_energy(self, pt_num):
        audio_data = self.record_file_data[pt_num]['audio'][0]
        return torch.sqrt(torch.mean(audio_data ** 2))

    def average_audio_energy_old(self, pt_num, duration_split: float= 2):
        audio_data = self.record_file_data[pt_num]['audio'][0]
        num_splits = int(self.duration / duration_split)

        # Split the audio data into num_splits
        split_length = len(audio_data) // num_splits

        # Split the audio data into num_splits
        audio_data_splits = tuple(audio_data[i * split_length:(i + 1) * split_length] for i in range(num_splits))

        # Calculate the average energy for each split
        average_energy = []
        for split in audio_data_splits:
            average_energy.append(torch.sqrt(torch.mean(split ** 2)))
        return tuple(average_energy)
    
    def average_seismic_energy(self, pt_num, duration_split: float= 2):
        seismic_data = self.record_file_data[pt_num]['seismic'][0]
        num_splits = int(self.duration / duration_split)

        # Split the audio data into num_splits
        split_length = len(seismic_data) // num_splits

        # Split the audio data into num_splits
        seismic_data_splits = tuple(seismic_data[i * split_length:(i + 1) * split_length] for i in range(num_splits))

        # Calculate the average energy for each split
        average_energy = []
        for split in seismic_data_splits:
            average_energy.append(torch.sqrt(torch.mean(split ** 2)))
        return tuple(average_energy)


    # def build_record_file_data(self):
    #     self.data_file_paths.sort(key=lambda x: get_index_from_file_path(x))
    #     # Make each data file path a key, and the corresponding audio and seismic data a value
    #     self.record_file_data = {}
    #
    #     for file_path in self.data_file_paths:
    #         sample = torch.load(file_path)
    #         audio_data = sample["data"]["shake"]["audio"]
    #         seismic_data = sample["data"]["shake"]["seismic"]
    #
    #         # Optionally reshape the data
    #         audio_data = self.get_1d_data(audio_data)
    #         seismic_data = self.get_1d_data(seismic_data)
    #
    #         # Store the data in the dictionary with the file path as the key
    #         self.record_file_data[self.file_path_to_index[file_path]] = {'audio': audio_data, 'seismic': seismic_data}
    #         self.record_file_data[self.file_path_to_index[file_path]]['file_path'] = file_path

    def build_record_file_data(self,processing_duration, duration_in_file: float = 2, debug_info = False):
        self.record_file_data = {}
        self.data_file_paths.sort(key=lambda x: get_index_from_file_path(x))

        assert self.duration % duration_in_file == 0, "duration must be divisible by duration_in_file"
        num_slits_per_file = int(duration_in_file / processing_duration)

        num_entries_total = len(self.data_file_paths) * num_slits_per_file

        if debug_info:
            print(f"Duration in file: {duration_in_file}")
            print(f"Processing duration: {processing_duration}")
            print(f"Number of splits per file: {num_slits_per_file}")
            print(f"Number of entries total: {num_entries_total}")

        # iterate with index over the data_file_paths
        for i, file_path in enumerate(self.data_file_paths):
            sample = torch.load(file_path)
            audio_data = sample["data"]["shake"]["audio"]
            # breakpoint()
            seismic_data = sample["data"]["shake"]["seismic"]

            # Optionally reshape the data
            audio_data = self.get_1d_data(audio_data)
            seismic_data = self.get_1d_data(seismic_data)

            # Split the audio and seismic data into num_splits
            split_length_audio = audio_data.shape[1] // num_slits_per_file
            split_lenght_seismic = seismic_data.shape[1] // num_slits_per_file
            audio_data_splits = tuple(audio_data[0][i * split_length_audio:(i + 1) * split_length_audio] for i in range(num_slits_per_file))
            # breakpoint()

            seismic_data_splits = tuple(seismic_data[0][i * split_lenght_seismic:(i + 1) * split_lenght_seismic] for i in range(num_slits_per_file))

            # Store the data in the dictionary with the file path as the key
            for j in range(num_slits_per_file):
                self.record_file_data[i * num_slits_per_file + j] = {'audio': audio_data_splits[j], 'seismic': seismic_data_splits[j]}
                # breakpoint()
                self.record_file_data[i * num_slits_per_file + j]['file_path'] = file_path

        self.duration = processing_duration
        assert len(self.record_file_data) == num_entries_total, "record_file_data does not have the correct number of entries"

    '''
    Adds a label 'silence' to the self.record_file_data dictionary
    '''
    def add_label_silence(self, rmse_threshold = 300):
        num_samples = len(self.record_file_data)
        if self.record_file_data == {}:
            raise ValueError("record_file_data is empty. Please call build_record_file_data() first.")

        if 'silence' in self.record_file_data[0]:
            print("silence key already exists in record_file_data. Skipping...")
            return

        if 'rmse' not in self.record_file_data[0]:
            self.add_rmse_values()

        total_silence = 0
        for i in range(num_samples):
            if self.record_file_data[i]['rmse'] < rmse_threshold:
                total_silence += 1
                self.record_file_data[i]['silence'] = True
            else:
                self.record_file_data[i]['silence'] = False
        print(f"Total silence: {total_silence}")

    def add_rmse_values(self):
        num_samples = len(self.record_file_data)
        for i in range(num_samples):
            self.record_file_data[i]['rmse'] = self.average_audio_energy(i)

    
    def add_stft_values(self, nperseg:int = 256, noverlap:int = 0):
        num_samples = len(self.record_file_data)
        for i in range(num_samples):
            f, t, Zxx = self.calculate_stft_single_point(i, nperseg=nperseg, noverlap=noverlap)
            self.record_file_data[i]['stft'] = (f,t,Zxx)

    def get_all_stft_values(self):
        num_samples = len(self.record_file_data)
        all_stft_values = []
        for i in range(num_samples):
            all_stft_values.append(self.record_file_data[i]['stft'])
        return all_stft_values


    def add_silence_tags(self, threshold=75):
        """
        Adds a 'silence' tag to records in `record_file_data` based on their audio energy levels.

        This method iterates over all records in `record_file_data`, a list of dictionaries where each
        dictionary represents a single audio record. It calculates the average audio energy for each record
        using the `average_audio_energy` method. If the average energy is below a specified threshold,
        the method adds a key-value pair `'silence': True` to the corresponding dictionary to tag it as silence.

        Parameters:
        - threshold (int, optional): The energy level below which a record is considered silent.
                                      Defaults to 75.

        Modifies:
        - Each dictionary in `record_file_data` may have a new key `'silence'` with the value `True`
          if its average audio energy is below the threshold.

        Returns:
        - None. The function modifies `record_file_data` in place.

        Example:
        - Assuming `record_file_data` is a list of dictionaries, each representing an audio record,
          calling `add_silence_tags()` will add a `'silence': True` key-value pair to each record
          whose average audio energy is below the threshold.
        """
        num_records = len(self.record_file_data)
        for i in range(num_records):
            current_energy = self.average_audio_energy(i)
            if current_energy < threshold:
                self.record_file_data[i]['silence'] = True

    '''
    Apply a list of tuples of processing functions to the data
    ("function name", function, function_args)

    Example:
    [("downsample", downsample_torch_audio, {"original_freq": 8000, "new_freq": 1000}), ("quantize", quantize, {"bitwidth": 8})]
    '''
    def apply_processing_to_data(self, processing_functions: list, data_category = 'audio', inplace = True, debug_info = False):
        if inplace:
            for function_name, function, function_args in processing_functions:
                # breakpoint()
                for pt_num in self.record_file_data.keys():
                    if function_args is None:
                        # breakpoint()
                        self.record_file_data[pt_num][data_category] = function(self.record_file_data[pt_num][data_category])
                    else:
                        if debug_info:
                            print(f"Shape of data before processing: {self.record_file_data[pt_num][data_category].shape}")
                        current_output = function(self.record_file_data[pt_num][data_category], **function_args)
                        self.record_file_data[pt_num][data_category] = current_output
                self.processing_functions.append((function_name, function, function_args))
        else:
            new_record_file_data = copy.deepcopy(self.record_file_data)
            for function_name, function, function_args in processing_functions:
                for pt_num in new_record_file_data.keys():
                    if function_args is None:
                        new_record_file_data[pt_num][data_category] = function(new_record_file_data[pt_num][data_category][0])
                    else:
                        current_output = function(new_record_file_data[pt_num][data_category][0], **function_args)
                        current_output = current_output.unsqueeze(0)
                        new_record_file_data[pt_num][data_category] = current_output
            return new_record_file_data



    def print_state_processing(self):
        print("Vehicle Name: ", self.vehicle_name)
        for processing_steps in self.processing_functions:
            print("\tFunction Name: ", processing_steps[0])
            print("\tFunction Args: ", processing_steps[2])



        

    '''
    Returns a tuple view of the audio data.
    '''
    def get_audio_data(self,pt_num,duration_split = 2):
        assert self.duration >= duration_split, "duration_split must be less than or equal to self.duration"
        assert self.duration % duration_split == 0, "duration must be divisible by duration_split"

        # Get the audio data
        audio_data = self.record_file_data[pt_num]['audio'][0]
        num_splits = int(self.duration / duration_split)

        # Split the audio data into num_splits
        split_length = len(audio_data) // num_splits

        # Split the audio data into num_splits
        audio_data_splits = tuple(audio_data[i * split_length:(i + 1) * split_length] for i in range(num_splits))


        return audio_data_splits

    '''
    return length of number of data files
    '''
    def __len__(self):
        return len(self.record_file_data)

    '''
    return all keys in the record_file_data
    '''
    def get_keys(self):
        return self.record_file_data.keys()

    def split_to_custom_duration(self, duration, audio_type = 'audio'):
        pass

    def print_state(self):
        assert self.data_file_paths != [], "No data files have been added to the vehicle"
        print("Vehicle Name: ", self.vehicle_name)
        print("Record Entries...")
        all_keys = list(self.record_file_data[0].keys())
        print("Keys: ", all_keys)


    def build_data_tensor(self):
        # sort the data_file_paths using self.vehicle_name as key
        self.data_file_paths.sort(key=lambda x: int(x.split(self.vehicle_name + '_')[-1].split('.pt')[0]))


        num_files = len(self.data_file_paths)
        # build a tensor of shape (num_files, len_each_sample)
        num_columns_audio, num_columns_seismic = self.get_num_columns()

        self.data_tensor_audio = torch.zeros((num_files, num_columns_audio))
        self.data_tensor_seismic = torch.zeros((num_files, num_columns_seismic))

        for i, file_path in enumerate(self.data_file_paths):
            sample = torch.load(file_path)
            audio_data = sample["data"]["shake"]["audio"]
            seismic_data = sample["data"]["shake"]["seismic"]
            audio_data = self.get_1d_data(audio_data)
            seismic_data = self.get_1d_data(seismic_data)
            self.data_tensor_audio[i] = audio_data
            self.data_tensor_seismic[i] = seismic_data
        return self.data_tensor_audio, self.data_tensor_seismic


    def return_noise_eliminated_data(self):
        pass


    def processing_pipeline(self, *processing_functions):
        assert self.record_file_data != {}, "record_file_data is empty. Please call build_record_file_data() first."
        pass





    



