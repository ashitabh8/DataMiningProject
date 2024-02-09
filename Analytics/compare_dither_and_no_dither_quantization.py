from Utils import data_processing, Vehicle, helpers
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
import numpy as np
from scipy.signal import spectrogram

"""
Compare the effect of dithering and no dithering on quantization
"""


def dithered_quantization_test(audio_signal, noise_level=1e-3):
    dithered_audio = helpers.dither_audio(audio_signal,noise_level)
    quantized_audio = helpers.a_law_quantize(dithered_audio, 8)
    return quantized_audio

def downsample_dither_quantized(audio_signal, noise_level=1e-3):
    downsampled_audio = helpers.downsample_torch_audio(audio_signal, 8000, 1000)
    dithered_audio = helpers.dither_audio(downsampled_audio,noise_level)
    quantized_audio = helpers.a_law_quantize(dithered_audio, 8)
    return quantized_audio

def downsample_no_dither_quantized(audio_signal):
    downsampled_audio = helpers.downsample_torch_audio(audio_signal, 8000, 1000)
    quantized_audio = helpers.a_law_quantize(downsampled_audio, 8)
    return quantized_audio


def plot_spectrograms(*audio_signals, titles, fs=[]):
    """
    Calculate and plot spectrograms for given audio signals.

    Parameters:
    *audio_signals: Variable length audio signal list.
    titles (list): Titles for each subplot.
    fs (int): Sampling frequency.
    """
    assert len(audio_signals) == len(titles), "Number of audio signals and titles must be equal."
    assert len(audio_signals) == len(fs), "Number of audio signals and sampling frequencies must be equal."
    num_signals = len(audio_signals)
    
    # Create subplots
    fig = make_subplots(rows=num_signals, cols=1, subplot_titles=titles)

    for i, audio in enumerate(audio_signals, start=1):
        # Calculate spectrogram
        frequencies, times, Sxx = spectrogram(audio, fs=fs[i-1])

        # Plot spectrogram
        fig.add_trace(
            go.Heatmap(
                z=10 * np.log10(Sxx),
                x=times,
                y=frequencies,
                coloraxis="coloraxis",
            ),
            row=i, col=1
        )

    # Update layout
    fig.update_layout(
        height=400 * num_signals, 
        width=800,
        title_text="Spectrograms of Audio Signals",
        coloraxis=dict(colorscale='Viridis'),
        xaxis_title="Time [s]",
        yaxis_title="Frequency [Hz]"
    )

    fig.show()


def main():
    # Step 1 - Get the data
    
    vehicle = Vehicle.Vehicle('tesla_rs3')
    data_points = data_processing.get_file_paths('tesla')
    vehicle.add_data_files(data_points)
    vehicle.build_record_file_data()

    audio_data = vehicle.record_file_data[0]['audio'][0]
    downsampled_audio = helpers.downsample_torch_audio(audio_data, 8000, 1000)

    dithered_and_quantized_audio_1_e_1 = dithered_quantization_test(audio_data,1e-1)
    # dithered_and_quantized_audio_point_25 = dithered_quantization_test(audio_data,0.25)
    no_dither_quantized_audio = helpers.a_law_quantize(audio_data, 8)
    downsampled_dithered_and_quantized_audio_1_e_1 = downsample_dither_quantized(audio_data,1e-1)
    downsampled_no_dither_quantized_audio = downsample_no_dither_quantized(audio_data)


    plot_spectrograms(
        audio_data,
        dithered_and_quantized_audio_1_e_1,
        downsampled_audio,
        downsampled_dithered_and_quantized_audio_1_e_1,
        downsampled_no_dither_quantized_audio,
        # dithered_and_quantized_audio_point_25,
        no_dither_quantized_audio,
        titles=[
            "Original Audio",
            "Dithered and Quantized (Noise Level = 1e-1), Quantization Bitwidth = 8",
            "Downsampled Audio",
            "Downsampled, Dithered and Quantized (Noise Level = 1e-1), Quantization Bitwidth = 8",
            "Downsampled, No Dither Quantized (Quantization Bitwidth = 8)",
            # "Dithered and Quantized (Noise Level = 0.25)",
            "No Dither Quantized (Quantization Bitwidth = 8)"
        ],
        fs = [
            8000,
            8000,
            1000,
            1000,
            1000,
            # 8000,
            8000
        ]
    )


if __name__ == "__main__":
    main()

