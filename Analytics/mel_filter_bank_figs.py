import numpy as np
import plotly.graph_objects as go
import plotly.offline

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

# Parameters for the Mel filterbank
num_filters = 26  # Number of Mel filters
fft_bins = 512  # Number of FFT bins
sample_rate = 22050  # Sample rate of the audio signal

# Generate the Mel filterbank
mel_filters = mel_filterbank(num_filters, fft_bins, sample_rate)

# Plot the Mel filterbank using Plotly
fig = go.Figure()

for n in range(mel_filters.shape[0]):
    fig.add_trace(go.Scatter(y=mel_filters[n], mode='lines', name=f'Filter {n+1}'))

fig.update_layout(
    title='Mel Filter Banks',
    xaxis_title='Frequency (Bin)',
    yaxis_title='Amplitude',
    template='plotly_white'
)

# Plot offline in a browser
plotly.offline.plot(fig, filename='mel_filter_banks.html')

