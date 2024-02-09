# import Utils.Vehicle as Vehicle 
from Utils import Vehicle, data_processing, helpers
# from Utils.data_processing import get_file_paths
# from globals import DATA_DIR_PATH
import plotly.graph_objs as go
from plotly.offline import plot

'''
    Test: Vehicle Class
    Description: Testing data imports and data concatenation into a single dataframe
'''
def test_vehicle_class():
    vehicle = Vehicle.Vehicle('tesla_rs3')
    data_points = data_processing.get_file_paths('tesla')
    vehicle.add_data_files(data_points)
    data_tensor_audio, data_tensor_seismic = vehicle.build_data_tensor()
    print("Shape Tensor Audio: ", data_tensor_audio.shape)
    print("Shape Tensor Seismic: ", data_tensor_seismic.shape)

    print("First 10 data files: ", vehicle.data_file_paths[:10])



'''
    Test: Vehicle Class Data Record Function
    Description: Testing record function
'''
def test_vehicle_class_record_function():
    vehicle = Vehicle.Vehicle('tesla_rs3')
    data_points = data_processing.get_file_paths('tesla')
    vehicle.add_data_files(data_points)
    vehicle.build_record_file_data()
    print("First 10 data files: ", vehicle.data_file_paths[:10])
    print("0 audio data: ", vehicle.record_file_data[0]['audio'].shape)
    print("0 seismic data: ", vehicle.record_file_data[0]['seismic'].shape)
    print("0 file path: ", vehicle.record_file_data[0]['file_path'])
    print("1 audio data: ", vehicle.record_file_data[1]['audio'].shape)
    print("1 seismic data: ", vehicle.record_file_data[1]['seismic'].shape)
    print("1 file path: ", vehicle.record_file_data[1]['file_path'])
    print("2 audio data: ", vehicle.record_file_data[2]['audio'].shape)
    print("2 seismic data: ", vehicle.record_file_data[2]['seismic'].shape)
    print("2 file path: ", vehicle.record_file_data[2]['file_path'])
    vehicle.print_state()

'''
Make plots for a record
'''
def make_plot_for_record():
    vehicle = Vehicle.Vehicle('tesla_rs3')
    data_points = data_processing.get_file_paths('tesla')
    vehicle.add_data_files(data_points)
    vehicle.build_record_file_data()
    fig = vehicle.get_plot_audio_data(0,8000)
    plot(fig, filename='./tmp/temp-plot.html', auto_open=True)


'''
Test dithering and plot
'''
def dither_audio_and_plot_test():
    vehicle = Vehicle.Vehicle('tesla_rs3')
    data_points = data_processing.get_file_paths('tesla')
    vehicle.add_data_files(data_points)
    vehicle.build_record_file_data()
    test_audio_data = vehicle.record_file_data[0]['audio'][0]
    dithered_audio = helpers.dither_audio(test_audio_data,1e-3)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=dithered_audio))

    # Setting the title, x-axis and y-axis labels
    fig.update_layout(
        title='Audio Data Visualization with dithering(noise level = 1e-3)',
        xaxis_title='Time',
        yaxis_title='Amplitude',
        # Assuming self.vehicle_name is a property of your class
        annotations=[{
            'text': f"Frequency: {8000}, Vehicle Name: tesla_rs3",
            'showarrow': False,
            'xref': 'paper',
            'yref': 'paper',
            'x': 0.5,
            'y': 0
        }]
    )

    plot(fig, filename='./tmp/temp-plot.html', auto_open=True)


if __name__ == "__main__":
    # test_vehicle_class()
    # test_vehicle_class_record_function()
    # make_plot_for_record()
    dither_audio_and_plot_test()






