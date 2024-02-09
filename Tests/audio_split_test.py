# import Utils.Vehicle as Vehicle 
from Utils import Vehicle, data_processing, helpers
# from Utils.data_processing import get_file_paths
# from globals import DATA_DIR_PATH
import plotly.graph_objs as go
from plotly.offline import plot

'''
Testing Audio split function

Expected output:
type audio_sample_0:  <class 'tuple'>
Len audio sample 0:  8
Shape audio sample 0:  torch.Size([2000])
Shape audio sample 0:  torch.Size([2000])
Shape audio sample 0:  torch.Size([2000])
Shape audio sample 0:  torch.Size([2000])
Shape audio sample 0:  torch.Size([2000])
Shape audio sample 0:  torch.Size([2000])
Shape audio sample 0:  torch.Size([2000])
Shape audio sample 0:  torch.Size([2000])
'''
def test_audio_split(vehicle):
    audio_sample_0 = vehicle.get_audio_data(0, 0.25)
    print("type audio_sample_0: ", type(audio_sample_0))
    print("Len audio sample 0: ", len(audio_sample_0))
    for i in range(len(audio_sample_0)):
        print("Shape audio sample 0: ", audio_sample_0[i].shape)


if __name__ == "__main__":
    vehicle = Vehicle.Vehicle('tesla_rs3')
    data_points = data_processing.get_file_paths('tesla')
    vehicle.add_data_files(data_points)
    vehicle.build_record_file_data()
    test_audio_split(vehicle)


