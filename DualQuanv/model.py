import torch
import torch.nn as nn
from  layers import DoReFaW, DoReFaA, QuanConv, SwitchBN2d, DualConvNew

class QCNN_Conv_Relu_split(nn.Module):
    def __init__(self, num_classes):
        super(QCNN_Conv_Relu_split, self).__init__()
        # self.conv1 = QuanConv(in_channels=1, out_channels=16, kernel_size=(3, 3), quan_name = 'dorefa', nbit_w=32, stride=1, padding=1, bias=True, has_offset=False, fix = True)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding=1, bias=True)
        self.relu1 = nn.ReLU()
        self.conv2 = QuanConv(in_channels=16, out_channels=8, kernel_size=(3, 3), quan_name = 'dorefa', nbit_w=8, stride=1, padding=1, bias=True, has_offset=False)
        self.relu2 = nn.ReLU()
        self.max_pool2D_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = QuanConv(in_channels=8, out_channels=8, kernel_size=(3, 3), quan_name = 'dorefa', nbit_w=8, stride=1, padding=1, bias=True, has_offset=False)
        self.relu3 = nn.ReLU()
        self.dropout_3 = nn.Dropout(p=0.3)
        self.conv4 = QuanConv(in_channels=8, out_channels=16, kernel_size=(3, 3), quan_name = 'dorefa', nbit_w=8, stride=1, padding=1, bias=True, has_offset=False)
        self.relu4 = nn.ReLU()
        self.dropout_2 = nn.Dropout(p=0.2)
        self.conv5 = QuanConv(in_channels=16, out_channels=16, kernel_size=(3, 3), quan_name = 'dorefa', nbit_w=8, stride=1, padding=1, bias=True, has_offset=False)
        self.relu5 = nn.ReLU()
        self.dropout_1 = nn.Dropout(p=0.1)
        self.conv6 = QuanConv(in_channels=16, out_channels=8, kernel_size=(3, 3), quan_name = 'dorefa', nbit_w=8, stride=1, padding=1, bias=True, has_offset=False)
        self.relu6 = nn.ReLU()
        # self.conv7 = QuanConv(in_channels=8, out_channels=num_classes, kernel_size=(3, 3), quan_name = 'dorefa', nbit_w=32, stride=1, padding=1, bias=True, has_offset=False, fix = True)
        self.conv7 = nn.Conv2d(in_channels=8, out_channels=num_classes, kernel_size=(3, 3), stride=1, padding=1, bias=True)
        self.relu7 = nn.ReLU()
        # self.global_avg_pool2d = QAdaptiveAvgPool2d(output_size=(1, 1))
        # self.global_avg_pool2d.last_layer = True
        self.global_avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling

    def forward(self, x):
        # breakpoint()
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        # breakpoint()
        x = self.max_pool2D_1(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.dropout_3(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.dropout_2(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.dropout_1(x)
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.global_avg_pool2d(x)
        x = x.view(x.size(0), -1)
        return x

class QCNN_ReLU_BN(nn.Module):
    def __init__(self, num_classes):
        super(QCNN_ReLU_BN, self).__init__()
        # self.conv1 = QuanConv(in_channels=1, out_channels=16, kernel_size=(3, 3), quan_name = 'dorefa', nbit_w=32, stride=1, padding=1, bias=True, has_offset=False, fix = True)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = DualConvNew(in_channels=16, out_channels=8, kernel_size=(3, 3), quan_name = 'dorefa',  stride=1, padding=1, bias=True, has_offset=False)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU()
        self.max_pool2D_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = DualConvNew(in_channels=8, out_channels=8, kernel_size=(3, 3), quan_name = 'dorefa',  stride=1, padding=1, bias=True, has_offset=False)
        self.bn3 = nn.BatchNorm2d(8)
        self.relu3 = nn.ReLU()
        self.dropout_3 = nn.Dropout(p=0.3)
        self.conv4 = DualConvNew(in_channels=8, out_channels=16, kernel_size=(3, 3), quan_name = 'dorefa',  stride=1, padding=1, bias=True, has_offset=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU()
        self.dropout_2 = nn.Dropout(p=0.2)
        self.conv5 = DualConvNew(in_channels=16, out_channels=16, kernel_size=(3, 3), quan_name = 'dorefa',  stride=1, padding=1, bias=True, has_offset=False)
        self.bn5 = nn.BatchNorm2d(16)
        self.relu5 = nn.ReLU()
        self.dropout_1 = nn.Dropout(p=0.1)
        self.conv6 = DualConvNew(in_channels=16, out_channels=8, kernel_size=(3, 3), quan_name = 'dorefa',  stride=1, padding=1, bias=True, has_offset=False)
        self.bn6 = nn.BatchNorm2d(8)
        self.relu6 = nn.ReLU()
        # self.conv7 = QuanConv(in_channels=8, out_channels=num_classes, kernel_size=(3, 3), quan_name = 'dorefa', nbit_w=32, stride=1, padding=1, bias=True, has_offset=False, fix = True)
        self.conv7 = nn.Conv2d(in_channels=8, out_channels=num_classes, kernel_size=(3, 3), stride=1, padding=1, bias=True)
        self.bn7 = nn.BatchNorm2d(num_classes)
        self.relu7 = nn.ReLU()
        # self.global_avg_pool2d = QAdaptiveAvgPool2d(output_size=(1, 1))
        # self.global_avg_pool2d.last_layer = True
        self.global_avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling

    def forward(self, x):
        # breakpoint()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        # breakpoint()
        x = self.max_pool2D_1(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout_3(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.dropout_2(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.dropout_1(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)
        x = self.global_avg_pool2d(x)
        x = x.view(x.size(0), -1)
        return x
    
class QCNN_DualQuant(nn.Module):
    def __init__(self, num_classes):
        super(QCNN_DualQuant, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding=1, bias=True)
        self.relu1 = nn.ReLU()
        self.conv2 = DualQuanConv(in_channels=16, out_channels=8, kernel_size=(3, 3), quan_name='dorefa', stride=1, padding=1, bias=True, has_offset=False)
        self.relu2 = nn.ReLU()
        self.max_pool2D_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = DualQuanConv(in_channels=8, out_channels=8, kernel_size=(3, 3), quan_name='dorefa', stride=1, padding=1, bias=True, has_offset=False)
        self.relu3 = nn.ReLU()
        self.dropout_3 = nn.Dropout(p=0.3)
        self.conv4 = DualQuanConv(in_channels=8, out_channels=16, kernel_size=(3, 3), quan_name='dorefa', stride=1, padding=1, bias=True, has_offset=False)
        self.relu4 = nn.ReLU()
        self.dropout_2 = nn.Dropout(p=0.2)
        self.conv5 = DualQuanConv(in_channels=16, out_channels=16, kernel_size=(3, 3), quan_name='dorefa', stride=1, padding=1, bias=True, has_offset=False)
        self.relu5 = nn.ReLU()
        self.dropout_1 = nn.Dropout(p=0.1)
        self.conv6 = DualQuanConv(in_channels=16, out_channels=8, kernel_size=(3, 3), quan_name='dorefa', stride=1, padding=1, bias=True, has_offset=False)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(in_channels=8, out_channels=num_classes, kernel_size=(3, 3), stride=1, padding=1, bias=True)
        self.relu7 = nn.ReLU()
        self.global_avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, pattern_idx=None, lower_bound_ratio=None, upper_bound_ratio=None):
        x = self.conv1(x,lower_bound_ratio=lower_bound_ratio, upper_bound_ratio=upper_bound_ratio)
        x = self.relu1(x)
        x = self.conv2(x, lower_bound_ratio=lower_bound_ratio, upper_bound_ratio=upper_bound_ratio)
        x = self.relu2(x)
        x = self.conv3(x, pattern_idx)
        x = self.relu3(x)
        x = self.max_pool2D_1(x)
        x = self.conv4(x, pattern_idx)
        x = self.relu4(x)
        x = self.dropout_3(x)
        x = self.conv5(x, pattern_idx)
        x = self.relu5(x)
        x = self.dropout_2(x)
        x = self.conv6(x, pattern_idx)
        x = self.relu6(x)
        x = self.dropout_1(x)
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.global_avg_pool2d(x)
        x = x.view(x.size(0), -1)
        return x
