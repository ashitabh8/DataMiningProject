
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Creating a CNN class
class ConvNeuralNetBasic(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeuralNetBasic, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5,1), stride=(2,1))
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization for the first convolutional layer
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5,1), stride=(2,1))
        self.bn2 = nn.BatchNorm2d(32)  # Batch normalization for the second convolutional layer
        self.max_pool1 = nn.MaxPool2d(kernel_size=(3,1), stride=(1,1))
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,1), stride=(1,1))
        self.bn3 = nn.BatchNorm2d(64)  # Batch normalization for the third convolutional layer
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,1), stride=(1,1))
        self.bn4 = nn.BatchNorm2d(64)  # Batch normalization for the fourth convolutional layer
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        
        self.fc1 = nn.Linear(4928, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.bn1(out)  # Apply batch normalization
        out = nn.ReLU()(out)  # Apply ReLU activation function

        out = self.conv_layer2(out)
        out = self.bn2(out)  # Apply batch normalization
        out = nn.ReLU()(out)  # Apply ReLU activation function
        out = self.max_pool1(out)
        
        out = self.conv_layer3(out)
        out = self.bn3(out)  # Apply batch normalization
        out = nn.ReLU()(out)  # Apply ReLU activation function

        # breakpoint()
        out = self.conv_layer4(out)
        out = self.bn4(out)  # Apply batch normalization
        out = nn.ReLU()(out)  # Apply ReLU activation function
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


class CNN_nodeep(nn.Module):
    def __init__(self, num_classes):
        super(CNN_nodeep, self).__init__()
        # Assuming the input shape is consistent with the output shapes provided.
        
        # breakpoint()
        self.conv2d_5 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), padding=1, stride = (1,1))
        self.relu_5 = nn.ReLU()
        self.batch_norm_5 = nn.BatchNorm2d(num_features=4)
        
        self.conv2d_6 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), padding=1)
        self.relu_6 = nn.ReLU()
        self.batch_norm_6 = nn.BatchNorm2d(num_features=8)
        self.max_pool2d_1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv2d_7 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1)
        self.relu_7 = nn.ReLU()
        self.batch_norm_7 = nn.BatchNorm2d(num_features=16)
        self.dropout_3 = nn.Dropout(p=0.3)
        
        self.conv2d_8 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1)
        self.relu_8 = nn.ReLU()
        self.batch_norm_8 = nn.BatchNorm2d(num_features=16)
        self.dropout_4 = nn.Dropout(p=0.1)
        
        self.conv2d_9 = nn.Conv2d(in_channels=16, out_channels=num_classes, kernel_size=(3, 3), padding=1)
        self.relu_9 = nn.ReLU()
        self.batch_norm_9 = nn.BatchNorm2d(num_features=num_classes)
        self.dropout_5 = nn.Dropout(p=0.1)
        
        self.global_avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        # self.activation = nn.Softmax(dim=1)  # Assuming classification problem

    def forward(self, x):
        x = self.conv2d_5(x)
        x = self.relu_5(x)
        x = self.batch_norm_5(x)
        
        x = self.conv2d_6(x)
        x = self.relu_6(x)
        x = self.batch_norm_6(x)
        x = self.max_pool2d_1(x)
        # breakpoint()
        
        x = self.conv2d_7(x)
        x = self.relu_7(x)
        x = self.batch_norm_7(x)
        x = self.dropout_3(x)
        
        x = self.conv2d_8(x)
        x = self.relu_8(x)
        x = self.batch_norm_8(x)
        x = self.dropout_4(x)
        
        x = self.conv2d_9(x)
        x = self.relu_9(x)
        x = self.batch_norm_9(x)
        x = self.dropout_5(x)
        
        x = self.global_avg_pool2d(x)
        # breakpoint()
        x = x.view(x.size(0), -1)  # Flatten the tensor
        # x = self.activation(x)
        
        return x





'''
Deeper version of the CNN_nodeep model
'''
class CNN_nodeep_v2(nn.Module):
    def __init__(self, num_classes):
        super(CNN_nodeep_v2, self).__init__()
        # Assuming the input shape is consistent with the output shapes provided.
        
        # breakpoint()
        self.conv2d_4 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), padding=1, stride = (1,1))
        self.relu_4 = nn.ReLU()
        self.batch_norm_4 = nn.BatchNorm2d(num_features=4)
        
        self.conv2d_5 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3, 3), padding=1, stride = (1,1))
        self.relu_5 = nn.ReLU()
        self.batch_norm_5 = nn.BatchNorm2d(num_features=4)


        self.conv2d_6 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), padding=1)
        self.relu_6 = nn.ReLU()
        self.batch_norm_6 = nn.BatchNorm2d(num_features=8)
        self.max_pool2d_1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv2d_7 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1)
        self.relu_7 = nn.ReLU()
        self.batch_norm_7 = nn.BatchNorm2d(num_features=16)
        self.dropout_3 = nn.Dropout(p=0.3)
        
        self.conv2d_8 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1)
        self.relu_8 = nn.ReLU()
        self.batch_norm_8 = nn.BatchNorm2d(num_features=16)
        self.dropout_4 = nn.Dropout(p=0.1)
        
        self.conv2d_9 = nn.Conv2d(in_channels=16, out_channels=num_classes, kernel_size=(3, 3), padding=1)
        self.relu_9 = nn.ReLU()
        self.batch_norm_9 = nn.BatchNorm2d(num_features=num_classes)
        self.dropout_5 = nn.Dropout(p=0.1)
        
        self.global_avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        # self.activation = nn.Softmax(dim=1)  # Assuming classification problem

    def forward(self, x):
        x = self.conv2d_4(x)
        x = self.relu_4(x)
        x = self.batch_norm_4(x)

        x = self.conv2d_5(x)
        x = self.relu_5(x)
        x = self.batch_norm_5(x)
        
        x = self.conv2d_6(x)
        x = self.relu_6(x)
        x = self.batch_norm_6(x)
        x = self.max_pool2d_1(x)
        # breakpoint()
        
        x = self.conv2d_7(x)
        x = self.relu_7(x)
        x = self.batch_norm_7(x)
        x = self.dropout_3(x)
        
        x = self.conv2d_8(x)
        x = self.relu_8(x)
        x = self.batch_norm_8(x)
        x = self.dropout_4(x)
        
        x = self.conv2d_9(x)
        x = self.relu_9(x)
        x = self.batch_norm_9(x)
        x = self.dropout_5(x)
        
        x = self.global_avg_pool2d(x)
        # breakpoint()
        x = x.view(x.size(0), -1)  # Flatten the tensor
        # x = self.activation(x)
        
        return x
