import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd

class Quantizer(nn.Module):
    def __init__(self):
        super(Quantizer, self).__init__()

    def forward(self, inp, nbit, alpha=None, offset=None):
        self.scale = (2**nbit - 1) if alpha is None else (2**nbit - 1) / alpha

        if offset is None:
            out = torch.round(inp * self.scale) / self.scale
        else:
            out = (torch.round(inp * self.scale) + torch.round(offset)) / self.scale

        return out

    def backward(self, grad_output, grad_nbit, grad_alpha, grad_offset):
        print("entered backward")
        grad_input = grad_output.clone()
        print("grad_input: ", grad_input)
        return grad_input, None, None, None

def quantize(nbit):
    class Q(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            if nbit == 32:
                return x
            else:
                scale = 2 ** nbit - 1
                ctx.save_for_backward(x)
                return torch.round(x * scale) / scale

        @staticmethod
        def backward(ctx, grad_output):
            # x, = ctx.saved_tensors
            # print("grad_output: ", grad_output)
            grad_input = grad_output.clone()
            return grad_input
    return Q.apply

# Straight Through Estimator
class DynmQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, nbit):
        if nbit == 32:
            return x
        else:
            scale = 2 ** nbit - 1
            ctx.save_for_backward(x)
            return torch.round(x * scale) / scale

    @staticmethod
    def backward(ctx, grad_output):
        # x, = ctx.saved_tensors
        # print("grad_output: ", grad_output)
        grad_input = grad_output.clone()
        return grad_input,None

class DoReFaW(nn.Module):
    def __init__(self):
        super(DoReFaW, self).__init__()
        self.quantize = DynmQuantizer.apply

    def forward(self, inp, nbit_w, *args, **kwargs):
        """ forward pass """
        w = torch.tanh(inp)
        maxv = torch.abs(w).max()
        w = w / (2 * maxv) + 0.5
        w = (2 * self.quantize(w, nbit_w) - 1)
        return w

class DoReFaA(nn.Module):
    def __init__(self):
        super(DoReFaA, self).__init__()
        # self.quantize2 = quantize(nbit_a)
        self.quantize = DynmQuantizer.apply

    def forward(self, inp, nbit_a, *args, **kwargs):
        """ forward pass """
        a = torch.clamp(inp, 0, 1)
        a = self.quantize(a,nbit_a)
        return a

class SwitchBN2d(nn.Module):
    def __init__(self, num_features,eps=1e-05, momentum=0.1,affine=True):
        super(SwitchBN2d, self).__init__()
        self.num_features = num_features

        bit_list = [int(i) for i in range(1,33)]

        bns = []
        for i in range(len(bit_list)):
            bns.append(nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine))
        self.bn = nn.ModuleList(bns)

        self.bn_float = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine)
        self.bits = bit_list
        if affine:
            self.weight = None
            self.bias = None
        else:
            self.weight = nn.Parameter(torch.Tensor(self.num_features))
            self.bias = nn.Parameter(torch.Tensor(self.num_features))
        self.affine = affine
        self.quant_mode = True
        self.curr_bitwidth = None

    def forward(self, input):
        if self.quant_mode:
            assert self.curr_bitwidth is not None, "bitwidth is None"
            y = self.bn[self.curr_bitwidth-1](input)
            if not self.affine:
                assert self.weight is not None, "weight is None"
                assert self.bias is not None, "bias is None"
                y = self.weight[None, :, None, None] * y + self.bias[None, :, None, None]
        else:
            y = self.bn_float(input)
        return y

    def get_bitwidth(self):
        return self.curr_bitwidth

    def set_bitwidth(self, bitwidth):
        # print("Setting batchnorm bitwidth to: ", bitwidth)
        assert bitwidth in self.bits, "bitwidth not in self.bits, the layer is not initialized to handle this number of bitwidth"
        self.curr_bitwidth = bitwidth

class QuanConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, quan_name, nbit_w=32,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, has_offset=False, fix = False,batch_norm=False):
        super(QuanConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))

        nn.init.kaiming_uniform_(self.weight, mode= 'fan_out', nonlinearity= 'relu')
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.nbit_w = nbit_w
        self.nbit_a = nbit_w
        name_w_dict = {'dorefa': DoReFaW}
        name_a_dict = {'dorefa': DoReFaA}
        self.quan_w = name_w_dict[quan_name]()
        self.quan_a = name_a_dict[quan_name]()
        self.fix = fix

        if quan_name == 'pact':
            self.alpha_w = nn.Parameter(torch.ones(1))
        else:
            self.alpha_w = None

        if has_offset:
            self.offset = nn.Parameter(torch.zeros(1))
        else:
            self.offset = None
        
        self.batch_norm = batch_norm

    def forward(self, inp):
        if self.nbit_w < 32:
            w = self.quan_w(self.weight, self.nbit_w, self.alpha_w, self.offset)
        else:
            w = self.weight

        if self.nbit_a < 32:
            x = self.quan_a(inp, self.nbit_a)
        else:
            x = inp

        x = nn.functional.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

    def set_bitwidth(self, nbit):
        if not self.fix:
            self.nbit_w = nbit
            self.nbit_a = nbit

    def get_bitwidth(self):
        return self.nbit_w

class DualConvNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, quan_name, stride=1, padding=0, dilation=1, groups=1, bias=True, has_offset=False, batch_norm=False):
        super(DualConvNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups


        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', nonlinearity='relu')

        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        name_w_dict = {'dorefa': DoReFaW}
        name_a_dict = {'dorefa': DoReFaA, 'pact': DoReFaA}

        self.quan_w = name_w_dict[quan_name]()
        self.quan_a = name_a_dict[quan_name]()


        self.lower_band_ratio = None
        self.upper_band_ratio = None
        self.bitwidth_pattern_band = None
        self.bitwidth_non_pattern_band = None
        self.bitwidth_opts = None
        self.alpha_w = None
        self.offset = None


    def set_lower_upper_band_ratio(self, lower_band_ratio, upper_band_ratio):
        print("Setting lower and upper band ratio")
        self.lower_band_ratio = lower_band_ratio
        self.upper_band_ratio = upper_band_ratio
        print(f"Values set: {self.lower_band_ratio}, {self.upper_band_ratio}")
    

    def forward(self, input):
        assert self.lower_band_ratio is not None, "lower_band_ratio is None"
        assert self.upper_band_ratio is not None, "upper_band_ratio is None"
        # breakpoint()

        input_shape = input.shape

        height = input_shape[2]

        lower_bound_pattern = int(height * self.lower_band_ratio)
        upper_bound_pattern = int(height * self.upper_band_ratio)

        # create a mask
        # Create a mask for the pattern band
        pattern_mask = torch.zeros_like(input)
        pattern_mask[:, :, lower_bound_pattern:upper_bound_pattern, :] = 1

        # Create a mask for the non-pattern band
        non_pattern_mask = 1 - pattern_mask

        # Only input values between the pattern band should remain, make everything else zero
        input_w_patteern_band = input * pattern_mask

        input_w_non_pattern_band = input * non_pattern_mask

        w_pattern = self.quan_w(self.weight, self.bitwidth_pattern_band, self.alpha_w, self.offset)

        x_pattern_quantized = self.quan_a(input_w_patteern_band, self.bitwidth_pattern_band)

        output_pattern = nn.functional.conv2d(x_pattern_quantized, w_pattern, self.bias, self.stride, self.padding, self.dilation, self.groups)

        w_non_pattern = self.quan_w(self.weight, self.bitwidth_non_pattern_band, self.alpha_w, self.offset)

        x_non_pattern_quantized = self.quan_a(input_w_non_pattern_band, self.bitwidth_non_pattern_band)

        output_non_pattern = nn.functional.conv2d(x_non_pattern_quantized, w_non_pattern, self.bias, self.stride, self.padding, self.dilation, self.groups)

        final_output = output_pattern + output_non_pattern

        return final_output
        

# class DualQuanConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, quan_name, nbit_pattern_w=8, nbit_non_pattern_w=16,
#                  stride=1, padding=0, dilation=1, groups=1, bias=True, has_offset=False, batch_norm=False):
#         super(DualQuanConv, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
#         
#         # Create separate weight parameters for pattern and non-pattern bands
#         self.weight_pattern = nn.Parameter(torch.Tensor(out_channels, 1, *kernel_size))
#         self.weight_non_pattern = nn.Parameter(torch.Tensor(out_channels, in_channels - 1, *kernel_size))
#         
#         # Initialize the weight tensors using the Kaiming uniform initialization
#         nn.init.kaiming_uniform_(self.weight_pattern, mode='fan_out', nonlinearity='relu')
#         nn.init.kaiming_uniform_(self.weight_non_pattern, mode='fan_out', nonlinearity='relu')
#         
#         self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
#         
#         name_w_dict = {'dorefa': DoReFaW}
#         name_a_dict = {'dorefa': DoReFaA}
#         self.quan_w = name_w_dict[quan_name]()
#         self.quan_a = name_a_dict[quan_name]()
#         
#         self.nbit_pattern_w = nbit_pattern_w
#         self.nbit_non_pattern_w = nbit_non_pattern_w
#         
#         self.nbit_pattern_a = nbit_pattern_w
#         self.nbit_non_pattern_a = nbit_non_pattern_w
#
#         self.nbit_pattern_band = None
#         self.nbit_non_pattern_band = None
#
#         
#         self.alpha_w = None
#         self.offset = None
#
#     def forward(self, inp, pattern_idx=None, lower_band_ratio=None, upper_band_ratio=None):
#         shape_input = inp.shape
#
#         height = shape_input[2]
#
#         lower_bound_pattern = int(height * lower_band_ratio)
#         upper_bound_pattern = int(height * upper_band_ratio)
#
#         if pattern_idx is None:
#             # Case 1: No pattern band
#             # Quantize the non-pattern weights using the specified bitwidth
#             w_non_pattern = self.quan_w(self.weight_non_pattern, self.nbit_non_pattern_w, self.alpha_w, self.offset)
#             # Quantize the input activations using the specified bitwidth for non-pattern activations
#             x_non_pattern = self.quan_a(inp, self.nbit_non_pattern_a)
#             # Perform convolution using the quantized non-pattern weights and activations
#             x = nn.functional.conv2d(x_non_pattern, w_non_pattern, self.bias, self.stride, self.padding, self.dilation, self.groups)
#         else:
#             # Case 2 and Case 3: Pattern band exists
#             # Quantize the pattern weights using the specified bitwidth
#             w_pattern = self.quan_w(self.weight_pattern, self.nbit_pattern_w, self.alpha_w, self.offset)
#             # Quantize the input activations corresponding to the pattern band using the specified bitwidth for pattern activations
#             x_pattern = self.quan_a(inp[:, pattern_idx:pattern_idx+1, :, :], self.nbit_pattern_a)
#             
#             if pattern_idx == 0:
#                 # Case 2a: Pattern band is at the beginning
#                 # Quantize the non-pattern weights using the specified bitwidth
#                 w_non_pattern = self.quan_w(self.weight_non_pattern, self.nbit_non_pattern_w, self.alpha_w, self.offset)
#                 # Quantize the input activations corresponding to the non-pattern band using the specified bitwidth for non-pattern activations
#                 x_non_pattern = self.quan_a(inp[:, 1:, :, :], self.nbit_non_pattern_a)
#                 # Concatenate the quantized pattern weights and non-pattern weights along the output channel dimension
#                 w = torch.cat((w_pattern, w_non_pattern), dim=1)
#                 # Concatenate the quantized pattern activations and non-pattern activations along the input channel dimension
#                 x = torch.cat((x_pattern, x_non_pattern), dim=1)
#             elif pattern_idx == self.in_channels - 1:
#                 '''# Case 2b: Pattern band is at the end
#                 # Quantize the non-pattern weights using the specified bitwidth
#                 w_non_pattern = self.quan_w(self.weight_non_pattern, self.nbit_non_pattern_w, self.alpha_w, self.offset)
#                 # Quantize the input activations corresponding to the non-pattern band using the specified bitwidth for non-pattern activations
#                 x_non_pattern = self.quan_a(inp[:, :-1, :, :], self.nbit_non_pattern_a)
#                 # Concatenate the quantized non-pattern weights and pattern weights along the output channel dimension
#                 w = torch.cat((w_non_pattern, w_pattern), dim=1)
#                 # Concatenate the quantized non-pattern activations and pattern activations along the input channel dimension
#                 x = torch.cat((x_non_pattern, x_pattern), dim=1)'''
#                 # Case 2b: Pattern band is at the end
#                 # Treat it the same as Case 1 (no pattern band)
#                 w_non_pattern = self.quan_w(self.weight_non_pattern, self.nbit_non_pattern_w, self.alpha_w, self.offset)
#                 x_non_pattern = self.quan_a(inp, self.nbit_non_pattern_a)
#                 x = nn.functional.conv2d(x_non_pattern, w_non_pattern, self.bias, self.stride, self.padding, self.dilation, self.groups)
#
#             else:
#                 # Case 3: Pattern band is in the middle
#                 # Quantize the lower non-pattern weights using the specified bitwidth
#                 # w_non_pattern_lower = self.quan_w(self.weight_non_pattern[:, :lower_bound_pattern, :, :], self.nbit_non_pattern_w, self.alpha_w, self.offset)
#                 # # Quantize the upper non-pattern weights using the specified bitwidth
#                 # w_non_pattern_upper = self.quan_w(self.weight_non_pattern[:, upper_bound_pattern:, :, :], self.nbit_non_pattern_w, self.alpha_w, self.offset)
#                 # Quantize the input activations corresponding to the lower non-pattern band using the specified bitwidth for non-pattern activations
#                 x_non_pattern_lower = self.quan_a(inp[:, :lower_bound_pattern, :, :], self.nbit_non_pattern_a)
#                 # Quantize the input activations corresponding to the upper non-pattern band using the specified bitwidth for non-pattern activations
#                 x_non_pattern_upper = self.quan_a(inp[:, upper_bound_pattern:, :, :], self.nbit_non_pattern_a)
#
#                 x_pattern_bands = self.quan_a(inp[:, lower_bound_pattern:upper_bound_pattern, :, :], self.nbit_pattern_a)
#
#
#                 w_non_pattern = self.quan_w(self.weight_non_pattern, self.nbit_non_pattern_w, self.alpha_w, self.offset)
#
#                 w_pattern = self.quan_w(self.weight_pattern, self.nbit_pattern_w, self.alpha_w, self.offset)
#
#                 output_lower = nn.functional.conv2d(x_non_pattern_lower, w_non_pattern, self.bias, self.stride, self.padding, self.dilation, self.groups)
#                 output_upper = nn.functional.conv2d(x_non_pattern_upper, w_non_pattern, self.bias, self.stride, self.padding, self.dilation, self.groups)
#                 output_pattern = nn.functional.conv2d(x_pattern_bands, w_pattern, self.bias, self.stride, self.padding, self.dilation, self.groups)
#
#                 final_output = torch.cat((output_lower, output_pattern, output_upper), dim=1)
#
#                 return final_output
#                 # Concatenate the quantized lower non-pattern weights, pattern weights, and upper non-pattern weights along the output channel dimension
#                 w = torch.cat((w_non_pattern_lower, w_pattern, w_non_pattern_upper), dim=1)
#                 # Concatenate the quantized lower non-pattern activations, pattern activations, and upper non-pattern activations along the input channel dimension
#                 x = torch.cat((x_non_pattern_lower, x_pattern, x_non_pattern_upper), dim=1)
#
#
#
#
#             else:
#                 # Case 3: Pattern band is in the middle
#                 # Quantize the lower non-pattern weights using the specified bitwidth
#                 # w_non_pattern_lower = self.quan_w(self.weight_non_pattern[:, :self.lower_bound_pattern, :, :], self.nbit_non_pattern_w, self.alpha_w, self.offset)
#                 # # Quantize the upper non-pattern weights using the specified bitwidth
#                 # w_non_pattern_upper = self.quan_w(self.weight_non_pattern[:, self.upper_bound_pattern:, :, :], self.nbit_non_pattern_w, self.alpha_w, self.offset)
#                 #
#                 # w_pattern_bands = self.quan_w(self.weight_non_pattern[:, self.lower_bound_pattern:self.upper_bound_pattern, :, :], self.nbit_pattern_w, self.alpha_w, self.offset)
#                 # Quantize the input activations corresponding to the lower non-pattern band using the specified bitwidth for non-pattern activations
#                 x_non_pattern_lower = self.quan_a(inp[:, :self.lower_bound_pattern_activation, :, :], self.nbit_non_pattern_a)
#                 # Quantize the input activations corresponding to the upper non-pattern band using the specified bitwidth for non-pattern activations
#                 x_non_pattern_upper = self.quan_a(inp[:, self.upper_bound_pattern_activation:, :, :], self.nbit_non_pattern_a)
#
#                 x_pattern_bands = self.quan_a(inp[:, self.lower_bound_pattern_activation:self.upper_bound_pattern_activation, :, :], self.nbit_pattern_a)
#
#                 weight_non_pattern = self.quan_w(self.weight_non_pattern, self.nbit_non_pattern_w, self.alpha_w, self.offset)
#
#                 weight_pattern = self.quan_w(self.weight_pattern, self.nbit_pattern_w, self.alpha_w, self.offset)
#
#                 output_lower = nn.functional.conv2d(x_non_pattern_lower, weight_non_pattern, self.bias, self.stride, self.padding, self.dilation, self.groups)
#                 output_upper = nn.functional.conv2d(x_non_pattern_upper, weight_non_pattern, self.bias, self.stride, self.padding, self.dilation, self.groups)
#                 output_pattern = nn.functional.conv2d(x_pattern_bands, weight_pattern, self.bias, self.stride, self.padding, self.dilation, self.groups)
#
#
#                 #Stack the outputs
#                 x = torch.cat((output_lower, output_pattern, output_upper), dim=1)
#
#         return x
#                 
#
#
#                 # Concatenate the quantized lower non-pattern weights, pattern weights, and upper non-pattern weights along the output channel dimension
#                 w = torch.cat((w_non_pattern_lower, w_pattern, w_non_pattern_upper), dim=1)
#                 # Concatenate the quantized lower non-pattern activations, pattern activations, and upper non-pattern activations along the input channel dimension
#                 x = torch.cat((x_non_pattern_lower, x_pattern, x_non_pattern_upper), dim=1)
#             
#             # Perform convolution using the concatenated quantized weights and activations
#             x = nn.functional.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
#
#         return x
#
#     def set_bitwidth(self, nbit_pattern, nbit_non_pattern=None):
#         self.nbit_pattern_w = nbit_pattern
#         self.nbit_pattern_a = nbit_pattern
#         if nbit_non_pattern is not None:
#             self.nbit_non_pattern_w = nbit_non_pattern
#             self.nbit_non_pattern_a = nbit_non_pattern
#
#     def get_bitwidth(self):
#         return self.nbit_pattern_w, self.nbit_non_pattern_w
