"""Module containing activation functions."""
#TODO vectorize
from tensor import Tensor, TensorSpec


fn hardswish[dtype: DType](tensor: Tensor[dtype]) -> Tensor[dtype]:
    """Apply the hardswish activation function to the copy of a tensor."""
    var output = tensor
    for i in range(output.num_elements()):
        if tensor[i] <= -3 or tensor[i] >= 3:
            output[i] = 0
        else:
            output[i] = tensor[i] * (tensor[i] + 3) / 6
    return output


fn relu[dtype: DType](tensor: Tensor[dtype]) -> Tensor[dtype]:
    """Apply the relu activation function to the copy of a tensor."""
    var output = tensor
    for i in range(output.num_elements()):
        if tensor[i] < 0:
            output[i] = 0
    return output


fn hardswish_[dtype: DType](inout tensor: Tensor[dtype]) -> Tensor[dtype]:
    """Apply the hardswish activation function to a tensor."""
    for i in range(tensor.num_elements()):
        if tensor[i] <= -3 or tensor[i] >= 3:
            tensor[i] = 0
        else:
            tensor[i] = tensor[i] * (tensor[i] + 3) / 6
    return tensor


fn relu_[dtype: DType](inout tensor: Tensor[dtype]) -> Tensor[dtype]:
    """Apply the relu activation function to a tensor."""
    for i in range(tensor.num_elements()):
        if tensor[i] < 0:
            tensor[i] = 0
    return tensor
