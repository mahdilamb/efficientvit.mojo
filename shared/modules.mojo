from tensor import Tensor, TensorSpec, TensorShape
from memory import memset_zero


@always_inline
fn identity[dtype: DType](x: Tensor[dtype]) -> Tensor[dtype]:
    """Return the tensor, untouched."""
    return x
