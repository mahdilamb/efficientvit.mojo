from tensor import Tensor

# TODO conv2d
# TODO batchnorm
# TODO layer norm
# TODO linear


fn conv2d[
    dtype: DType, n: Int, height: Int, width: Int
](
    input: Tensor[dtype],
    weight: Tensor[dtype],
    bias: Tensor[dtype],
    stride: Tuple[Int, Int] = (1, 1),
    padding: Tuple[Int, Int] = (0, 0),
    dilation: Tuple[Int, Int] = (1, 1),
    groups: Int = 1,
) -> None:
    ...


fn main():
    ...
