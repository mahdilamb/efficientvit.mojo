from tensor import Tensor, TensorShape
from python import PythonObject, Python
from memory import memset, memset_zero
from algorithm import vectorize

alias NDArray = PythonObject


@always_inline
def to_shape(arr: PythonObject) -> TensorShape:
    if arr.shape.__len__() == 1:
        return TensorShape(arr.shape[0].__index__())
    if arr.shape.__len__() == 2:
        TensorShape(arr.shape[0].__index__(), arr.shape[1].__index__())
    if arr.shape.__len__() == 3:
        TensorShape(
            arr.shape[0].__index__(),
            arr.shape[1].__index__(),
            arr.shape[2].__index__(),
        )
    if arr.shape.__len__() == 4:
        TensorShape(
            arr.shape[0].__index__(),
            arr.shape[1].__index__(),
            arr.shape[2].__index__(),
            arr.shape[3].__index__(),
        )
    raise Error("Unsupported number of dims.")


def from_numpy_ptr[dtype: DType](arr: PythonObject) -> DTypePointer[dtype]:
    @parameter
    if dtype.is_float32():
        return DTypePointer[DType.float32](
            Pointer(
                __mlir_op.`pop.index_to_pointer`[
                    _type : __mlir_type[`!kgen.pointer<scalar<f32>>`]
                ](
                    SIMD[DType.index, 1](
                        arr.__array_interface__["data"][0].__index__()
                    ).value
                )
            ).address
        ).bitcast[dtype]()
    elif dtype.is_float64():
        return DTypePointer[DType.float64](
            Pointer(
                __mlir_op.`pop.index_to_pointer`[
                    _type : __mlir_type[`!kgen.pointer<scalar<f64>>`]
                ](
                    SIMD[DType.index, 1](
                        arr.__array_interface__["data"][0].__index__()
                    ).value
                )
            ).address
        ).bitcast[dtype]()
    raise Error("Unsupported dtype.")


@always_inline
def from_numpy[dtype: DType](arr: PythonObject) -> Tensor[dtype]:
    if arr.dtype != dtype.__str__():
        raise Error("Numpy array and parameter have different types.")
    let ptr = from_numpy_ptr[dtype](arr)
    return Tensor[dtype](
        ptr,
        to_shape(arr),
    )


fn to_numpy(x: DTypePointer[DType.float32], shape: TensorShape) raises -> NDArray:
    let np = Python.import_module("numpy")
    let ctypes = Python.import_module("ctypes")
    let ptr = Pointer(
        __mlir_op.`pop.index_to_pointer`[
            _type : __mlir_type[`!kgen.pointer<scalar<f32>>`]
        ](SIMD[DType.index, 1](x.__as_index()).value)
    )
    let c_ptr = ctypes.cast(ptr.__as_index(), ctypes.POINTER(ctypes.c_float))
    let np_shape = PythonObject([])
    for i in range(shape.rank()):
        _ = np_shape.append(shape[i])

    return np.ctypeslib.as_array(c_ptr, np_shape)


fn to_numpy(x: Tensor[DType.float32]) raises -> NDArray:
    return to_numpy(x.data(), x.shape())


@always_inline
fn full[dtype: DType](shape: TensorShape, value: SIMD[dtype, 1]) -> Tensor[dtype]:
    var out = Tensor[dtype](shape)

    @parameter
    fn op[simd_width: Int](i: Int):
        out[i] = value

    vectorize[1, op](shape.num_elements())
    return out


@always_inline
fn zeroes[dtype: DType](shape: TensorShape) -> Tensor[dtype]:
    let out = Tensor[dtype](shape)
    memset_zero(out.data(), shape.num_elements())
    return out


@always_inline
fn ones[dtype: DType](shape: TensorShape) -> Tensor[dtype]:
    return full[dtype](shape, 1.0)


@always_inline
fn arange[
    dtype: DType,
](stop: Int) -> Tensor[dtype]:
    var out = Tensor[dtype](TensorShape(stop))

    @parameter
    fn op[simd_width: Int](i: Int):
        out[i] = i

    vectorize[1, op](out.shape().num_elements())
    return out


@always_inline
fn arange[
    dtype: DType,
](start: Int, stop: Int) -> Tensor[dtype]:
    return arange[dtype](0, stop)
