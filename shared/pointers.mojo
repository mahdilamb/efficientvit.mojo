from python import Python
from python.object import PythonObject


def from_numpy(arr: PythonObject) -> DTypePointer[DType.float32]:
    """Copy data from numpy into a pointer."""
    let arr_p = Pointer(
        __mlir_op.`pop.index_to_pointer`[
            _type : __mlir_type[`!kgen.pointer<scalar<f32>>`]
        ](SIMD[DType.index, 1](arr.__array_interface__["data"][0].__index__()).value)
    )
    return DTypePointer[DType.float32](arr_p.address)


def from_numpy[size: Int](arr: PythonObject) -> DTypePointer[DType.float32]:
    """Copy data from numpy into a pointer."""
    let arr_p = Pointer(
        __mlir_op.`pop.index_to_pointer`[
            _type : __mlir_type[`!kgen.pointer<scalar<f32>>`]
        ](SIMD[DType.index, 1](arr.__array_interface__["data"][0].__index__()).value)
    )
    let p = DTypePointer[DType.float32].alloc(size)
    memcpy[DType.float32, size](p, arr_p)
    return p


def from_torch[size: Int](arr: PythonObject) -> DTypePointer[DType.float32]:
    """Copy data from pytorch into a pointer."""
    let arr_p = Pointer(
        __mlir_op.`pop.index_to_pointer`[
            _type : __mlir_type[`!kgen.pointer<scalar<f32>>`]
        ](SIMD[DType.index, 1](arr.data_ptr().__index__()).value)
    )
    let p = DTypePointer[DType.float32].alloc(size)
    memcpy[DType.float32, size](p, arr_p)
    return p


def from_torch(arr: PythonObject) -> DTypePointer[DType.float32]:
    """Copy data from pytorch into a pointer."""
    let arr_p = Pointer(
        __mlir_op.`pop.index_to_pointer`[
            _type : __mlir_type[`!kgen.pointer<scalar<f32>>`]
        ](SIMD[DType.index, 1](arr.data_ptr().__index__()).value)
    )
    return DTypePointer[DType.float32](arr_p.address)
