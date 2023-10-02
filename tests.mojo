from python import Python
from shared import pointers, strings, act
from testing import assert_equal, assert_false, assert_true
from tensor import Tensor, TensorSpec

alias Callable = fn () -> None
alias ThrowableCallable = fn () raises -> None


fn _run_test(name: String, func: Callable) -> None:
    print("Testing...", name)
    func()


fn _run_test(name: String, func: ThrowableCallable) -> None:
    print("Testing...", name)
    try:
        func()
    except e:
        print("Failed: ", e.value)


fn numpy_copied_test() raises:
    """Test that we can get a mojo buffer from a numpy array, that is copied into memory.
    """
    let np = Python.import_module("numpy")
    let a = np.random.rand(16).astype(np.float32)
    let p = pointers.from_numpy[16](a)
    for q in range(16):
        _ = assert_equal[DType.float64](
            a[q].to_float64(), p.load(q).cast[DType.float64]()
        )


fn numpy_shared_test() raises:
    """Test that we can get a mojo buffer from a numpy array, without copying."""
    let np = Python.import_module("numpy")
    let a = np.random.rand(16).astype(np.float32)
    let p = pointers.from_numpy[16](a, False)
    for q in range(16):
        _ = assert_equal[DType.float64](
            a[q].to_float64(), p.load(q).cast[DType.float64]()
        )


fn torch_copied_test() raises:
    """Test that we can get a mojo buffer from a torch array, that is copied into memory.
    """
    let torch = Python.import_module("torch")
    let a = torch.rand(16).type(torch.float32)
    let p = pointers.from_torch[16](a)
    for q in range(a.numel()):
        _ = assert_equal[DType.float64](
            a[q].to_float64(), p.load(q).cast[DType.float64]()
        )


fn torch_shared_test() raises:
    """Test that we can get a mojo buffer from a torch array, without copying."""
    let torch = Python.import_module("torch")
    let a = torch.rand(16).type(torch.float32)
    let p = pointers.from_torch[16](a, False)
    for q in range(a.numel()):
        _ = assert_equal[DType.float64](
            a[q].to_float64(), p.load(q).cast[DType.float64]()
        )


fn torch_v_relu() raises:
    let torch = Python.import_module("torch")
    let a = torch.rand(16).type(torch.float32).sub_(0.5).mul(2.0)
    let spec = TensorSpec(DType.float32, 16)
    let b = act.relu[DType.float32](Tensor(pointers.from_torch[16](a), spec))

    let c = torch.nn.functional.relu(a)
    for q in range(a.numel()):
        _ = assert_equal[DType.float64](c[q].to_float64(), b[q].cast[DType.float64]())


fn torch_v_hardswish() raises:
    let torch = Python.import_module("torch")
    let a = torch.rand(16).type(torch.float32).sub_(0.5).mul(5.0)
    let spec = TensorSpec(DType.float32, 16)
    let b = act.hardswish[DType.float32](Tensor(pointers.from_torch[16](a), spec))

    let c = torch.nn.functional.hardswish(a)
    for q in range(a.numel()):
        _ = assert_equal[DType.float64](c[q].to_float64(), b[q].cast[DType.float64]())


fn test_starts_with() raises:
    _ = assert_true(
        strings.startswith("Abcd", "Abc"), "Expected 'Abcd' to start with 'Abc'"
    )
    _ = assert_false(
        strings.startswith("Abcd", "abc"), "Expected 'Abcd' not to start with 'abc'"
    )


fn main() raises:
    """Run the tests in the module."""
    _run_test("numpy_copied_test", numpy_copied_test)
    _run_test("numpy_shared_test", numpy_shared_test)
    _run_test("torch_copied_test", torch_copied_test)
    _run_test("torch_shared_test", torch_shared_test)
    _run_test("torch_v_relu", torch_v_relu)
    _run_test("torch_v_hardswish", torch_v_hardswish)
    _run_test("test_starts_with", test_starts_with)
