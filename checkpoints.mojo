from memory.buffer import Buffer
from python.object import PythonObject
from python import Python
from shared import strings
from utils.list import Dim, DimList
from sys.info import sizeof
from tensor import TensorShape, Tensor
from python import Python, Dictionary
from utils.static_tuple import StaticTuple
from shared import modules

alias SEEK_SET: Int = 0
alias SEEK_CUR: Int = SEEK_SET + 1
alias SEEK_END: Int = SEEK_CUR + 1


@value
struct EffVitHeader:
    var model: String
    var width: Int
    var height: Int
    var num_weights: Int
    var type: String

    @staticmethod
    def from_string(owned ptr: Pointer[UInt8]) -> EffVitHeader:
        let frags = strings.split(ptr, ";")
        var i = 0
        var frag = frags[i]
        let kv: Pointer[Pointer[UInt8]]
        let k: String
        var out = EffVitHeader("", -1, -1, -1, "")
        while frag != frag.get_null():
            kv = strings.split(frag, "=")
            k = strings.from_ptr(kv[0])
            if k == "model":
                out.model = strings.from_ptr(kv[1])
            elif k == "width":
                out.width = strings.atol(kv[1])
            elif k == "height":
                out.height = strings.atol(kv[1])
            elif k == "type":
                out.type = strings.from_ptr(kv[1])
            elif k == "num_weights":
                out.num_weights = strings.atol(kv[1])
            i += 1
            frag = frags[i]
        return out

    def __repr__(self) -> String:
        var out: String = "Header {model="
        out += (
            self.model
            + ", size=("
            + String(self.height)
            + ", "
            + String(self.width)
            + "), dtype="
            + self.type
            + "}"
        )
        return out

    def __str__(self) -> String:
        return self.__repr__()


@always_inline
fn to_shape(ptr: Pointer[UInt8]) raises -> TensorShape:
    let str = strings.split(strings.from_ptr(ptr), ",")
    var d_ptr: Pointer[UInt8]
    var i = 0
    var dims = DynamicVector[Int]()
    while True:
        d_ptr = str[i]
        if d_ptr == d_ptr.get_null():
            break
        dims.push_back(strings.atol(d_ptr))
        i += 1
    if dims.size == 1:
        return TensorShape(dims[0])
    if dims.size == 2:
        return TensorShape(dims[0], dims[1])
    if dims.size == 3:
        return TensorShape(dims[0], dims[1], dims[2])
    if dims.size == 4:
        return TensorShape(dims[0], dims[1], dims[2], dims[3])
    raise Error("Unsupported dimension size")


@value
struct WeightsInfo:
    var name: String
    var type: String
    var shape: TensorShape
    var bytes: Int

    @staticmethod
    def from_string(owned ptr: Pointer[UInt8]) -> WeightsInfo:
        let frags = strings.split(ptr, ";")
        var i = 0
        var frag = frags[i]
        let kv: Pointer[Pointer[UInt8]]
        let k: String
        var out = WeightsInfo("", "", TensorShape(), -1)
        while frag != frag.get_null():
            kv = strings.split(frag, "=")
            k = strings.from_ptr(kv[0])
            if k == "name":
                out.name = strings.from_ptr(kv[1])
            elif k == "type":
                out.type = strings.from_ptr(kv[1])
            elif k == "shape":
                out.shape = to_shape(kv[1])
            elif k == "bytes":
                out.bytes = strings.atol(kv[1])
            i += 1
            frag = frags[i]
        return out

    def __repr__(self) -> String:
        var out: String = "Weights {name="
        out += (
            self.name
            + ", shape="
            + self.shape.__repr__()
            + ", bytes="
            + self.bytes
            + ", type="
            + self.type
            + "}"
        )
        return out

    def __str__(self) -> String:
        return self.__repr__()


@value
@register_passable("trivial")
struct FILE:
    ...


struct Weights:
    var _index: Dictionary
    var _fp: Pointer[FILE]

    def __init__(inout self, owned index: Dictionary, owned fp: Pointer[FILE]):
        self._index = index
        self._fp = fp

    @always_inline
    fn get[dtype: DType](self, name: String) raises -> Tensor[dtype]:
        let pos = self._index[name].__index__()
        _ = external_call["fseek", Int, Pointer[FILE], Int64, Int](
            self._fp, pos, SEEK_SET
        )
        let ptr = external_call[
            "fgets", Pointer[UInt8], Pointer[UInt8], Int32, Pointer[FILE]
        ](Pointer[UInt8]().alloc(128), 128, self._fp)
        ptr.store(strings.len(ptr) - 1, ord("\0"))
        let info = WeightsInfo.from_string(ptr)
        if info.type != dtype.__str__():
            raise Error("Invalid dtype supplied.")
        let data = DTypePointer[DType.uint8].alloc(info.bytes)
        _ = external_call[
            "fread", Int, DTypePointer[DType.uint8], Int, Int, Pointer[FILE]
        ](data, 1, info.bytes, self._fp)
        return Tensor(data.bitcast[dtype](), info.shape)


fn load(path: String) raises -> Weights:
    """Read an EfficientViT checkpoint file."""
    let fp = external_call["fopen", Pointer[FILE], Pointer[UInt8], Pointer[UInt8]](
        strings.to_ptr(path), strings.to_ptr("rb")
    )
    var pos: Int = 0

    @parameter
    fn read_line[size: Int]() -> Pointer[UInt8]:
        let ptr = external_call[
            "fgets", Pointer[UInt8], Pointer[UInt8], Int32, Pointer[FILE]
        ](Pointer[UInt8]().alloc(size), size, fp)
        let t = external_call["ftell", Int, Pointer[FILE]](fp)
        ptr.store(t - pos - 1, ord("\0"))
        pos = t
        return ptr

    let headers = EffVitHeader.from_string(read_line[128]())
    let index = Python.dict()
    for i in range(headers.num_weights):
        let qpos = pos
        let info = WeightsInfo.from_string(read_line[512]())
        index[info.name] = qpos
        _ = external_call["fseek", Int, Pointer[FILE], Int64, Int](
            fp, info.bytes, SEEK_CUR
        )
        pos = external_call["ftell", Int, Pointer[FILE]](fp)
    return Weights(index ^, fp ^)
