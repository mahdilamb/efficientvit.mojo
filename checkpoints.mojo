from memory.buffer import Buffer
from python.object import PythonObject
from python import Python
from shared import strings
from utils.list import Dim, DimList
from sys.info import sizeof


@value
struct EffVitHeader:
    var model: String
    var width: Int
    var height: Int
@value
struct WeightsInfo:
    var name: String
    var shape: DimList
    var type: String
    var bytes: Int

@value
@register_passable("trivial")
struct FILE:
    ...
fn to_dim_list(ptr:Pointer[UInt8]) -> DimList:
    let dims = strings.split(strings.from_ptr(ptr)[1:-1],",")
    var i = 0
    var dim : Pointer[UInt8]
    while True:
        dim = dims[i]
        if dim == dim.get_null():
            break
        strings.prints(dim)
        i+=1
    return DimList()

fn read(path: StringLiteral) raises -> None:
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
        let out = ptr.alloc(t - pos)
        memcpy[UInt8](out, ptr, t - pos)
        pos += t
        return out

    let headers = strings.split(read_line[128](), ";")
    let num_weights = strings.atol(strings.split(headers[5], "=")[1])
    let header = EffVitHeader(
        strings.from_ptr(strings.split(headers[2], "=")[1]),
        strings.atol(strings.split(headers[3], "=")[1]),
        strings.atol(strings.split(headers[4], "=")[1]),
    )
    for i in range(num_weights):
        let weights_header = strings.split(read_line[128](), ";")
        let info = WeightsInfo(
            strings.from_ptr(strings.split(weights_header[0], "=")[1]),
            to_dim_list(strings.split(weights_header[1], "=")[1]),
            strings.from_ptr(strings.split(weights_header[2], "=")[1]),
            strings.atol(strings.split(weights_header[3], "=")[1]),
        )
        print(info.bytes)



fn main() raises:
    print(read("assets/checkpoints/ImageNet-EfficientViT-B3-224x224.effvit.bin"))
