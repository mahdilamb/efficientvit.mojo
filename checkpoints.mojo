from memory.buffer import Buffer
from python.object import PythonObject
from python import Python
from shared import strings
from utils.list import Dim, DimList


@value
struct EffVitHeader:
    var model: String
    var width: Int
    var height: Int


@value
@register_passable("trivial")
struct FILE:
    ...


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
    print(header.model)


fn main() raises:
    print(read("assets/checkpoints/ImageNet-EfficientViT-B3-224x224.effvit.bin"))
