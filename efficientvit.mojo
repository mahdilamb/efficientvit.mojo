import checkpoints
from tensor import Tensor, TensorShape
from python import Python, PythonObject
from shared import imagenet, pointers, images

alias B3 = backbone[5](StaticIntTuple[5](1, 4, 6, 6, 9), 32)


fn load_images[
    n: Int, width: Int, height: Int
](paths: DynamicVector[StringRef],) raises -> Tensor[DType.float32]:
    """Load images via python."""
    if n != len(paths):
        raise Error("Number of paths does not match function parameter `n`.")
    let torch = Python.import_module("torch")
    let np = Python.import_module("numpy")
    let Image = Python.import_module("PIL.Image")
    let F = Python.import_module("torchvision.transforms.functional")

    let imgs = PythonObject([])
    for i in range(len(paths)):
        var img = Image.open(paths[i])
        img = F.resize(
            img,
            np.ceil(np.asarray((width, height)) / 0.95).astype("int").tolist(),
            F.InterpolationMode.BICUBIC,
        )
        img = F.center_crop(img, (width, height))
        img = F.to_tensor(img)
        img = F.normalize(img, imagenet.MEAN, imagenet.STD)
        _ = imgs.append(img)

    let ptr = pointers.from_numpy(np.ascontiguousarray(imgs).astype("float32"))
    var data = DTypePointer[DType.float32].alloc(width * height * 3 * n)
    memcpy[DType.float32, width * height * 3 * n](data, ptr)
    let output = Tensor[DType.float32](data, TensorShape(len(paths), 3, height, width))

    return output


fn backbone[
    num_depths: Int
](
    depths: StaticIntTuple[num_depths],
    dims: Int,
    expand_ratio: Int = 4,
    norm: StringLiteral = "batch",
    act: StringLiteral = "hardswish",
):
    ...


fn cls_head(weights: checkpoints.Weights):
    ...


fn main() raises:
    try:
        var imgs = DynamicVector[StringRef]()
        imgs.push_back("test_data/image.jpg")
        let arr = load_images[1, 224, 224](imgs)
        images.save_image[0]("test.png", arr)
        Python.throw_python_exception_if_error_state()
    except e:
        print(e.value)
