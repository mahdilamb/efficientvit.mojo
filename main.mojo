from python import Python
import sys
from shared import pointers, strings
from utils.list import Dim, DimList
from utils.static_tuple import StaticTuple

alias DEFAULT_MODEL = "b3"
alias AVAILABLE_MODELS = StaticTuple[1, StringLiteral](
    "b3",
)
"""
fn load_images[
    n: Int, width: Int, height: Int
](paths: InlinedFixedVector[n, StringLiteral]) raises:
    let size: Int = n * 3 * width * height
    let interop = Python.import_module("interop.transforms")
    var path_list = Python.evaluate("list()")
    for i in range(n):
        path_list.append(paths[i])
    let tensor = interop.transform(
        (width, height), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )(path_list)
    let p = pointers.from_torch[DimList(Dim(n), Dim(3), Dim(width), Dim(height))](
        tensor
    )


fn load_images[n: Int, width: Int, height: Int]():
    ...


fn load_images224[n: Int]():
    load_images[n, 224, 224]()
"""


fn main() raises:
    Python.add_to_path(".")

    var imgs = DynamicVector[StringRef]()
    let argv = sys.argv()
    let n = len(argv)
    if n == 1 and argv[1] == "--help":
        # TODO: print help
        return
    var model: String = DEFAULT_MODEL
    var is_path: Bool = True
    for i in range(1, n):
        let arg = argv[i]
        if strings.startswith(arg, "--model"):
            is_path = False
            continue
        if not is_path:
            if not model:
                model = arg
            else:
                raise Error("Model is already supplied")
        else:
            imgs.push_back(arg)
    if not model:
        raise Error("Please supply the model by the --model parameter.")
    # TODO check model
    # TODO load images via interop
