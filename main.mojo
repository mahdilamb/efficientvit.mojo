from python import Python
from sys import argv

# (0.485, 0.456, 0.406),(0.229, 0.224, 0.225)

"""
fn load_images[n:Int, width:Int,height:Int](paths:InlinedFixedVector[n, StringLiteral]) raises:
    let size:Int = n* 3 * width*height
    let interop = Python.import_module("interop.transforms")
    var path_list = Python.evaluate("list()")
    for i in range(n):
        path_list.append(paths[i])
    let tensor = interop.transform((width,height),(0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(path_list)
    let p = pointers.from_torch[DimList(Dim(n),Dim(3),Dim(width),Dim(height))](tensor)
"""


fn load_images[n: Int, width: Int, height: Int]():
    ...


fn load_images224[n: Int]():
    load_images[n, 224, 224]()


fn main() raises:
    Python.add_to_path(".")
    let n = argv().__len__() - 1
    print(n)
    # var blah = InlinedFixedVector[n,StringLiteral](n)
    # blah[0] = 'tests/data/dog.jpg'

    # load_images[1,244,244](blah)
