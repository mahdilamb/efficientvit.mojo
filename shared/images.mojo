from tensor import Tensor, TensorShape
from shared import tensors
from python import Python


fn save_image[index: Int](path: String, img: Tensor[DType.float32]) raises:
    """Save a tensor as an image."""
    let np = Python.import_module("numpy")
    let Image = Python.import_module("PIL.Image")
    let pixels = 3 * img.shape()[2] * img.shape()[3]
    let offset = index * pixels
    let ptr = img.data().offset(offset)

    let arr = np.transpose(
        np.clip(
            (
                (
                    tensors.to_numpy(
                        ptr, TensorShape(3, img.shape()[2], img.shape()[3])
                    )
                    * 255
                )
                + 0.5
            ),
            0,
            255,
        ).astype("uint8"),
        (1, 2, 0),
    )

    _ = Image.fromarray(arr).save(path)
