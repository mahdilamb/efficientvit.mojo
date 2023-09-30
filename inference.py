# TODO convert to mojo
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
from typing import Literal, Callable, TypeAlias, Any
import torch.nn.functional as F
import functools

Layer: TypeAlias = Callable[[torch.Tensor], torch.Tensor]
Act: TypeAlias = Literal["relu", "hardswish"] | None
Norm: TypeAlias = Literal["batch", "layer"] | None
image_size = np.asarray((224, 224))
img_transforms = transforms.Compose(
    [
        transforms.Resize(
            np.ceil(image_size / 0.95).astype(int).tolist(),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def identity(x: torch.Tensor):
    return x


def act_by_name(act: Act) -> Layer | None:
    if act == "hardswish":
        return F.hardswish
    elif act == "relu":
        return F.relu
    elif act is not None:
        raise ValueError(f"Unsupported act: {act}")


def norm_by_name(norm: Norm, weights: dict[str, torch.Tensor], location: str, **kwargs):
    if norm == "batch":
        return functools.partial(
            F.batch_norm,
            running_mean=weights[f"{location}.running_mean"],
            running_var=weights[f"{location}.running_var"],
            weight=weights[f"{location}.weight"],
            bias=weights.get(f"{location}.bias"),
            **kwargs,
        )
    elif norm == "layer":
        return functools.partial(
            F.layer_norm,
            weight=weights[f"{location}.weight"],
            bias=weights.get(f"{location}.bias"),
            **kwargs,
        )
    elif norm is not None:
        raise ValueError(f"Unsupported norm: {norm}")


def padding_from_kernel_size(kernel_size: tuple[int, int], dilation: int):
    return [(k // 2) * dilation for k in kernel_size]


def padding_from_conv_weights(
    weights: dict[str, torch.Tensor], location: str, dilation
):
    return padding_from_kernel_size(weights[f"{location}.weight"].shape[-2:], dilation)


def conv_layer(
    weights: dict[str, torch.Tensor],
    location: str,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    norm: Norm = "batch",
    act: Act = "hardswish",
):
    padding = padding_from_conv_weights(weights, f"{location}.conv", dilation)
    layers: list[Layer] = [
        functools.partial(
            F.conv2d,
            weight=weights[f"{location}.conv.weight"],
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=weights.get(f"{location}.conv.bias"),
        )
    ]
    norm_layer = norm_by_name(norm, weights, f"{location}.norm")
    if norm_layer:
        layers.append(norm_layer)

    act_fn = act_by_name(act)
    if act_fn:
        layers.append(act_fn)
    return sequential(*layers)


def residual_block(main: Layer, shortcut: Layer | None):
    if shortcut is None:
        return main

    def with_shortcut(x: torch.Tensor):
        return main(x) + shortcut(x)

    return with_shortcut


def sequential(*layers: Callable):
    def call(x: torch.Tensor):
        for layer in layers:
            x = layer(x)
        return x

    return call


def dsconv(
    weights: dict[str, torch.Tensor],
    location: str,
    act: tuple[Act, Act],
    stride: tuple[int, int],
    norm: tuple[Norm, Norm],
):
    return sequential(
        conv_layer(
            weights,
            f"{location}.depth_conv",
            stride=stride[0],
            groups=weights[f"{location}.depth_conv.conv.weight"].shape[0],
            norm=norm[0],
            act=act[0],
        ),
        conv_layer(
            weights,
            f"{location}.point_conv",
            stride=stride[1],
            norm=norm[1],
            act=act[1],
        ),
    )


def mbconv(
    weights: dict[str, torch.Tensor],
    location: str,
    act: tuple[Act, Act, Act],
    stride: int,
    norm: tuple[Norm, Norm, Norm],
):
    return sequential(
        conv_layer(
            weights,
            f"{location}.inverted_conv",
            stride=1,
            norm=norm[0],
            act=act[0],
        ),
        conv_layer(
            weights,
            f"{location}.depth_conv",
            stride=stride,
            groups=weights[f"{location}.depth_conv.conv.weight"].shape[0],
            norm=norm[1],
            act=act[1],
        ),
        conv_layer(weights, f"{location}.point_conv", norm=norm[2], act=act[2]),
    )


@torch.autocast("cuda", enabled=False)
def relu_linear_att(
    qkv: torch.Tensor, dim: int, eps: float, kernel_func: Layer
) -> torch.Tensor:
    B, _, H, W = list(qkv.size())

    if qkv.dtype == torch.float16:
        qkv = qkv.float()

    qkv = torch.reshape(
        qkv,
        (
            B,
            -1,
            3 * dim,
            H * W,
        ),
    )
    qkv = torch.transpose(qkv, -1, -2)
    q, k, v = (
        qkv[..., 0:dim],
        qkv[..., dim : 2 * dim],
        qkv[..., 2 * dim :],
    )

    # lightweight linear attention
    q = kernel_func(q)
    k = kernel_func(k)

    # linear matmul
    trans_k = k.transpose(-1, -2)

    v = F.pad(v, (0, 1), mode="constant", value=1)
    kv = torch.matmul(trans_k, v)
    out = torch.matmul(q, kv)
    out = out[..., :-1] / (out[..., -1:] + eps)

    out = torch.transpose(out, -1, -2)
    out = torch.reshape(out, (B, -1, H, W))
    return out


def efficient_vit_block(
    weights: dict[str, torch.Tensor],
    location: str,
    heads_ratio: float = 1.0,
    dim: int = 32,
    norm: Norm = "batch",
    act: Act = "hardswish",
):
    def lite_mla(
        location: str,
        heads: int | None = None,
        heads_ratio: float = 1.0,
        norm: tuple[Norm, Norm] = (None, "batch"),
        act: tuple[Act, Act] = (None, None),
        kernel: Act = "relu",
        scales: tuple[int, ...] = (5,),
        eps=1.0e-15,
    ):
        qkv = conv_layer(
            weights=weights,
            location=f"{location}.qkv",
            norm=norm[0],
            act=act[0],
        )
        in_channels = weights[f"{location}.qkv.conv.weight"].shape[0] // 3
        heads = heads or int(in_channels // dim * heads_ratio)
        total_dim = heads * dim
        aggreg = [
            sequential(
                functools.partial(
                    F.conv2d,
                    weight=weights[f"{location}.aggreg.{i}.0.weight"],
                    bias=weights.get(f"{location}.aggreg.{i}.0.bias"),
                    padding=padding_from_kernel_size((scale, scale), 1),
                    groups=3 * total_dim,
                ),
                functools.partial(
                    F.conv2d,
                    weight=weights[f"{location}.aggreg.{i}.1.weight"],
                    bias=weights.get(f"{location}.aggreg.{i}.1.bias"),
                    groups=3 * heads,
                ),
            )
            for i, scale in enumerate(scales)
        ]
        kernel_fn = act_by_name(kernel)
        if kernel_fn:
            kernel_fn = functools.partial(kernel_fn, inplace=False)
        else:
            kernel_fn = identity
        proj = conv_layer(
            weights=weights,
            location=f"{location}.proj",
            norm=norm[1],
            act=act[1],
        )

        def call(x: torch.Tensor):
            _qkv = qkv(x)
            multi_scale_qkv = [_qkv]
            for op in aggreg:
                multi_scale_qkv.append(op(_qkv))
            multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

            out = relu_linear_att(
                multi_scale_qkv, dim=dim, eps=eps, kernel_func=kernel_fn
            )
            out = proj(out)

            return out

        return call

    context_module = residual_block(
        lite_mla(
            f"{location}.context_module.main",
            heads_ratio=heads_ratio,
            norm=(None, norm),
        ),
        identity,
    )
    local_module = residual_block(
        mbconv(
            weights,
            f"{location}.local_module.main",
            act=(act, act, None),
            norm=(None, None, norm),
            stride=1,
        ),
        identity,
    )
    return sequential(context_module, local_module)


def backbone(
    weights: dict[str, torch.Tensor],
    depths: tuple[int, ...],
    dim: int = 32,
    expand_ratio: int = 4,
    norm: Norm = "batch",
    act: Act = "hardswish",
):
    def local_block(
        location: str,
        stride: int,
        expand_ratio: float,
        fewer_norm: bool = False,
    ):
        if expand_ratio == 1:
            return dsconv(
                weights=weights,
                location=location,
                stride=(stride, stride),
                norm=(None, norm) if fewer_norm else (norm, norm),
                act=(act, None),
            )
        return mbconv(
            weights,
            location,
            act=(act, act, None),
            stride=stride,
            norm=(None, None, norm) if fewer_norm else (norm, norm, norm),
        )

    input_stem: Layer = sequential(
        *[
            conv_layer(
                weights, "backbone.input_stem.op_list.0", stride=2, norm=norm, act=act
            )
        ]
        + [
            residual_block(
                local_block(
                    location=f"backbone.input_stem.op_list.{i+1}.main",
                    stride=1,
                    expand_ratio=1,
                ),
                identity,
            )
            for i in range(depths[0])
        ]
    )

    stages: list[Layer] = []
    stage: list[Layer] = []
    for d in depths[1:3]:
        for i in range(d):
            stride = 2 if i == 0 else 1
            stage.append(
                residual_block(
                    local_block(
                        f"backbone.stages.{len(stages)}.op_list.{i}.main",
                        stride=stride,
                        expand_ratio=expand_ratio,
                    ),
                    identity if stride == 1 else None,
                )
            )
        stages.append(sequential(*stage))
        stage = []
    for d in depths[3:]:
        stage.append(
            residual_block(
                local_block(
                    f"backbone.stages.{len(stages)}.op_list.0.main",
                    stride=2,
                    expand_ratio=expand_ratio,
                    fewer_norm=True,
                ),
                None,
            )
        )

        for i in range(d):
            stage.append(
                efficient_vit_block(
                    weights=weights,
                    location=f"backbone.stages.{len(stages)}.op_list.{i+1}",
                    dim=dim,
                    norm=norm,
                    act=act,
                )
            )

        stages.append(sequential(*stage))
        stage = []

    def call(x: torch.Tensor):
        output: dict[str, torch.Tensor] = {"input": x}
        output["stage0"] = x = input_stem(x)
        for stage_id, stage in enumerate(stages, 1):
            output[f"stage{stage_id}"] = x = stage(x)
        output["stage_final"] = x
        return output

    return call


def _try_squeeze(x: torch.Tensor):
    return torch.flatten(x, start_dim=1) if x.dim() > 2 else x


def linear(
    weights: dict[str, torch.Tensor],
    location: str,
    norm: Norm = None,
    act: Act = None,
    norm_kwargs: dict[str, Any] | None = None,
):
    norm_kwargs = norm_kwargs or {}
    layers: list[Layer] = [
        _try_squeeze,
        functools.partial(
            F.linear,
            weight=weights[f"{location}.linear.weight"],
            bias=weights.get(f"{location}.linear.bias"),
        ),
    ]

    norm_layer = norm_by_name(norm, weights, f"{location}.norm", **norm_kwargs)
    if norm_layer:
        layers.append(norm_layer)
    act_fn = act_by_name(act)
    if act_fn:
        layers.append(act_fn)
    return sequential(*layers)


def cls_head(
    weights: dict[str, torch.Tensor],
    norm: Norm = "batch",
    act: Act = "hardswish",
    fid="stage_final",
):
    layers: Layer = sequential(
        *[
            conv_layer(weights, "head.op_list.0", norm=norm, act=act),
            functools.partial(F.adaptive_avg_pool2d, output_size=1),
            linear(
                weights=weights,
                location="head.op_list.2",
                norm="layer",
                act="hardswish",
                norm_kwargs={
                    "normalized_shape": (
                        weights["head.op_list.2.linear.weight"].shape[0],
                    )
                },
            ),
            linear(weights, "head.op_list.3", None, None),
        ]
    )

    def call(x_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        x = x_dict[fid]
        x = layers(x)
        return x

    return call


if __name__ == "__main__":
    img = img_transforms(Image.open("./test_data/image.jpg"))[None, ...]

    weights = torch.load(
        "assets/checkpoints/ImageNet-EfficientViT-B3-224x224.pt", map_location="cpu"
    )["state_dict"]
    b3 = sequential(
        backbone(weights, (1, 4, 6, 6, 9), 32),
        cls_head(weights),
    )(img)
    from torchvision.models._meta import _IMAGENET_CATEGORIES

    print(_IMAGENET_CATEGORIES[torch.argmax(b3[0]).item()])
