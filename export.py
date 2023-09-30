#!/usr/bin/env python3

from typing import Generator, Literal, TypeAlias
import typing
import numpy as np
import torch

ModelName: TypeAlias = Literal["b0", "b1", "b2", "b3"]
Task: TypeAlias = Literal["cls"]


@torch.inference_mode()
def write(
    output: str,
    torch_checkpoint: str,
    model_name: ModelName,
    resolution: tuple[int, int],
):
    weights: dict[str, torch.Tensor] = torch.load(torch_checkpoint, map_location="cpu")[
        "state_dict"
    ]

    header = f"name=EfficientViT.binary;version=0.0.1;model={model_name};width={resolution[0]};height={resolution[1]};num_weights={len(weights)}\n"
    with open(output + ".txt", "w") as fp:
        fp.write(header.replace(";", "\n") + "\n")
    with open(output, "wb") as fp:
        fp.write(header.encode())
        for key, value in weights.items():
            value_np: np.ndarray = value.numpy()
            weights_header = f"name={key.strip()};shape={str(value_np.shape)[1:-1].replace(' ','').rstrip(',') or 1};bytes={value_np.itemsize*value_np.size};type={value_np.dtype}\n"
            fp.write(weights_header.encode())
            value_np.tofile(fp, format="C")
            with open(output + ".txt", "a") as a_fp:
                a_fp.write(weights_header)


if __name__ == "__main__":
    import argparse
    import os
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument("torch_checkpoint")
    parser.add_argument(
        "--model_name", nargs="?", choices=typing.get_args(ModelName), type=str.lower
    )
    parser.add_argument("w", type=int, nargs="?", default=-1)
    parser.add_argument("h", type=int, nargs="?", default=-1)
    parser.add_argument("--output", "-O", required=False)

    params = vars(parser.parse_args())
    filename = os.path.basename(params["torch_checkpoint"])
    try:
        guesses = next(re.finditer(r"([BbLl]\d).(\d*).(\d*)", filename)).groups()
    except:
        guesses = None
    if params["w"] == -1:
        if guesses is None:
            raise ValueError(
                "Could not determine resolution from file name. Please supply argument."
            )
        params["w"] = int(guesses[1])
        params["h"] = int(guesses[2])
    elif params["h"] == -1:
        params["h"] = params["w"]
    if not params["output"]:
        params["output"] = (
            os.path.splitext(params["torch_checkpoint"])[0] + ".effvit.bin"
        )
    params["resolution"] = (params["w"], params["h"])
    if params["model_name"] is None:
        if guesses is None:
            raise ValueError(
                "Could not determine model from file name. Please supply argument `--model_name`."
            )
        params["model_name"] = guesses[0].lower()
    del params["h"]
    del params["w"]
    write(**params)
