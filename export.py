#!/usr/bin/env python

from typing import Literal,  TypeAlias
import typing
import torch
ModelName:TypeAlias = Literal["b0", "b1", "b2", "b3"]
def write(
    output: str,
    torch_checkpoint: str,
    model: ModelName,
    resolution: tuple[int, int],
):
    weights:dict[str,torch.Tensor] = torch.load(torch_checkpoint, map_location="cpu")['state_dict']
    with open(output, "wb") as fp:
        fp.write((f"name=EfficientViT.binary;version=0.0.1;model={model};width={resolution[0]};height={resolution[1]};num_weights={len(weights)}\n").encode())
        for key,value in weights.items():
            value = value.numpy()
            fp.write(f"name={key};shape={value.shape};type={value.dtype};bytes={value.itemsize*value.size}\n".encode())
            value.tofile(fp)
            fp.write(f"\n".encode())

if __name__ == "__main__":
    import argparse
    import os
    import re
    parser = argparse.ArgumentParser()
    parser.add_argument("torch_checkpoint")
    parser.add_argument("--model", nargs="?", choices=typing.get_args(ModelName),type=str.lower)
    parser.add_argument("w", type=int, nargs="?",default=-1)
    parser.add_argument("h", type=int,  nargs="?",default=-1)
    parser.add_argument("--output","-O",required=False)
    
    params = vars(parser.parse_args())
    filename = os.path.basename(params["torch_checkpoint"])
    try:
        guesses =next(re.finditer(r"([BbLl]\d).(\d*).(\d*)", filename)).groups()
    except:
        guesses = None
    if params["w"] == -1:
        if guesses is None:
            raise ValueError("Could not determine resolution from file name. Please supply argument.")
        params["w"] = int(guesses[1])
        params["h"] = int(guesses[2])
    elif params["h"] == -1:
        params["h"] = params["w"]
    if not params["output"]:
        params["output"] = os.path.splitext(params["torch_checkpoint"])[0] + ".effvit.bin"
    params["resolution"] = (params["w"], params["h"])
    if params["model"] is None:
        if guesses is None:
            raise ValueError("Could not determine model from file name. Please supply argument `--model`.")
        params["model"] = guesses[0].lower()
    del params["h"]
    del params["w"]
    write(**params)
