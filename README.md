# EfficientViT.mojo

Incomplete conversion of EfficientViT into mojo. (WIP - mostly just to get my head around the language).

## Usage

Download checkpoint:

```shell
make checkpoint dataset=ImageNet resolution=224x224 model=EfficientViT-B3
```

Checkpoints will get downloaded to `./assets/checkpoints`.

## Roadmap

- [x] Export torch weights into binary
- [x] Read exported weights
- [x] Design script that runs inference with just weights
- [x] Enable image loading
- [ ] Convert script into functions that call torch
- [ ] Convert torch functions to native mojo

## Credits

Original EfficientViT can be found at: [https://github.com/mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit)
