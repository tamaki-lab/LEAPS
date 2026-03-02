# Student LEAPS architectures

## pth/leaps/leaps_k400_128_4h_6l_v1L.pth

Model architecture:

- ViT base model
- 128 hidden dimensions
- 512 feedforward dimensions
- 4 attention heads
- 6 layers
- 3D patch shape is (2, 16, 16)

Distillation settings:

- Kinetics-400 dataset
- 50 epochs for distillation
- teacher Model: [MCG-NJU/VideoMAE-Large](https://huggingface.co/MCG-NJU/videomae-large)
