# DIV2K Dataset & Patch Bank Strategy

This directory contains the split definitions and configuration for the DIV2K dataset used by SPA-VAEGAN and the Baseline CVAE.

## Official DIV2K Splits
The official DIV2K dataset consists of 1,000 high-resolution 2K images:
- **DIV2K_train_HR**: 800 images (`0001.png` - `0800.png`)
- **DIV2K_valid_HR**: 100 images (`0801.png` - `0900.png`)
- **Official Test Set**: Withheld by the challenge organizers.

### Project Split Partition (Fixed Seed = 42)
To provide an authentic, uncontaminated held-out benchmark:
1. **Training Set** (`dataset/splits/train.txt`): 720 images sampled from `DIV2K_train_HR`.
2. **Validation Set** (`dataset/splits/val.txt`): Remaining 80 images from `DIV2K_train_HR` (used strictly for model monitoring and checkpoint selection).
3. **Test Set** (`dataset/splits/test.txt`): All 100 official `DIV2K_valid_HR` images (used exclusively for the final quantitative benchmark).

## Patch Banks
To avoid the CPU bottleneck of decoding 2K PNG images on the fly during GPU training, patches are pre-extracted into raw memory-mapped `.npy` files:
- **Train Bank**: 30 random $288 \times 288$ patches per image ($720 \times 30 = 21,600$ patches). Augmentation is performed via random $256 \times 256$ sub-crops, random horizontal/vertical flips, and 90° rotations. Flat patches (pixel std < 0.05) are filtered out to maintain structural information.
- **Validation Bank**: 5 fixed center/quadrant $256 \times 256$ patches per image ($80 \times 5 = 400$ patches).
- **Test Bank**: 20 fixed deterministic $256 \times 256$ patches per image ($100 \times 20 = 2,000$ patches).

## Download Links
- Training images: `http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip` (~3.5 GB)
- Validation images: `http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip` (~0.4 GB)
