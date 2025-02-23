# Breast Cancer Tumor Segmentation

Breast Cancer Segmentation done with Pytorch.

The model produces a Binary Cross Entropy Loss of 0.163. The model also produces a dice score of 0.88 This produces over an 80% confidence in each pixel's classification.
The activation function can be modified to increase the dice score and by default it is set to 0.45. 


Created for BioHacks 2025 at UCSC Feb 22 - 23


# Setup
## Conda Environment
The code is dependent on a conda environment using with python version 3.10.16
## Dependencies:
- Pytorch v=2.5.1
- torchvision v=0.18.1
- Albumentations v=1.4.20
- Pillow v=11.1.0
- Matplotlib v=3.10.0
- tqdm v=4.67.1
- Pandas v= 2.2.3
= Openpyxl v=3.1.5

## Tested On
- MacOS ARM
- Arch AMD
- Windows Intel


Dataset used:
Pawłowska, A., Ćwierz-Pieńkowska, A., Domalik, A., Jaguś, D., Kasprzak, P., Matkowski, R., Fura, Ł., Nowicki, A., & Zolek, N. A Curated benchmark dataset for ultrasound based breast lesion analysis. Sci Data 11, 148 (2024). https://doi.org/10.1038/s41597-024-02984-z

Accessed Feb 22 