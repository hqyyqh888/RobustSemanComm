# Robust Semantic Communication
This repository contains the demo code for our TWC work "Robust semantic communications with masked VQ-VAE enabled codebook", available at: https://ieeexplore.ieee.org/document/10101778 and has been accepted for publication in TWC.

For any reproduce, further research or development, please kindly cite our TWC Journal paper:

`Q. Hu, G. Zhang, Z. Qin, Y. Cai, G. Yu and G. Y. Li“Robust semantic communications with masked VQ-VAE enabled codebook,” IEEE Transactions on Wireless Communications, early access, DOI: ,10.1109/TWC.2023.3265201, 2023.`

# Requirements
The following versions have been tested: Python 3.9 + Pytorch 1.8.0. But newer versions should also be fine.

# Introductions
The running instructions are available at the script 'running_command.sh', please kindly choose the required one and paste it to 'execute.sh'. Then, you can use 'bash execute.sh' to excute the script.

Moreover, the results raised in the paper require initializing the model with official pretrained weights or the weights trained with mask strategy, you can modify the settings of the model at the end of 'model.py' to train the whole model.

We also provide the code for MIMO communication, just modify the transmission procedure in the class 'VectorQuantizer'.
