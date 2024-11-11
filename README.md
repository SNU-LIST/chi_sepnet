# $\chi$-sepnet (chi-sepnet)
* The code is for reconstructing susceptibility source-separated maps by deep neural network ($\chi$-sepnet, chi-sepnet).
* Matlab toolbox including conventional source separation method ($\chi$-separation) is also available (https://github.com/SNU-LIST/chi-separation.git).
* The source data for training can be shared to academic institutions. Request should be sent to snu.list.software@gmail.com. For each request, internal approval from our Institutional Review Board (IRB) is required (i.e. takes time).
* Don't hesitate to contact for usage and bugs: minjoony@snu.ac.kr
* Last update : Sep 23, 2024


## Usage
⭐ If you have both GRE and SE data, you have option for chi-sepnet-R2' (better quality).

⭐ If you only have GRE data, neural network (chi-sepnet-R2*) will deliver high quality susceptibility source-separated maps.

⭐ If you acquired data with different resolution from 1 x 1 x 1 mm<sup>3</sup>, the resolution generalization method (can process resolution > 0.6 mm; check the reference) is required.

⭐ If you acquired data with different B<sub>0</sub> direction from [0, 0, 1], the B<sub>0</sub> direction correction to [0, 0, 1] is required.

⭐ Input data with the same orientation with trained data (check the figure below) is recommended.
![xsepnet_data_order](https://github.com/user-attachments/assets/a99aeefd-8e01-4810-80e7-c02b6130d5db)

## Inference
You can follow the steps below for the inference.

1. Clone this repository
    ```bash
    git clone https://github.com/SNU-LIST/QSMnet.git
    ```
2. Create conda environment via downloaded yaml file
    ```bash
    conda env create -f chisepnet_env.yaml
    ```
3. Activate xsepnet conda environment
    ```bash
    conda activate xsepnet
    ```
5. Run the inference code
    ```bash
    python test.py
    ```
## References
>M. Kim, S. Ji, J. Kim, K. Min, H. Jeong, J. Youn, T. Kim, J. Jang, B. Bilgic, H. Shin, J. Lee, $\chi$-sepnet: Deep neural network for magnetic susceptibility source separation, arXiv prepring, 2024

>S. Ji, J. Park, H.-G. Shin, J. Youn, M. Kim and J. Lee, Successful generalization for data with higher or lower resolution than training data resolution in deep learning powered QSM reconstruction, ISMRM, 2023
