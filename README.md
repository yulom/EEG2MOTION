# EEG2MOTION: Towards Open-Vocabulary Human Motion Synthesis from Non-invasive Brain Signals

Official implementation of **EEG2MOTION: Towards Open-Vocabulary Human Motion Synthesis from Non-invasive Brain Signals**. https://arxiv.org/abs/2608.14754

Please visit our Demo Page to see results: https://yulom.github.io/EEG2MOTIONdemopage/

## 1. Dataset Preparation

Please download the **EEG2MOTION dataset** from Zenodo:

https://zenodo.org/records/22741447

After downloading, extract the dataset into the `EEG_data` folder.

The directory structure should be organized as follows:

```text
EEG2MOTION/
├── EEG_data/
│   ├── 20260416S1/
│   ├── ...
│   ├── 20260515S9/
│   ├── HumanML3D_train_processed.pt
│   ├── HumanML3D_val_processed.pt
│   ├── HumanML3D_test_processed.pt
│   └── videomae_global_features/
│
├── EMMM.ipynb
├── ContrastTask.ipynb
└── ...
```

The nine subject folders (`S1`–`S9`) contain the EEG recordings and corresponding experimental information.

For the final experimental results reported in the paper, we used the following six subjects:

```text
S1, S2, S4, S6, S7, S9
```

The other subjects are retained in the released dataset for completeness and potential future research.

Please refer to the dataset description on Zenodo for more details about the data organization and file formats.

## 2. MMM Preparation

Our motion synthesis framework is primarily built upon **MMM (Generative Masked Motion Model)** [1]. Therefore, reproducing the motion synthesis experiments requires downloading and preparing the pretrained models and related resources provided by MMM.

Please follow the instructions in the official MMM repository:

https://github.com/exitudio/MMM

In particular, run the following commands from the MMM repository:

```bash
bash dataset/prepare/download_glove.sh
bash dataset/prepare/download_extractor.sh
bash dataset/prepare/download_model.sh
```

These commands download the GloVe embeddings, motion feature extractor, and pretrained MMM model required by EEG2MOTION.

## 3. Environment Setup

We provide the required Python environment in `environment.yml`.

The recommended way to create the environment is:

```bash
conda env create -f environment.yml
conda activate EMMM
```

If you encounter problems when creating the environment from `environment.yml`, you can manually install the dependencies:

```bash
conda create --name EMMM
conda activate EMMM

conda install plotly tensorboard scipy matplotlib pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

pip install git+https://github.com/openai/CLIP.git einops gdown

pip install --upgrade nbformat

pip install accelerate transformers peft smplx trimesh h5py tqdm pyyaml notebook
```

The experiments were developed and tested with the above environment configuration.

## 4. Reproduction

The two main programs provided in this repository are:

* [`EMMM.ipynb`](https://github.com/yulom/EEG2MOTION/blob/main/EMMM.ipynb)
* [`ContrastTask.ipynb`](https://github.com/yulom/EEG2MOTION/blob/main/ContrastTask.ipynb)

They correspond to the two main experimental tasks described in the paper.

### 4.1 EEG-based Human Motion Synthesis

`EMMM.ipynb` implements the complete human motion synthesis pipeline.

The pipeline takes EEG signals as input and generates the corresponding human motion. It includes EEG preprocessing, EEG representation learning, EEG-to-motion conditioning, and motion generation based on the pretrained MMM model.

Please make sure that:

1. The EEG2MOTION dataset has been downloaded and placed under `EEG_data/`.
2. The required MMM resources and pretrained models have been downloaded.
3. The Python environment has been correctly configured.

Then open and run:

```text
EMMM.ipynb
```

The notebook contains the complete implementation used for the human motion synthesis experiments.

### 4.2 Cross-modal Contrastive Learning

`ContrastTask.ipynb` implements the cross-modal contrastive learning experiments described in the paper.

The notebook provides the implementation for learning representations between EEG signals and different modalities, including the corresponding motion, text, and video representations.

Run:

```text
ContrastTask.ipynb
```

to reproduce the contrastive learning experiments.

## 5. Experimental Results

The results reported in the paper are provided in the `eeg_logger/` directory.

The logger files contain the experimental results generated during our experiments and can be used to inspect the reported performance and compare reproduced results with those presented in the paper.

```text
eeg_logger/
├── ...
```

## 6. Citation

If you find this work useful, please cite:

```bibtex
@article{peng2026eeg2motion,
  title={EEG2MOTION: Towards Open-Vocabulary Human Motion Synthesis from Non-invasive Brain Signals},
  author={Peng, Yulong and Pan, Yijian and Yang, Yuqi and Zheng, Nenggan and Chen, Weidong and Hu, Xiaoling and Zhang, Shaomin},
  journal={arXiv preprint arXiv:2608.14754},
  year={2026}
}
```

## 7. Acknowledgements

This work builds upon the **MMM: Generative Masked Motion Model**. We thank the authors of MMM for making their code and pretrained models publicly available.

Please also refer to the original MMM repository for details:

https://github.com/exitudio/MMM

```bibtex
@inproceedings{pinyoanuntapong2024mmm,
  title={MMM: Generative Masked Motion Model},
  author={Ekkasit Pinyoanuntapong and Pu Wang and Minwoo Lee and Chen Chen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024},
}
```

The human motion sequences included in the EEG2MOTION dataset were sampled from the **HumanML3D dataset**. If you use the motion data, please also cite the original HumanML3D paper:

```bibtex
@InProceedings{Guo_2022_CVPR,
    author    = {Guo, Chuan and Zou, Shihao and Zuo, Xinxin and Wang, Sen and Ji, Wei and Li, Xingyu and Cheng, Li},
    title     = {Generating Diverse and Natural 3D Human Motions From Text},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {5152-5161}
}
```

