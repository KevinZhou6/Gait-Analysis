# Gait-Analysis
This repository is the official implementation of our IEEE SMC 24 paper: [Study on the Influence of Embodied Avatars on Gait Parameters in Virtual Environments and Real World](https://ieeexplore.ieee.org/abstract/document/10831053)

<h5 align="center">

[![arxiv](https://img.shields.io/badge/Arxiv-2411.18949-red)](https://arxiv.org/abs/2411.18949) &ensp;
![GitHub Repo stars](https://img.shields.io/github/stars/KevinZhou6/Gait-Analysis)
</h5>

## Avatars
<img src="https://github.com/KevinZhou6/Gait-Analysis/blob/main/assets/avatars.png"  />

## 📣 News
- May. 13, 2024. Code release.
- Apr. 25, 2024. Accepted by IEEE SMC 2024 for Oral Presentation.
- Dec. 29, 2023. Project release.

### Objective
This deep learning model is designed to investigate the impact of virtual experience and avatars on gait parameters in both real-world and virtual environments.



### Advantages 
1.  The CNN component of our network initially extracts spatial local features from the sequence of skeletal joint points.
2.  The LSTM component then processes these features, capturing spatiotemporal information.
3.  The Attention layer assigns weights to the vectors, identifying which time steps in the sequence are more significant for the analysis.
<img src="https://github.com/KevinZhou6/Gait-Analysis/blob/main/assets/model.png"  />



## BibTeX
```
@inproceedings{zhou2024study,
  title={Study on the Influence of Embodied Avatars on Gait Parameters in Virtual Environments and Real World},
  author={Zhou, Tianyi and Ding, Ding and Wang, Shengyu and Shi, Chuhan and Xu, Xiangyu},
  booktitle={2024 IEEE International Conference on Systems, Man, and Cybernetics (SMC)},
  pages={481--487},
  year={2024},
  organization={IEEE}
}
```

### End
#### If you appreciate the model, please give it a star.
####  For inquiries regarding the gait dataset, feel free to contact me via email.

