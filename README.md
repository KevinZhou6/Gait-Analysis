# Gait-Analysis
This repository is the official implementation of our IEEE SMC 24 paper: [Study on the Influence of Embodied Avatars on Gait Parameters in Virtual Environments and Real World](https://ieeexplore.ieee.org/abstract/document/10831053)

### Objective

This deep learning model is designed to investigate the impact of virtual experience and avatars on gait parameters in both real-world and virtual environments.


### Background
1. The skeletal joint information captured by the Kinect camera has a certain periodicity. This type of information can be considered as time-series data and can be analyzed using neural networks.
<img src ="https://github.com/KevinZhou6/Gait-Analysis/blob/main/Gait-Analysis/Navel.png">


### Advantages 
1.  The CNN component of our network initially extracts spatial local features from the sequence of skeletal joint points.
2.  The LSTM component then processes these features, capturing spatiotemporal information.
3.  The Attention layer assigns weights to the vectors, identifying which time steps in the sequence are more significant for the analysis.

### NetWork
<img src="https://github.com/KevinZhou6/Gait-Analysis/blob/main/Gait-Analysis/network.png"  />

<br/>

### Accuracy & Loss
<img src ="https://github.com/KevinZhou6/Gait-Analysis/blob/main/Gait-Analysis/loss.png">
<br/>

## BibTeX
```
@INPROCEEDINGS{10831053,
  author={Zhou, Tianyi and Ding, Ding and Wang, Shengyu and Shi, Chuhan and Xu, Xiangyu},
  booktitle={2024 IEEE International Conference on Systems, Man, and Cybernetics (SMC)}, 
  title={Study on the Influence of Embodied Avatars on Gait Parameters in Virtual Environments and Real World}, 
  year={2024},
  volume={},
  number={},
  pages={481-487},
  keywords={Avatars;Neural networks;Virtual environments;Cybernetics;Virtual Reality;Gait Analysis;Virtual Avatars;Embodiment},
  doi={10.1109/SMC54092.2024.10831053}}
```

### End
#### If you appreciate the model, please give it a star.
####  For inquiries regarding the gait dataset, feel free to contact me via email.

