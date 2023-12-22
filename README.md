# Gait-Analysis
CNN-LSTM NetWork 

### Objective

This deep learning model is designed to investigate the impact of virtual experience and avatars on gait parameters in both real-world and virtual environments.


### Advantages 
1.  The CNN component of our network initially extracts spatial local features from the sequence of skeletal joint points.
2.  The LSTM component then processes these features, capturing spatiotemporal information.
3.  The Attention layer assigns weights to the vectors, identifying which time steps in the sequence are more significant for the analysis.

### NetWork
<img src="https://github.com/KevinZhou6/Gait-Analysis/blob/main/Gait-Analysis/network.png"  />

<br/>

### Accuracy & Loss
<img src ="https://github.com/KevinZhou6/Gait-Analysis/blob/main/Gait-Analysis/loss.png">
