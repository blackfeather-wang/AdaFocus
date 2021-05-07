# AdaFocus

## Introduction

In this paper, we explore the spatial redundancy in video recognition with the aim to improve the computational efficiency. It is observed that the most informative region in each frame of a video is usually a small image patch, which shifts smoothly across frames. Therefore, we model the patch localization problem as a sequential decision task, and propose a reinforcement learning based approach for efficient spatially adaptive video recognition (AdaFocus). In specific, a light-weighted ConvNet is first adopted to quickly process the full video sequence, whose features are used by a recurrent policy network to localize the most task-relevant regions. Then the selected patches are inferred by a high-capacity network for the final prediction. During offline inference, once the informative patch sequence has been generated, the bulk of computation can be done in parallel, and is efficient on modern GPU devices. In addition, we demonstrate that the proposed method can be easily extended by further considering the temporal redundancy, e.g., dynamically skipping less valuable frames. Extensive experiments on five benchmark datasets, i.e., ActivityNet, FCVID, Mini-Kinetics, Something-Something V1\&V2, demonstrate that our method is significantly more efficient than the competitive baselines.


<p align="center">
    <img src="./figure/intro.png" width= "600">
</p>


## Result

- ActivityNet

<p align="center">
    <img src="./figure/actnet.png" width= "700">
</p>


- Something-Something V1&V2

<p align="center">
    <img src="./figure/sthsth.png" width= "900">
</p>


- Visualization

<p align="center">
    <img src="./figure/Visualization.png" width= "900">
</p>
