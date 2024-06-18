# Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving?

### [arXiv](http://arxiv.org/abs/2312.03031) | [知乎](https://zhuanlan.zhihu.com/p/669454065)

https://github.com/NVlabs/BEV-Planner/assets/27915819/93afa127-813f-4d36-b4f2-84f6b8d9b905

## INTRODUCTION
End-to-end autonomous driving recently emerged as a promising research direction to target autonomy from a full-stack perspective. Along this line, many of the latest works follow an open-loop evaluation setting on nuScenes to study the planning behavior. In this paper, we delve deeper into the problem by conducting thorough analyses and demystifying more devils in the details. We initially observed that the nuScenes dataset, characterized by relatively simple driving scenarios, leads to an under-utilization of perception information in end-to-end models incorporating ego status, such as the ego vehicle's velocity. These models tend to rely predominantly on the ego vehicle's status for future path planning. 
Beyond the limitations of the dataset, we also note that current metrics do not comprehensively assess the planning quality, leading to potentially biased conclusions drawn from existing benchmarks. To address this issue, we introduce a new metric to evaluate whether the predicted trajectories adhere to the road. 
We further propose a simple baseline able to achieve competitive results without relying on perception annotations.
Given the current limitations on the benchmark and metrics, we suggest the community reassess relevant prevailing research and be cautious whether the continued pursuit of state-of-the-art would yield convincing and universal conclusions.


## Start
### 1.Setting up Environment
### 2.Preparing Dataset
### 3.Training

### 4.Eval