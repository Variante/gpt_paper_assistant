---
title: Arxiv Daily
---

# Personalized Daily Arxiv Papers 10/26/2026

<time id="gptpa-last-update" datetime="2026-10-26T14:00:03.785078+00:00">Last update: 2026-10-26T14:00:03.785078+00:00</time>

This project is adapted from [tatsu-lab/gpt_paper_assistant](https://github.com/tatsu-lab/gpt_paper_assistant). The source code of this project is at [Variante/gpt_paper_assistant](https://github.com/Variante/gpt_paper_assistant)

About me on [Bilibili](https://space.bilibili.com/823532). Help keep the website running:

<a href="https://www.buymeacoffee.com/Variante"><img src="https://img.buymeacoffee.com/button-api/?text=Help cover GPT cost&emoji=💓&slug=Variante&button_colour=60904a&font_colour=ffffff&font_family=Inter&outline_colour=000000&coffee_colour=FFDD00" /></a>


<a id="topics"></a>

## Topics

Paper selection prompt and criteria (jump to the section by clicking the link):

[1. Application of diffusion models and vision-language models (VLMs) to robot manipulation.
](#topic-1)

[2. New methodological improvements to self-supervised learning (SSL) for image or video representation.
](#topic-2)

[3. Video segmentation using unsupervised or self-supervised methods.
](#topic-3)

[4. Transfer learning across modalities (e.g., audio-to-video, optical flow, language-to-video) for improved video understanding.
](#topic-4)

[5. Advances in 3D generation using generative models, including image-to-3D and text-to-3D.
](#topic-5)

[6. Recent progress in 3D reconstruction and generation with Gaussian Splatting, NeRF, or mesh generation.
](#topic-6)

[7. Agentic frameworks and environments for robot learning.
](#topic-7)

[Go beyond](#go-beyond)


---
<a id="topic-1"></a>

### Topic 1
1000\. [If at First You Don't Succeed, Backtrack: Lightweight Recovery for Frozen VLAs](https://arxiv.org/abs/2610.32941) [[more](#1000-if-at-first-you-dont-succeed-backtrack-lightweight-recovery-for-frozen-vlas)]  
**Authors:** Thomas Landeg, Thierry Tambe

1002\. [Robotic Optical Coherence Tomography-Guided Geometric Modeling of Soft Tissue: 3D Representations for Data-Driven Closed-Loop Shape Control](https://arxiv.org/abs/2610.32961) [[more](#1002-robotic-optical-coherence-tomography-guided-geometric-modeling-of-soft-tissue-3d-representations-for-data-driven-closed-loop-shape-control)]  
**Authors:** Chaoyu Zhang, Guangshen Ma, Mark Draelos

1003\. [RynnUnify: Towards Unified Embodied Brain](https://arxiv.org/abs/2610.32965) [[more](#1003-rynnunify-towards-unified-embodied-brain)]  
**Authors:** Minghao Zhu, Bohan Hou, Kun Xiang, ..., Jianfei Yang, Xin Li

1005\. [UniAction: Unified Action Conditioning for World Models](https://arxiv.org/abs/2610.32977) [[more](#1005-uniaction-unified-action-conditioning-for-world-models)]  
**Authors:** Bowen Xue, Zizhang Li, Hong-Xing Yu, Jiajun Wu


Back to [[top](#topics)]

---
<a id="topic-2"></a>

### Topic 2
2004\. [Internal Discrepancy for Contrastive Learning: Generalization Bounds and Downstream Behaviors](https://arxiv.org/abs/2610.32729) [[more](#2004-internal-discrepancy-for-contrastive-learning-generalization-bounds-and-downstream-behaviors)]  
**Authors:** Jialiang Wang, Qing Ma, Xiong Zhou, ..., Haoliang Li, Xianming Liu


Back to [[top](#topics)]

---
<a id="topic-3"></a>

### Topic 3


Back to [[top](#topics)]

---
<a id="topic-4"></a>

### Topic 4


Back to [[top](#topics)]

---
<a id="topic-5"></a>

### Topic 5
5001\. [CityLift: Compositional 3D Urban Generation from a Single Image](https://arxiv.org/abs/2610.32728) [[more](#5001-citylift-compositional-3d-urban-generation-from-a-single-image)]  
**Authors:** Haoran Gong, Jie Feng, Chenxu Wang

5006\. [SceneBloom: Agentic Indoor Environment Generation with Realistic and Diverse Layouts for Embodied Learning](https://arxiv.org/abs/2610.32978) [[more](#5006-scenebloom-agentic-indoor-environment-generation-with-realistic-and-diverse-layouts-for-embodied-learning)]  
**Authors:** Yuhan Liu, Lianhui Qin, Shenji Wan


Back to [[top](#topics)]

---
<a id="topic-6"></a>

### Topic 6


Back to [[top](#topics)]

---
<a id="topic-7"></a>

### Topic 7
7007\. [RobustRLlib: A Unified Library and Benchmark for Robust Reinforcement Learning Algorithms](https://arxiv.org/abs/2610.32971) [[more](#7007-robustrllib-a-unified-library-and-benchmark-for-robust-reinforcement-learning-algorithms)]  
**Authors:** Xi Chen, Fengqi Hu, Zeyu Shen, ..., Anqi Liu, Laixi Shi


Back to [[top](#topics)]

---
<a id="go-beyond"></a>

### Go beyond


Back to [[top](#topics)]

---
## Full paper list
 <a id="1000-if-at-first-you-dont-succeed-backtrack-lightweight-recovery-for-frozen-vlas"></a>

### 1000\. [If at First You Don't Succeed, Backtrack: Lightweight Recovery for Frozen VLAs](https://arxiv.org/abs/2610.32941)
**ArXiv:** 2610.32941 [[page](https://arxiv.org/abs/2610.32941)] [[pdf](https://arxiv.org/pdf/2610.32941.pdf)]

**Authors:** Thomas Landeg, Thierry Tambe

**Abstract:** Vision-Language-Action (VLA) models perform well on robotic manipulation, yet remain brittle once their own actions carry them into states their training data barely covers. Recovering from such failures usually requires recovery demonstrations, policy retraining, or a separate model at inference time. We introduce Backtrack, a lightweight failure detection and recovery mechanism for frozen VLAs that needs none of these. A small failure prediction head, trained only on outcome-labeled rollouts of the frozen policy, reads the VLA's existing internal representations to estimate whether the current episode will fail. When the predicted failure probability exceeds a threshold, a controller pauses the policy, retraces the arm's recently achieved motion, and hands control back to the unchanged policy so it can retry from an earlier configuration. The head adds only 0.48-0.88% to the policy's parameter count. We evaluate Backtrack on LIBERO with three VLAs spanning two action-generation families: SmolVLA and $π$0.5, which use flow-matching action experts, and the autoregressive $π$0-FAST. On LIBERO-Long, the suite of long-horizon multi-step tasks and the one on which all three policies fail most often, Backtrack reduces each policy's failure rate by 19-48%, raising success by up to 7.4 percentage points; on the three shorter suites, where control success is already 85-99%, its effect is smaller. On MetaWorld, all twelve controller configurations tested with SmolVLA improve on the unassisted policy.

**Comment:** Criterion 1: Backtrack augments frozen robotic manipulation VLAs with a lightweight failure-prediction head and arm-motion retracing controller, reducing failure rates by 19–48% on LIBERO-Long without policy retraining.

**Relevance:** 10
Back to [[topic](#topic-1)] [[top](#topics)]

<a id="1002-robotic-optical-coherence-tomography-guided-geometric-modeling-of-soft-tissue-3d-representations-for-data-driven-closed-loop-shape-control"></a>

### 1002\. [Robotic Optical Coherence Tomography-Guided Geometric Modeling of Soft Tissue: 3D Representations for Data-Driven Closed-Loop Shape Control](https://arxiv.org/abs/2610.32961)
**ArXiv:** 2610.32961 [[page](https://arxiv.org/abs/2610.32961)] [[pdf](https://arxiv.org/pdf/2610.32961.pdf)]

**Authors:** Chaoyu Zhang, Guangshen Ma, Mark Draelos

**Abstract:** In medical robotics, data-driven soft-tissue shape control requires deforming tissue toward desired morphologies through sequential physical interactions. This challenge is particularly important in microscale manipulation, where localized deformation is difficult to observe, represent, and control, while physics-based models can be computationally expensive and difficult to adapt to realistic environments. Optical coherence tomography (OCT) provides high-resolution volumetric perception of tissue deformation, but target-conditioned control requires effective representations of both observed and desired tissue states. In this work, we introduce an OCT-guided data-driven framework for tissue shape control, learned from user-guided demonstrations and executed with closed-loop OCT feedback. We study full-volume, tissue-masked sparse-volume, and voxelized point-cloud representations, with or without residual conditioning, using deterministic (Explicit) and generative diffusion (Diffusion) policies. Across 12 configurations and 576 phantom rollouts on 3-DoF depth and 6-DoF directional tasks, OCT-derived 3D representations support both Explicit and Diffusion policies, with performance depending on representation and residual conditioning. Residual conditioning reduces directional Shape RMSE from 0.748 to 0.555 mm for full-volume Diffusion, while voxelized point-cloud Explicit with residuals achieves 0.534 mm for depth and 0.443 mm for directional control, with comparable accuracy at intermediate unseen targets and consistent performance across deployments. We further evaluate the framework in 60 ex vivo porcine rollouts, extending to biological tissue with final geometric errors comparable to human demonstrations. These results establish volumetric OCT as current and goal representations for learned closed-loop tissue shape control across deterministic and generative action models.

**Comment:** Matches criterion 1 by applying generative diffusion policies to OCT-guided, closed-loop robotic soft-tissue manipulation, evaluated in 576 phantom rollouts and 60 ex vivo porcine rollouts.

**Relevance:** 10
Back to [[topic](#topic-1)] [[top](#topics)]

<a id="1003-rynnunify-towards-unified-embodied-brain"></a>

### 1003\. [RynnUnify: Towards Unified Embodied Brain](https://arxiv.org/abs/2610.32965)
**ArXiv:** 2610.32965 [[page](https://arxiv.org/abs/2610.32965)] [[pdf](https://arxiv.org/pdf/2610.32965.pdf)]

**Authors:** Minghao Zhu, Bohan Hou, Kun Xiang, ..., Jianfei Yang, Xin Li

**Abstract:** We introduce RynnUnify, a 16B unified embodied model that combines strong embodied understanding with long-horizon visual planning through interleaved video--language reasoning. With capabilities spanning spatial reasoning, physical prediction, and grounding, RynnUnify connects an understanding of the physical world with visual anticipation of how it evolves across successive subtasks. Building on these capabilities, we further develop an agentic extension through reinforcement learning to selectively use video predictions for reasoning about unobserved states and interaction outcomes. We also develop a world-action extension of the full RynnUnify model, translating its understanding and prediction capabilities into robot control. To evaluate these capabilities, we introduce Rynn-VP and Rynn-GEU for long-horizon visual planning and anticipatory reasoning. Experiments show that RynnUnify achieves the highest average embodied understanding score among the evaluated unified multimodal models and improves the Rynn-VP overall score by 9.75 points over Cosmos 3 Nano. Its agentic extension improves the Rynn-GEU average by 6.0 points over direct reasoning, while the world-action extension achieves 51.5\% success on 20 selected composite RoboCasa365 tasks and 46.7\% average success across three real-robot tasks. Together, these results establish RynnUnify as a shared foundation for embodied understanding, visual planning, and robot control, taking a step toward a unified embodied brain.

**Comment:** Matches criteria 1 and 7 by combining interleaved video–language planning, RL-based use of video predictions, and a world-action robot-control extension, achieving 51.5% success on 20 composite RoboCasa365 tasks and 46.7% average success across three real-robot tasks.

**Relevance:** 10
Back to [[topic](#topic-1)] [[top](#topics)]

<a id="1005-uniaction-unified-action-conditioning-for-world-models"></a>

### 1005\. [UniAction: Unified Action Conditioning for World Models](https://arxiv.org/abs/2610.32977)
**ArXiv:** 2610.32977 [[page](https://arxiv.org/abs/2610.32977)] [[pdf](https://arxiv.org/pdf/2610.32977.pdf)]

**Authors:** Bowen Xue, Zizhang Li, Hong-Xing Yu, Jiajun Wu

**Abstract:** Actions that condition world models are highly heterogeneous, and thus existing action-conditioned world models are typically built for individual action types, fragmenting already limited action-annotated data across separate models. We introduce UniAction, a unified action-conditioning framework that represents any action as a sparse set of action elements, each with a discrete semantic identity and a continuous spatiotemporal grounding coordinate. By mapping these coordinates directly into the pretrained video model's native positional space, UniAction allows heterogeneous actions to condition a shared world model within the same spatiotemporal structure. We further introduce cardinality-calibrated attention to address the bias from differing action representation granularities. We evaluate UniAction on four representative action forms: force, camera control, hand pose, and robot pose. UniAction outperforms conditioning alternatives in action following, and a single jointly trained model surpasses the action-specific experts, showing that heterogeneous action data benefit from unified model and joint training.

**Comment:** Criterion 1: UniAction unifies action-conditioned world modeling across inputs including robot poses through sparse semantic–spatiotemporal action elements and cardinality-calibrated attention, with joint training outperforming action-specific experts.

**Relevance:** 9
Back to [[topic](#topic-1)] [[top](#topics)]

---
<a id="2004-internal-discrepancy-for-contrastive-learning-generalization-bounds-and-downstream-behaviors"></a>

### 2004\. [Internal Discrepancy for Contrastive Learning: Generalization Bounds and Downstream Behaviors](https://arxiv.org/abs/2610.32729)
**ArXiv:** 2610.32729 [[page](https://arxiv.org/abs/2610.32729)] [[pdf](https://arxiv.org/pdf/2610.32729.pdf)]

**Authors:** Jialiang Wang, Qing Ma, Xiong Zhou, ..., Haoliang Li, Xianming Liu

**Abstract:** Contrastive learning has emerged as a popular paradigm for training representation models. While prior studies pursue more practical and less restrictive assumptions to enhance theoretical understanding of contrastive learning, they essentially describe the interconnections among representations within the data or induced by the function class. To encapsulate these assumptions, we propose the measure of internal discrepancy that quantifies the difference between the mean representation of positive samples and latent cluster centers. We then prove an internal discrepancy-aware generalization bound for downstream supervised tasks following contrastive learning, without requiring additional assumptions. This generalization bound not only serves as a unifying framework connecting various assumptions but also motivates us to further reduce the internal discrepancy, thereby introducing the relaxed k-means loss as an approximation. Moreover, we investigate the downstream behaviors of contrastive learning from the perspective of contrastive learning-incurred supervised losses. In the presence of class imbalance, we shed light on why contrastive learning can facilitate imbalanced learning. Finally, we investigate the optimality of these incurred supervised losses, proving several intriguing properties.

**Comment:** Criterion 2: Rigorously analyzes contrastive representation learning through an internal-discrepancy-aware downstream generalization bound and introduces a relaxed k-means loss to reduce that discrepancy.

**Relevance:** 9
Back to [[topic](#topic-2)] [[top](#topics)]

---
<a id="5001-citylift-compositional-3d-urban-generation-from-a-single-image"></a>

### 5001\. [CityLift: Compositional 3D Urban Generation from a Single Image](https://arxiv.org/abs/2610.32728)
**ArXiv:** 2610.32728 [[page](https://arxiv.org/abs/2610.32728)] [[pdf](https://arxiv.org/pdf/2610.32728.pdf)]

**Authors:** Haoran Gong, Jie Feng, Chenxu Wang

**Abstract:** Generating editable 3D urban scenes from a single image requires detailed building assets with a coherent spatial arrangement. Occlusion and perspective ambiguity make it difficult to infer complete building geometry together with consistent relative scales and positions. We present CityLift, a framework that generates individual textured building meshes from an urban image by coupling scene-level spatial modeling with building-level synthesis. CityLift first learns building shape and placement jointly through coupled scene and instance streams, producing coarse sparse structures in a common coordinate system. We then introduce generative spatial inheritance, which maps each building's coarse occupancy and scene-conditioned 3D feature field into a local volume to guide a flow model that generates finer structure. The same coordinate mapping places the generated meshes back into the scene, preserving their spatial relationship to the scaffold. To address the scarcity of urban data pairing scene images with individual 3D buildings, we construct CityLift-5K, comprising 4,997 synthetic scenes with instance masks, building meshes, and placement annotations. Experiments on CityLift-5K and scene blocks from UrbanScene3D and InstanceBuilding demonstrate improved building geometry and scene composition. The project page is available at https://ranhaogong.github.io/CityLift/.

**Comment:** Criterion 5: CityLift uses generative spatial inheritance to guide a flow model for single-image textured building-mesh generation, demonstrating improved geometry and scene composition on CityLift-5K, UrbanScene3D, and InstanceBuilding.

**Relevance:** 10
Back to [[topic](#topic-5)] [[top](#topics)]

<a id="5006-scenebloom-agentic-indoor-environment-generation-with-realistic-and-diverse-layouts-for-embodied-learning"></a>

### 5006\. [SceneBloom: Agentic Indoor Environment Generation with Realistic and Diverse Layouts for Embodied Learning](https://arxiv.org/abs/2610.32978)
**ArXiv:** 2610.32978 [[page](https://arxiv.org/abs/2610.32978)] [[pdf](https://arxiv.org/pdf/2610.32978.pdf)]

**Authors:** Yuhan Liu, Lianhui Qin, Shenji Wan

**Abstract:** Training generalizable embodied agents requires realistic, varied, and simulation-ready indoor environments. Language-guided methods support semantic planning and object-level reasoning, but precise spatial grounding remains difficult. Visual generative models offer rich priors over plausible arrangements, yet their outputs require geometric reasoning to become physically valid 3D scenes. We introduce SceneBloom, an agentic framework that combines visual layout sampling with contract-guided 3D compilation. A hierarchical sampler uses diffusion priors to generate furniture arrangements within functional zones. A compiler then uses relations extracted from each proposal to guide geometric fitting and local pose correction. Across five room types, SceneBloom attains the highest visual quality and physical stability among the compared methods and is the only method with no detected mesh collisions and out-of-bounds objects. Its layouts are also closer to the distribution of real rooms than those of all compared baselines. Object navigation policies trained in these realistic scenes achieve a higher success rate in held-out real scans than those trained on an equal number of real training scans. Code is available at https://github.com/Tinaliu0123/scenebloom.

**Comment:** Criterion 5: SceneBloom combines diffusion-based layout sampling with contract-guided 3D compilation to generate indoor scenes, achieving the highest reported visual quality and physical stability among compared methods with no detected collisions or out-of-bounds objects.

**Relevance:** 8
Back to [[topic](#topic-5)] [[top](#topics)]

---
<a id="7007-robustrllib-a-unified-library-and-benchmark-for-robust-reinforcement-learning-algorithms"></a>

### 7007\. [RobustRLlib: A Unified Library and Benchmark for Robust Reinforcement Learning Algorithms](https://arxiv.org/abs/2610.32971)
**ArXiv:** 2610.32971 [[page](https://arxiv.org/abs/2610.32971)] [[pdf](https://arxiv.org/pdf/2610.32971.pdf)]

**Authors:** Xi Chen, Fengqi Hu, Zeyu Shen, ..., Anqi Liu, Laixi Shi

**Abstract:** Robust reinforcement learning (RL) is critical for reliable real-world deployment, where policies may fail under diverse shifts and disruptions. Existing robust RL algorithms offer a rich set of mechanisms and principles, yet two major barriers limit their broader potential: limited algorithm reusability and an unclear empirical capability boundary. We introduce RobustRLlib, an algorithm-centric library and benchmark that addresses both challenges. It integrates 17 representative robust online, offline, and safe RL algorithms under a unified interface for substantially improved reusability, together with an enhanced evaluation platform spanning diverse real-world shifts and advanced humanoid and vision-language-action tasks. Systematic evaluation over 273 perturbation configurations provides an up-to-date capability map of current robust RL algorithms, revealing where they remain effective and where substantial robustness gaps persist. As an open-source, user-friendly platform, RobustRLlib offers a reproducible and extensible codebase for applying existing methods to new problems and developing new robust RL algorithms.

**Comment:** Criterion 7: RobustRLlib provides a unified learning and evaluation platform with humanoid and vision-language-action tasks, benchmarking 17 robust RL algorithms across 273 perturbation configurations.

**Relevance:** 7
Back to [[topic](#topic-7)] [[top](#topics)]
