---
title: Arxiv Daily
---

# Personalized Daily Arxiv Papers 10/19/2026

<time id="gptpa-last-update" datetime="2026-10-19T14:00:02.321302+00:00">Last update: 2026-10-19T14:00:02.321302+00:00</time>

This project is adapted from [tatsu-lab/gpt_paper_assistant](https://github.com/tatsu-lab/gpt_paper_assistant). The source code of this project is at [Variante/gpt_paper_assistant](https://github.com/Variante/gpt_paper_assistant)

About me on [Bilibili](https://space.bilibili.com/823532). Help keep the website running:

<a href="https://www.buymeacoffee.com/Variante"><img src="https://img.buymeacoffee.com/button-api/?text=Help cover GPT cost&emoji=🍍&slug=Variante&button_colour=5a448f&font_colour=ffffff&font_family=Poppins&outline_colour=000000&coffee_colour=FFDD00" /></a>


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
1002\. [Enhancing Visual Task-Redirection Attacks on Vision-Language-Action Models via Closed-Loop Optimization](https://arxiv.org/abs/2610.23566) [[more](#1002-enhancing-visual-task-redirection-attacks-on-vision-language-action-models-via-closed-loop-optimization)]  
**Authors:** Chunjin Ye, Guanying Chen, Jianjie Huang, ..., Tieyun Qian, Xiaochun Cao


Back to [[top](#topics)]

---
<a id="topic-2"></a>

### Topic 2
2000\. [Learning to Discriminate Scene Structures Makes Self-Supervised Depth Learning Scalable](https://arxiv.org/abs/2610.23600) [[more](#2000-learning-to-discriminate-scene-structures-makes-self-supervised-depth-learning-scalable)]  
**Authors:** Mengtan Zhang, Yuchuan Cui, Yu Ma, ..., Wei Ye, Rui Fan


Back to [[top](#topics)]

---
<a id="topic-3"></a>

### Topic 3
3005\. [FLoTS-SLAM: Optical Flow-Derived Uncertainty for Monocular Dynamic SLAM with Two-Stage Optimization](https://arxiv.org/abs/2610.23496) [[more](#3005-flots-slam-optical-flow-derived-uncertainty-for-monocular-dynamic-slam-with-two-stage-optimization)]  
**Authors:** Haorui Du, Lipu Zhou


Back to [[top](#topics)]

---
<a id="topic-4"></a>

### Topic 4


Back to [[top](#topics)]

---
<a id="topic-5"></a>

### Topic 5
5001\. [Video Diffusion Avatars: Taming Video Diffusion Models into Controllable 3D Avatars](https://arxiv.org/abs/2610.23772) [[more](#5001-video-diffusion-avatars-taming-video-diffusion-models-into-controllable-3d-avatars)]  
**Authors:** Simon Giebenhain, Tobias Kirschstein, Liam Schoneveld, ..., Zhe Chen, Matthias Nießner

5004\. [Persistent Planar Memory for Video World Models](https://arxiv.org/abs/2610.23795) [[more](#5004-persistent-planar-memory-for-video-world-models)]  
**Authors:** Yuze He, Bin Tan, Zelin Gao, ..., Nan Xue, Yinghao Xu


Back to [[top](#topics)]

---
<a id="topic-6"></a>

### Topic 6


Back to [[top](#topics)]

---
<a id="topic-7"></a>

### Topic 7
7003\. [Learning Robot Metareasoning: Adaptive Computation Budgets for Real-Time Control](https://arxiv.org/abs/2610.23662) [[more](#7003-learning-robot-metareasoning-adaptive-computation-budgets-for-real-time-control)]  
**Authors:** Lorenzo Magnino, Antonio Marino, Michael Amir, Daigo Shishika, Amanda Prorok


Back to [[top](#topics)]

---
<a id="go-beyond"></a>

### Go beyond


Back to [[top](#topics)]

---
## Full paper list
 <a id="1002-enhancing-visual-task-redirection-attacks-on-vision-language-action-models-via-closed-loop-optimization"></a>

### 1002\. [Enhancing Visual Task-Redirection Attacks on Vision-Language-Action Models via Closed-Loop Optimization](https://arxiv.org/abs/2610.23566)
**ArXiv:** 2610.23566 [[page](https://arxiv.org/abs/2610.23566)] [[pdf](https://arxiv.org/pdf/2610.23566.pdf)]

**Authors:** Chunjin Ye, Guanying Chen, Jianjie Huang, ..., Tieyun Qian, Xiaochun Cao

**Abstract:** Vision-language-action (VLA) models enable language-guided robot manipulation, but remain vulnerable to visual attacks that undermine user control over robot behavior. Prior studies optimize visual perturbations using fixed target supervision. However, the perturbation can change earlier actions and, in turn, the physical states reached later. As the attacked trajectory evolves, this supervision may no longer match the behavior needed at the states actually reached. Accurately matching these targets can therefore fail to advance the attacker-specified task. Even with suitable supervision, reducing average matching error does not ensure sustained task progress. The robot may still fail at a critical stage, and further updates can disrupt previously successful behavior. To address these limitations, we propose TRACO, a framework that enhances existing targeted visual attacks through closed-loop optimization. Its Execution-Driven Supervision rebuilds supervision for the states reached under attack to guide subsequent updates. Within this framework, Progress-Guided Selection focuses updates where progress stalls while maintaining supervision on the earlier behavior needed to reach that point. Experiments across VLA models and simulation environments show improved task-redirection success with TRACO, highlighting the security implications of closed-loop attack optimization.

**Comment:** Criterion 1: TRACO studies VLA-based robot manipulation through closed-loop visual attacks, combining Execution-Driven Supervision and Progress-Guided Selection to improve task-redirection success across VLA models and simulation environments.

**Relevance:** 8
Back to [[topic](#topic-1)] [[top](#topics)]

---
<a id="2000-learning-to-discriminate-scene-structures-makes-self-supervised-depth-learning-scalable"></a>

### 2000\. [Learning to Discriminate Scene Structures Makes Self-Supervised Depth Learning Scalable](https://arxiv.org/abs/2610.23600)
**ArXiv:** 2610.23600 [[page](https://arxiv.org/abs/2610.23600)] [[pdf](https://arxiv.org/pdf/2610.23600.pdf)]

**Authors:** Mengtan Zhang, Yuchuan Cui, Yu Ma, ..., Wei Ye, Rui Fan

**Abstract:** Self-supervised monocular depth estimation is appealing for its potential to scale with unlabeled data, yet existing methods often degrade when trained on diverse scenes. This study finds that, under self-supervision, models tend to encode dataset-specific structural biases rather than transferable scene geometry, leading to cross-domain optimization conflicts. To address this issue, this study proposes the MaSS framework, aiming to make self-supervised depth learning scalable to diverse data. MaSS introduces a dedicated structural representation pathway explicitly decoupled from contextual features, which is guided by a novel scene-structure discrimination objective to learn generalized, layout-aware structural representations for depth decoding. Trained and evaluated on four diverse datasets, both individually and jointly, MaSS substantially outperforms prior self-supervised methods and is the first to consistently benefit from mixed-domain training. Zero-shot evaluation on eight additional datasets further demonstrates its remarkable generalizability, particularly on those entirely out-of-distribution scenes. The source code is publicly available at https://mias.group/MaSS.

**Comment:** Criterion 2: MaSS introduces a novel scene-structure discrimination objective and a structural representation pathway decoupled from contextual features, improving self-supervised depth representations with consistent mixed-domain training gains and zero-shot generalization evaluated on eight additional datasets.

**Relevance:** 9
Back to [[topic](#topic-2)] [[top](#topics)]

---
<a id="3005-flots-slam-optical-flow-derived-uncertainty-for-monocular-dynamic-slam-with-two-stage-optimization"></a>

### 3005\. [FLoTS-SLAM: Optical Flow-Derived Uncertainty for Monocular Dynamic SLAM with Two-Stage Optimization](https://arxiv.org/abs/2610.23496)
**ArXiv:** 2610.23496 [[page](https://arxiv.org/abs/2610.23496)] [[pdf](https://arxiv.org/pdf/2610.23496.pdf)]

**Authors:** Haorui Du, Lipu Zhou

**Abstract:** Monocular simultaneous localization and mapping (SLAM) remains challenging in dynamic environments. Real-time dynamic object identification in such environments is a critical problem. Recent approaches estimate per-pixel uncertainty from rendering residuals or multiview feature inconsistency to identify dynamic regions. These methods lack direct motion evidence, and their accuracy may degrade when reconstruction is inaccurate or features are ambiguous. In addition, such uncertainty merely downweights pixels in bundle adjustment (BA), so inaccurate uncertainty estimates can still impair camera pose estimation and 3D reconstruction. Accordingly, we present a monocular RGB SLAM system for dynamic scenes. Specifically, a motion uncertainty estimation method is proposed to derive uncertainty from the residual between optical flow and camera motion induced flow, directly reflecting object motion. The estimation varies with frame alignment accuracy, reducing false detections and yielding accurate pixel-level masks of dynamic regions. To make BA more robust, we estimate camera poses from high confidence pixels. These poses are then used to reproject dynamic pixels during joint refinement of poses and depth, reducing the influence of object motion on both pose and depth updates. Moreover, we provide a quantitative evaluation of dynamic masks and analyze how these masks affect the downstream tracking and reconstruction of dynamic objects. Our approach improves the mean dynamic mask IoU by 35\% over a state-of-the-art baseline on DAVIS, and reduces the average ATE by 59\% on Bonn and 19\% on TUM relative to the SLAM backbone used in our system. The source code will be made publicly available.

**Comment:** Matches criterion 3 by deriving unsupervised dynamic-region masks from residuals between optical flow and camera-induced flow, improving mean dynamic-mask IoU by 35% over a state-of-the-art baseline on DAVIS.

**Relevance:** 7
Back to [[topic](#topic-3)] [[top](#topics)]

---
<a id="5001-video-diffusion-avatars-taming-video-diffusion-models-into-controllable-3d-avatars"></a>

### 5001\. [Video Diffusion Avatars: Taming Video Diffusion Models into Controllable 3D Avatars](https://arxiv.org/abs/2610.23772)
**ArXiv:** 2610.23772 [[page](https://arxiv.org/abs/2610.23772)] [[pdf](https://arxiv.org/pdf/2610.23772.pdf)]

**Authors:** Simon Giebenhain, Tobias Kirschstein, Liam Schoneveld, ..., Zhe Chen, Matthias Nießner

**Abstract:** Video diffusion models (VDMs) trained on internet-scale data encode a rich prior on appearance, motion, and physics that is difficult to replicate with explicit 3D rendering and costly 3D data capture. We introduce Video Diffusion Avatars (VDAs), a method that creates photo-realistic, multi-view consistent avatars from a single image by fine-tuning a VDM for precise control over camera viewpoint, facial expression, and upper-body pose. Departing from 3D morphable models (3DMMs), VDAs use a learned expression space that lifts the expressiveness ceiling of 3DMMs, e.g., for the tongue and mouth interior. However, learned expression codes exhibit heavy viewpoint bias, as our DISTRACT benchmark reveals. We show that this bias can be removed through disentanglement, enabling their use as driving signals for multi-view consistent avatars. Camera and upper-body pose are controlled jointly via proxy-mesh renderings, giving independent control over expression, pose, and viewpoint. A camera-triplet training strategy makes the approach efficient: our models train for 34 hours on a single A100 using only public multi-view data. VDAs outperform state-of-the-art single-image 3D avatars on novel-view reenactment, reducing face LPIPS by 29% and improving perceptual quality (JOD) by 66% on BecomingLit. A static 3DGS reconstruction probe further confirms that our generations are the most multi-view consistent. Built on a video prior, VDAs produce effects beyond the reach of explicit renderers, including physically plausible hair dynamics, shadows, reflections, and refractions.

**Comment:** Criterion 5: Video Diffusion Avatars fine-tunes a video diffusion model to create controllable, multi-view-consistent avatars from a single image, reducing face LPIPS by 29% on BecomingLit and validating geometric consistency through a 3DGS reconstruction probe.

**Relevance:** 9
Back to [[topic](#topic-5)] [[top](#topics)]

<a id="5004-persistent-planar-memory-for-video-world-models"></a>

### 5004\. [Persistent Planar Memory for Video World Models](https://arxiv.org/abs/2610.23795)
**ArXiv:** 2610.23795 [[page](https://arxiv.org/abs/2610.23795)] [[pdf](https://arxiv.org/pdf/2610.23795.pdf)]

**Authors:** Yuze He, Bin Tan, Zelin Gao, ..., Nan Xue, Yinghao Xu

**Abstract:** Video world models that generate and explore environments at scale need a persistent 3D memory of what they have produced. Plane primitives offer a natural candidate: a compact set of oriented planes captures the dominant structural elements (e.g., walls, floors, facades) of indoor and outdoor scenes, while requiring only 7 parameters per primitive. We present PPMem, a video world model that adopts plane primitives as its persistent 3D memory. PPMem augments a video diffusion transformer with a planar prediction head and a Depth Token Fusion (DTF) module: each autoregressive chunk jointly produces RGB video and planar geometry, which merges into a persistent plane memory that conditions subsequent chunks via rendered depth. The resulting memory is three orders of magnitude more compact than per-pixel point clouds (~100K primitives, 5MB for a full scene), geometrically explicit (metric depth, normals, exportable surfaces), cross-view consistent (jointly predicted by the shared backbone), and adaptive (refined by the merging procedure as generation proceeds). On DL3DV and RealEstate10K, PPMem outperforms prior 3D-grounded world models in video quality while jointly producing metric-scale depth, surface normals, and exportable planar geometry, sustaining coherence across hundreds of frames under large viewpoint changes. Project page: https://hyzcluster.github.io/ppmem.

**Comment:** Criterion 5: PPMem jointly generates video and exportable 3D planar geometry through a diffusion transformer with planar prediction and Depth Token Fusion, maintaining a persistent scene memory three orders of magnitude more compact than per-pixel point clouds.

**Relevance:** 8
Back to [[topic](#topic-5)] [[top](#topics)]

---
<a id="7003-learning-robot-metareasoning-adaptive-computation-budgets-for-real-time-control"></a>

### 7003\. [Learning Robot Metareasoning: Adaptive Computation Budgets for Real-Time Control](https://arxiv.org/abs/2610.23662)
**ArXiv:** 2610.23662 [[page](https://arxiv.org/abs/2610.23662)] [[pdf](https://arxiv.org/pdf/2610.23662.pdf)]

**Authors:** Lorenzo Magnino, Antonio Marino, Michael Amir, Daigo Shishika, Amanda Prorok

**Abstract:** Planning while moving faces an historical trade-off between deliberation and reactivity. A planner running concurrently with execution needs time to compute, so the robot must commit to an open-loop action chunk for that duration: a longer commitment buys a better plan at the cost of feedback frequency. Existing controllers fix this action commitment, and consequently the computation budget, before deployment. However, the optimal computation time is highly state-dependent (due to factors such as unmodeled dynamics, safety constraints, and marginal optimisation gains), i.e. dynamic, therefore offline static tuning risks leaving systems blind during critical maneuvers or unable to find global effective solutions. We cast the identification of the optimal computation budget as a semi-Markov decision process (SMDP) in which a single decision defines both the robot action commitments and the time allocated to the solver. To solve this, we train a reinforcement-learning policy conditioned on the current state, the executing plan, and inexpensive planner statistics. Across ground-robot navigation, agile quadrotor flight, and contact-rich manipulation, the learned policy changes its budget with the demands of the scenario, discovering efficient reasoning patterns that outperform fixed-budget baselines. https://lorenzomagnino.github.io/learning-robot-metareasoning-review/

**Comment:** Criterion 7: Formulates robot metareasoning as an SMDP and trains an RL policy to adapt planning budgets and action commitments, outperforming fixed-budget baselines on robotic tasks including contact-rich manipulation.

**Relevance:** 8
Back to [[topic](#topic-7)] [[top](#topics)]
