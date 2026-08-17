---
title: Arxiv Daily
---

# Personalized Daily Arxiv Papers 08/17/2026

This project is adapted from [tatsu-lab/gpt_paper_assistant](https://github.com/tatsu-lab/gpt_paper_assistant). The source code of this project is at [Variante/gpt_paper_assistant](https://github.com/Variante/gpt_paper_assistant)

About me on [Bilibili](https://space.bilibili.com/823532). Help keep the website running:

<a href="https://www.buymeacoffee.com/Variante"><img src="https://img.buymeacoffee.com/button-api/?text=Help cover GPT cost&emoji=🍙&slug=Variante&button_colour=081c71&font_colour=ffffff&font_family=Comic&outline_colour=000000&coffee_colour=FFDD00" /></a>


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

[Go beyond](#go-beyond)


---
<a id="topic-1"></a>

### Topic 1
1000\. [hint$^2$: Hierarchical World Models for Inference-Time Temporal Logic Guidance](https://arxiv.org/abs/2608.13678) [[more](#1000-hint2-hierarchical-world-models-for-inference-time-temporal-logic-guidance)]  
**Authors:** Moritz Zoellner, Anastasios Manganaris, Ahmed H. Qureshi, Rohan Paleja

1002\. [Evolve Vision-Language-Action Model into an Agent with On-the-fly Tool-use](https://arxiv.org/abs/2608.14047) [[more](#1002-evolve-vision-language-action-model-into-an-agent-with-on-the-fly-tool-use)]  
**Authors:** Yi Ding, Yanzhao Yu, Xili Dai, Xianbiao Qi, Peiwen Sun, Xueqian Wang, Xiangyu Yue, Jianan Wang

1003\. [Reflex: Enabling Fast and Predictive Vision-Language-Action Models for Reaction-Critical Manipulation](https://arxiv.org/abs/2608.14379) [[more](#1003-reflex-enabling-fast-and-predictive-vision-language-action-models-for-reaction-critical-manipulation)]  
**Authors:** Yuxuan Chen, Wanruo Zhang, Xiao Li

1004\. [BICPO-VLA: Behavior-Identified Continuation Preference Optimization for Smooth Asynchronous Vision-Language-Action Control](https://arxiv.org/abs/2608.13924) [[more](#1004-bicpo-vla-behavior-identified-continuation-preference-optimization-for-smooth-asynchronous-vision-language-action-control)]  
**Authors:** Ming Shang, Yuchen Huang, Jiaoyang Chen, Haoyuan Hu, Han Yu, Liping Song, Luyun Feng, Shuo Bao, Wei Dong, Xinzhou Wang, Fuchun Sun


Back to [[top](#topics)]

---
<a id="topic-2"></a>

### Topic 2
2006\. [GhostPoint: Self-Supervised Representation Learning by Hallucinating Occluded LiDAR Structure](https://arxiv.org/abs/2608.14428) [[more](#2006-ghostpoint-self-supervised-representation-learning-by-hallucinating-occluded-lidar-structure)]  
**Authors:** Mohamed Abdelsamad, Bin Yang, Michael Ulrich, Miao Zhang, Yakov Miron, Alexandru Paul Condurache, Abhinav Valada

2007\. [Self-Supervised Visual On-Policy Distillation](https://arxiv.org/abs/2608.14144) [[more](#2007-self-supervised-visual-on-policy-distillation)]  
**Authors:** Yijiang Li, Yijun Liang, Yunjie Tian, Bingyang Wang, Ke Zhang, Zhenfei Yin, Di Fu, Philip Torr, Nuno Vasconcelos


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
5005\. [Owner3D: Ownership-Guided Style Writing for Training-Free Localized 3D Stylization](https://arxiv.org/abs/2608.14078) [[more](#5005-owner3d-ownership-guided-style-writing-for-training-free-localized-3d-stylization)]  
**Authors:** Suchang Tao, Kaifeng Shi, Zhiyan Liu, Zhuoyuan Jiang, Yuqi Ouyang


Back to [[top](#topics)]

---
<a id="topic-6"></a>

### Topic 6
6001\. [HiCo-GS: Hierarchical Context Aggregation and Geometric Consistency for Octree Gaussian Splatting](https://arxiv.org/abs/2608.14136) [[more](#6001-hico-gs-hierarchical-context-aggregation-and-geometric-consistency-for-octree-gaussian-splatting)]  
**Authors:** Wei Zhang, Shengkai Yu, Shiqiang Gong, Qi Zhang, Qiang Li, Qi Wang


Back to [[top](#topics)]

---
<a id="go-beyond"></a>

### Go beyond


Back to [[top](#topics)]

---
## Full paper list
 <a id="1000-hint2-hierarchical-world-models-for-inference-time-temporal-logic-guidance"></a>

### 1000\. [hint$^2$: Hierarchical World Models for Inference-Time Temporal Logic Guidance](https://arxiv.org/abs/2608.13678)
**ArXiv:** 2608.13678 [[page](https://arxiv.org/abs/2608.13678)] [[pdf](https://arxiv.org/pdf/2608.13678.pdf)]

**Authors:** Moritz Zoellner, Anastasios Manganaris, Ahmed H. Qureshi, Rohan Paleja

**Abstract:** A central goal of robot learning is to enable robots to execute rich instructions specified at runtime. Large-scale language-conditioned policies have made substantial progress toward this goal, yet still struggle with temporal structure and safety constraints. Linear Temporal Logic (LTL) provides a powerful language to express complex, non-Markovian instructions. However, guiding learned manipulation policies toward LTL satisfaction remains challenging because modern policies generate short-horizon action chunks and replan in closed loop, while almost all LTL specifications are evaluated over long-horizon trajectories. In this paper, we introduce hint$^2$, a method for guiding short-horizon policies toward satisfying complex LTL specifications at inference time using hierarchical world models. Our key idea is to derive two separate guidance objectives using each world model's abstraction level. A high-level model predicts future action-induced transitions in task-relevant atomic propositions to guide progress through the LTL automaton, while a low-level dynamics model predicts immediate state evolution for accurate local safety guidance. Our results show that hint$^2$ overcomes the limitations of current LTL-guided diffusion methods, outperforms existing inference-time steering methods in CALVIN, and successfully completes instructions with complex liveness and safety constraints more elegantly than language-conditioned alternatives. Finally, we demonstrate that hint$^2$ can handle complex instructions on a real UR5e manipulator.

**Comment:** Criterion 1: this paper applies hierarchical world models for inference-time guidance of language-conditioned manipulation policies, using LTL-guided planning over atomic propositions and demonstrating improved execution of complex instruction constraints on CALVIN and a real UR5e robot.

**Relevance:** 10
Back to [[topic](#topic-1)] [[top](#topics)]

<a id="1002-evolve-vision-language-action-model-into-an-agent-with-on-the-fly-tool-use"></a>

### 1002\. [Evolve Vision-Language-Action Model into an Agent with On-the-fly Tool-use](https://arxiv.org/abs/2608.14047)
**ArXiv:** 2608.14047 [[page](https://arxiv.org/abs/2608.14047)] [[pdf](https://arxiv.org/pdf/2608.14047.pdf)]

**Authors:** Yi Ding, Yanzhao Yu, Xili Dai, Xianbiao Qi, Peiwen Sun, Xueqian Wang, Xiangyu Yue, Jianan Wang

**Abstract:** This paper integrates end-to-end Visual-Language-Action (VLA) models with agentic tool-use to propose Agentic Robot with Tool-use (ART). ART is a tool-injection framework that tunes any VLA model to leverage off-the-shelf tool modules for low-level vision, high-level affordance, and embodiment enhancement. Compared to vanilla VLA models with a whole continuous action solution space, ART reduces the complexity of the action solution space through tool-use, which not only improves generalizability across different tasks but also reduces data dependency. To demonstrate the advantages (high generalizability and low data dependency) of this framework, we first built a dataset of 30K tool-use trajectories and action demonstrations, which is much smaller than those used by baseline methods. We then designed a training regimen for long-trajectory tool-use reasoning in challenging environments. Experiments show that ART achieves a 20% higher success rate than mainstream baselines on simulation and real-world tasks, such as pick-and-place in the dark at novel viewpoints. Empirical results highlight the benefits of an agent-based approach: modular tool utilization enables more efficient training, lightweight deployment, and scalable integration of new tools. This design fosters robustness, adaptability, and extensibility, paving the way for the practical deployment of VLA systems in complex real-world scenarios.

**Comment:** Criterion 1: it explicitly evolves a VLA model into an agent via the ART tool-injection framework (off-the-shelf vision, affordance, and embodiment tools), with reported 20% higher success on simulation/real pick-and-place manipulation tasks.

**Relevance:** 10
Back to [[topic](#topic-1)] [[top](#topics)]

<a id="1003-reflex-enabling-fast-and-predictive-vision-language-action-models-for-reaction-critical-manipulation"></a>

### 1003\. [Reflex: Enabling Fast and Predictive Vision-Language-Action Models for Reaction-Critical Manipulation](https://arxiv.org/abs/2608.14379)
**ArXiv:** 2608.14379 [[page](https://arxiv.org/abs/2608.14379)] [[pdf](https://arxiv.org/pdf/2608.14379.pdf)]

**Authors:** Yuxuan Chen, Wanruo Zhang, Xiao Li

**Abstract:** Vision-Language-Action (VLA) models have recently achieved promising performance in robotic manipulation. However, existing benchmarks mainly evaluate generalization on static manipulation tasks and largely overlook dynamic interaction scenarios. To address this gap, we present ReflexBench, a benchmark for reaction-critical manipulation. ReflexBench contains six dynamic tasks and introduces an evaluation framework that decouples simulator stepping from robot control while supporting configurable latency under synchronous and asynchronous inference. Building upon ReflexBench, we propose ReflexVLA, an efficient VLA model designed for reaction-critical manipulation without large-scale robot-data pretraining. ReflexVLA enhances temporal reasoning through latent future prediction and multi-frame temporal fusion within the vision backbone, while reducing deployment latency through batched visual encoding and CUDA Graph replay. Experiments show that ReflexVLA consistently improves dynamic manipulation performance while maintaining competitive accuracy on standard static manipulation benchmarks, and real-world experiments further demonstrate its effectiveness under practical deployment conditions. Project website: https://reflexvla.github.io

**Comment:** Criterion 1: this work is explicitly about VLA-based robotic manipulation, introducing ReflexBench for reaction-critical tasks and ReflexVLA with latent future prediction and temporal fusion to improve manipulation control under latency constraints.

**Relevance:** 10
Back to [[topic](#topic-1)] [[top](#topics)]

<a id="1004-bicpo-vla-behavior-identified-continuation-preference-optimization-for-smooth-asynchronous-vision-language-action-control"></a>

### 1004\. [BICPO-VLA: Behavior-Identified Continuation Preference Optimization for Smooth Asynchronous Vision-Language-Action Control](https://arxiv.org/abs/2608.13924)
**ArXiv:** 2608.13924 [[page](https://arxiv.org/abs/2608.13924)] [[pdf](https://arxiv.org/pdf/2608.13924.pdf)]

**Authors:** Ming Shang, Yuchen Huang, Jiaoyang Chen, Haoyuan Hu, Han Yu, Liping Song, Luyun Feng, Shuo Bao, Wei Dong, Xinzhou Wang, Fuchun Sun

**Abstract:** The request-to-handoff gap has three coupled sources: ambiguity about the behavior intended at request time, physical-state drift accumulated during action generation, and residual incompatibility when the new action finally assumes control. BICPO-VLA addresses them in sequence. First, an instruction-aware causal history encoder identifies the behavior supported by the command and current task progress. Second, sequential Haar subspace generation decomposes each action chunk into complementary pairwise scaffold and residual coefficients, enabling two specialized generation stages followed by exact reconstruction. By reducing iterative refinement in the original action space, it shortens the interval over which the robot continues moving before the new chunk becomes available. Finally, BICPO rolls the known outgoing actions to the actual handoff state and applies reference-relative Flow-DPO among behaviorally matched candidates, adapting the generated chunk to the remaining request-to-handoff mismatch without changing its intended behavior.

**Comment:** Criterion 1: this paper directly targets vision-language-action robotic control by using an instruction-aware causal history encoder plus reference-relative Flow-DPO for smoother asynchronous behavior continuation, fitting VLM-enabled robotic policy execution.

**Relevance:** 9
Back to [[topic](#topic-1)] [[top](#topics)]

---
<a id="2006-ghostpoint-self-supervised-representation-learning-by-hallucinating-occluded-lidar-structure"></a>

### 2006\. [GhostPoint: Self-Supervised Representation Learning by Hallucinating Occluded LiDAR Structure](https://arxiv.org/abs/2608.14428)
**ArXiv:** 2608.14428 [[page](https://arxiv.org/abs/2608.14428)] [[pdf](https://arxiv.org/pdf/2608.14428.pdf)]

**Authors:** Mohamed Abdelsamad, Bin Yang, Michael Ulrich, Miao Zhang, Yakov Miron, Alexandru Paul Condurache, Abhinav Valada

**Abstract:** 3D object detection from LiDAR point clouds is a core problem in autonomous driving. Recent advances in self-supervised learning (SSL) enable scalable pretraining and transfers well to per-point tasks such as semantic and panoptic segmentation, but transfer to 3D detection remains weaker. We analyze recent SSL methods and find that most objectives are defined only on measured LiDAR returns from visible surfaces, leaving occluded and unobserved regions unconstrained. This visible-surface bias can be sufficient for point-wise prediction, but 3D detection requires robustness to missing structure. To address this gap, we propose GhostPoint, an SSL framework that hallucinates latent features in local neighborhoods around discovered instances, generated via a novel instance voxel dilation. In GhostPoint, an encoder processes observed returns, and an additional predictor infers neighborhood representations from observed context. In addition to standard encoder-level supervision, we introduce a predictor-level supervision scheme on sampled voxels from generated neighborhoods. Specifically, observed (visible/masked) voxels match teacher-encoder targets, while unobserved voxels match teacher-predictor hallucinations. This design encourages the learned representation to explicitly model structure beyond observed returns. Extensive evaluations on nuScenes and Waymo demonstrate that our method achieves state-of-the-art performance, consistently improving downstream 3D detection, especially under sparse scans and limited labels.

**Comment:** Matches criterion 2: it introduces GhostPoint, a new self-supervised representation-learning framework for LiDAR that hallucinates occluded structure via instance voxel dilation and predictor-level supervision, with reported state-of-the-art improvements on downstream 3D detection (nuScenes and Waymo), especially under sparse scans.

**Relevance:** 8
Back to [[topic](#topic-2)] [[top](#topics)]

<a id="2007-self-supervised-visual-on-policy-distillation"></a>

### 2007\. [Self-Supervised Visual On-Policy Distillation](https://arxiv.org/abs/2608.14144)
**ArXiv:** 2608.14144 [[page](https://arxiv.org/abs/2608.14144)] [[pdf](https://arxiv.org/pdf/2608.14144.pdf)]

**Authors:** Yijiang Li, Yijun Liang, Yunjie Tian, Bingyang Wang, Ke Zhang, Zhenfei Yin, Di Fu, Philip Torr, Nuno Vasconcelos

**Abstract:** Visual on-policy distillation relies heavily on an informative teacher-student asymmetry, through either a larger, stronger teacher or privileged supervision, such as reference answers or ground-truth regions of interest. This raises a fundamental question: where can informative asymmetry come from when nothing privileged is available? We answer this by inverting where the asymmetry comes from. Rather than adding privileged information to the teacher, we subtract information from the student. This asymmetry creates the same effective learning signal for free as a teacher with access to information unavailable to the student, without ground-truth annotations, rewards, or a separate stronger teacher model. Building on this principle, we introduce Self-Supervised Visual On-Policy Distillation (S$^2$VOPD), a simple yet effective method that constructs on-policy learning signals from asymmetric augmented views. S$^2$VOPD distills the teacher's distribution conditioned on the original image on-policy into the student distribution conditioned on a strongly augmented view of the same image. We systematically explore a broad design space of visual augmentations and uncover that (1) asymmetry matters: all four augmentation families improve performance, while symmetric self-distillation degrades it; (2) strength matters: performance peaks at a moderate strength; and (3) the gap must remain task-consistent: augmentations that completely remove the question-relevant evidence can induce large but uninformative discrepancies. Across six fine-grained perception benchmarks, S$^2$VOPD improves Qwen3.5-4B from 70.7% to 77.4%, above all open-source models compared, up to Qwen3-VL at 235B, and surpasses GPT-5.4. While holding training data the same, it recovers 96% of the improvement achieved by methods with privileged information. Website is at https://williamium3000.github.io/s2vopd

**Comment:** Criterion 2: S²VOPD proposes a new self-supervised visual SSL objective with asymmetric teacher-student distillation between original and strongly augmented views, and systematically evaluates augmentation choices while improving Qwen3.5-4B from 70.7% to 77.4% on six fine-grained perception benchmarks.

**Relevance:** 6
Back to [[topic](#topic-2)] [[top](#topics)]

---
<a id="5005-owner3d-ownership-guided-style-writing-for-training-free-localized-3d-stylization"></a>

### 5005\. [Owner3D: Ownership-Guided Style Writing for Training-Free Localized 3D Stylization](https://arxiv.org/abs/2608.14078)
**ArXiv:** 2608.14078 [[page](https://arxiv.org/abs/2608.14078)] [[pdf](https://arxiv.org/pdf/2608.14078.pdf)]

**Authors:** Suchang Tao, Kaifeng Shi, Zhiyan Liu, Zhuoyuan Jiang, Yuqi Ouyang

**Abstract:** Localized 3D stylization aims to modify the appearance of a specified object part while preserving the remaining surfaces. In large reconstruction models (LRMs), this task is challenging because style is injected into intermediate appearance representations before rendering, while compact triplane features are shared across target and non-target surfaces, causing style leakage and boundary ambiguity. We propose Owner3D, a training-free framework for localized 3D stylization that integrates localized appearance control directly into the LRM reconstruction process. Specifically, Owner3D introduces ownership-guided style writing to restrict reference-style injection to target regions, producing a single localized stylized triplane without additional training while avoiding separate global style and appearance representations. To resolve appearance ambiguity near semantic boundaries, we further introduce boundary dual slots that maintain separate local feature sources for target and non-target regions. Finally, a surface-first texture readout hierarchically combines surface, 3D, and triplane ownership evidence to robustly recover appearance under incomplete visibility. Experiments on a benchmark constructed from Google Scanned Objects and PartNet demonstrate that Owner3D consistently outperforms existing 3D stylization methods in target-region style fidelity and non-target appearance preservation, reducing appearance leakage by 86.4% and 89.9% compared with StyleSplat and LAENeRF, respectively.

**Comment:** Criterion 5: this paper directly advances 3D asset generation quality by introducing ownership-guided style writing with boundary dual slots for localized 3D stylization in triplane reconstruction models, reporting reduced appearance leakage of 86.4% and 89.9% versus StyleSplat and LAENeRF on Google Scanned Objects and PartNet.

**Relevance:** 8
Back to [[topic](#topic-5)] [[top](#topics)]

---
<a id="6001-hico-gs-hierarchical-context-aggregation-and-geometric-consistency-for-octree-gaussian-splatting"></a>

### 6001\. [HiCo-GS: Hierarchical Context Aggregation and Geometric Consistency for Octree Gaussian Splatting](https://arxiv.org/abs/2608.14136)
**ArXiv:** 2608.14136 [[page](https://arxiv.org/abs/2608.14136)] [[pdf](https://arxiv.org/pdf/2608.14136.pdf)]

**Authors:** Wei Zhang, Shengkai Yu, Shiqiang Gong, Qi Zhang, Qiang Li, Qi Wang

**Abstract:** Octree-based anchor Gaussian Splatting has emerged as a scalable representation for city-scale novel view synthesis, where multi-level anchors adaptively capture scene content from coarse building structures to fine architectural details. However, we identify a fundamental limitation in existing methods: cross-level feature isolation, where each level's anchor features are optimized independently with no inter-level communication, causing color drift on building facades and over-smoothing in textured regions. We present HiCo-GS, a high-fidelity reconstruction framework with two complementary modules. Cross-Level Context Aggregation (CLCA) enables bidirectional hierarchical prior injection by leveraging the octree's spatial containment structure to aggregate per-level context vectors into parent-self-child triplets, fused via a lightweight MLP with residual connection. Coarse-level structural priors flow down to inform fine-level anchors, while fine-level detail statistics feed back to prevent over-smoothing, at negligible computational overhead. Depth-Normal Geometric Consistency (DNGC) regularization enforces agreement between rendered normals and depth-derived normals through an alpha-weighted consistency loss, complemented by edge-aware smoothness losses with progressive warmup that exploit the strong planar priors ubiquitous in urban geometry to suppress floating artifacts. We further introduce the China-Pagoda dataset comprising 8 ancient Chinese pagodas with over 1,200 images each, featuring dense ornamental carvings, curved multi-layer eaves, and repetitive fine-grained textures. Extensive experiments on Mill19, UrbanScene3D, MatrixCity, and China-Pagoda demonstrate that HiCo-GS achieves state-of-the-art rendering quality and substantially cleaner geometry across real-world and synthetic urban benchmarks.Code: https://github.com/WZ-CS/HiCo-GS.

**Comment:** Criterion 6: HiCo-GS is a direct 3D reconstruction advance using Gaussian Splatting, introducing CLCA and DNGC for hierarchical context/geometric consistency and reporting SOTA urban novel-view reconstruction performance on Mill19, UrbanScene3D, MatrixCity, and China-Pagoda.

**Relevance:** 10
Back to [[topic](#topic-6)] [[top](#topics)]
