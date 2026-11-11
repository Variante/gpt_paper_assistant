---
title: Arxiv Daily
---

# Personalized Daily Arxiv Papers 11/11/2026

<time id="gptpa-last-update" datetime="2026-11-11T15:00:01.911487+00:00">Last update: 2026-11-11T15:00:01.911487+00:00</time>

This project is adapted from [tatsu-lab/gpt_paper_assistant](https://github.com/tatsu-lab/gpt_paper_assistant). The source code of this project is at [Variante/gpt_paper_assistant](https://github.com/Variante/gpt_paper_assistant)

About me on [Bilibili](https://space.bilibili.com/823532). Help keep the website running:

<a href="https://www.buymeacoffee.com/Variante"><img src="https://img.buymeacoffee.com/button-api/?text=Help cover GPT cost&emoji=🐱&slug=Variante&button_colour=28513e&font_colour=ffffff&font_family=Poppins&outline_colour=000000&coffee_colour=FFDD00" /></a>


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
1000\. [Elastic-WAM: Adaptive Spatiotemporal Imagination for Efficient Robot Control](https://arxiv.org/abs/2611.14257) [[more](#1000-elastic-wam-adaptive-spatiotemporal-imagination-for-efficient-robot-control)]  
**Authors:** Haitian Li, Tianfu Li, Xuxi Zhao, ..., Haonan Chen, Lin Shao

1001\. [From Video Frames to Robot Meta-Frames: A Unified Latent Interface for World Action Models](https://arxiv.org/abs/2611.14365) [[more](#1001-from-video-frames-to-robot-meta-frames-a-unified-latent-interface-for-world-action-models)]  
**Authors:** Guoqiang Liang, Yiqi Lin, Yanzhe Chen, Kevin Yuchen Ma, Mike Zheng Shou

1003\. [One Policy for 100 Household Tasks: Our Solution for the 2026 BEHAVIOR Challenge](https://arxiv.org/abs/2611.14432) [[more](#1003-one-policy-for-100-household-tasks-our-solution-for-the-2026-behavior-challenge)]  
**Authors:** Ilia Larchenko, Gleb Zarin, Atindra Nair, ..., Mirko Kiefer, Anjana Yodaiken

1004\. [Is Your Robot's Memory Just a Clock? Temporal Shortcuts in Memory-Augmented VLA Policies](https://arxiv.org/abs/2611.14488) [[more](#1004-is-your-robots-memory-just-a-clock-temporal-shortcuts-in-memory-augmented-vla-policies)]  
**Authors:** Chubin Zhang, Bo An


Back to [[top](#topics)]

---
<a id="topic-2"></a>

### Topic 2
2002\. [OnlyAct: Extracting Action Representations from Foundation Models for World Models](https://arxiv.org/abs/2611.14420) [[more](#2002-onlyact-extracting-action-representations-from-foundation-models-for-world-models)]  
**Authors:** Runjia Li, Minghao Chen, Junyu Xie, ..., Andrea Vedaldi, Tomas Jakab

2005\. [RS-JEPA: Remote Sensing Joint-Embedding Predictive Architecture with Scan-Path Views for Cross-Sensor Image Retrieval](https://arxiv.org/abs/2611.14496) [[more](#2005-rs-jepa-remote-sensing-joint-embedding-predictive-architecture-with-scan-path-views-for-cross-sensor-image-retrieval)]  
**Authors:** Tanishq Rachamalla, Srishti Kaushik, Muhammad Attique Khan, Gustau Camps-Valls, Swalpa Kumar Roy


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


Back to [[top](#topics)]

---
<a id="topic-6"></a>

### Topic 6


Back to [[top](#topics)]

---
<a id="topic-7"></a>

### Topic 7


Back to [[top](#topics)]

---
<a id="go-beyond"></a>

### Go beyond


Back to [[top](#topics)]

---
## Full paper list
 <a id="1000-elastic-wam-adaptive-spatiotemporal-imagination-for-efficient-robot-control"></a>

### 1000\. [Elastic-WAM: Adaptive Spatiotemporal Imagination for Efficient Robot Control](https://arxiv.org/abs/2611.14257)
**ArXiv:** 2611.14257 [[page](https://arxiv.org/abs/2611.14257)] [[pdf](https://arxiv.org/pdf/2611.14257.pdf)]

**Authors:** Haitian Li, Tianfu Li, Xuxi Zhao, ..., Haonan Chen, Lin Shao

**Abstract:** World-Action Models improve robot control by predicting future visual dynamics, but iterative video denoising incurs substantial inference overhead. Recent approaches reduce this cost through partial future denoising, yet typically rely on fixed refinement budgets. Our empirical observations reveal that preferred refinement depths vary across tasks and states, while different future regions exhibit varying relevance to action generation. Motivated by these findings, we propose Elastic-WAM, which adaptively allocates world refinement computation across temporal denoising time and spatial future space. A unified spatiotemporal refinement demand field governs when to stop world refinement, how far to advance each denoising step, and which regions to refresh. We parameterize this field with a shared temporal basis and spatial amplitudes, and learn it through a three-stage action-grounded supervised procedure. At inference, the learned field guides recurrent sparse world refinement to improve task success while reducing inference overhead. Evaluations across diverse simulation benchmarks and real-world long-horizon manipulation demonstrate state-of-the-art control performance with reduced inference latency.

**Comment:** Matches criterion 1: Elastic-WAM adaptively allocates spatiotemporal denoising computation in a World-Action Model, achieving state-of-the-art robot control with reduced inference latency across simulation benchmarks and real-world long-horizon manipulation.

**Relevance:** 10
Back to [[topic](#topic-1)] [[top](#topics)]

<a id="1001-from-video-frames-to-robot-meta-frames-a-unified-latent-interface-for-world-action-models"></a>

### 1001\. [From Video Frames to Robot Meta-Frames: A Unified Latent Interface for World Action Models](https://arxiv.org/abs/2611.14365)
**ArXiv:** 2611.14365 [[page](https://arxiv.org/abs/2611.14365)] [[pdf](https://arxiv.org/pdf/2611.14365.pdf)]

**Authors:** Guoqiang Liang, Yiqi Lin, Yanzhe Chen, Kevin Yuchen Ma, Mike Zheng Shou

**Abstract:** Pretrained video diffusion models (VDMs) offer a powerful foundation for World Action Models (WAMs), which jointly predict robot actions and future observations. Building such models requires injecting low-dimensional robot signals (e.g. proprioception, actions, and values) into a latent space shaped by natural videos, and a key design choice is the interface through which these signals enter: existing schemes treat it as a formatting detail, spending a separate latent frame on each signal or bolting on dedicated encoders and decoders. We argue that signal injection is instead a distribution-alignment problem: how well a representation matches the prior of the pretrained VDMs during denoising. Using the relative reconstruction error of a frozen VDM denoiser as a probe, we find that conventional spatial encodings sit far off this prior, whereas frequency-domain packing yields latents that the denoiser corrects far less. Motivated by this diagnostic result, we propose MetaWAM, which folds all robot signals into a single unified meta-frame: each signal is assigned to a disjoint frequency-domain channel band and lifted to the spatial latent by an inverse DCT, yielding a decodable interface aligned with the pretrained VDM prior without any added module or architectural change. A fixed band layout together with slot-aware routing lets the same meta-frame support policy, world-model, and value prediction over a shared set of robot variables. On RoboCasa, MetaWAM reaches 68.9% success while collapsing the per-signal frames into one and cutting inference latency by 39%, and attains 85.0% success on three real-world Franka tasks. Code and models are available at https://github.com/showlab/MetaWAM

**Comment:** Criterion 1: MetaWAM adapts pretrained video diffusion models for robot policy, world-model, and value prediction using a unified frequency-packed meta-frame, achieving 68.9% success on RoboCasa while reducing inference latency by 39%.

**Relevance:** 10
Back to [[topic](#topic-1)] [[top](#topics)]

<a id="1003-one-policy-for-100-household-tasks-our-solution-for-the-2026-behavior-challenge"></a>

### 1003\. [One Policy for 100 Household Tasks: Our Solution for the 2026 BEHAVIOR Challenge](https://arxiv.org/abs/2611.14432)
**ArXiv:** 2611.14432 [[page](https://arxiv.org/abs/2611.14432)] [[pdf](https://arxiv.org/pdf/2611.14432.pdf)]

**Authors:** Ilia Larchenko, Gleb Zarin, Atindra Nair, ..., Mirko Kiefer, Anjana Yodaiken

**Abstract:** We describe our solution for the 2026 BEHAVIOR Challenge: 100 long-horizon household tasks in the OmniGibson simulator. We placed 9th on the final leaderboard, the only independent team in the top 10. The core of the solution is a single vision-language-action (VLA) model, one checkpoint for all 100 tasks, built on our 1st-place 2025 solution, which is based on $π_{0.5}$. On the 50 tasks shared with 2025, its Q rises from 0.260 to 0.371; across all 100 tasks it scores 0.392. Our main conclusion from 2025 was that the model was undertrained, so most of the 2026 work went into training the model for longer, optimizing some training parameters, making each training step slightly faster and reducing the gradient noise: a migration from 50 to 100 tasks (semantic stages built from the demonstrations' skill annotations, task and stage embeddings of new tasks warm-started from similar old ones, and a 21-dimensional action space matching the teleoperation controls); variance-reduced multi-sample flow matching with a simple noise-vs-compute model that picks the number of action-expert samples per observation without ablations; and a 200k-step training run, about one pass over the 211M frames of the demonstrations. A cloud evaluation infrastructure keeps a shared record with provenance, episode diagnostics and restorable simulator states, and runs six simulators on one GPU together faster than real time. On top of the learned policy, 14 tasks are routed to a GPT-6 Astra coding agent that controls the robot through observation and motion tools, starting every episode from a frozen copy of its own practice session on training instances. Across these 14 tasks, Astra's mean Q is 0.639, against 0.108 for the learned policy. The public submission included only part of the Astra results and scored 0.446 Q with 27.3% full-task success.

**Comment:** Criteria 1 and 7: This BEHAVIOR Challenge solution combines a single VLA policy for 100 household tasks with a coding agent using observation tools, motion tools, and saved practice sessions, with the agent achieving mean Q of 0.639 versus the learned policy's 0.108 across 14 tasks.

**Relevance:** 9
Back to [[topic](#topic-1)] [[top](#topics)]

<a id="1004-is-your-robots-memory-just-a-clock-temporal-shortcuts-in-memory-augmented-vla-policies"></a>

### 1004\. [Is Your Robot's Memory Just a Clock? Temporal Shortcuts in Memory-Augmented VLA Policies](https://arxiv.org/abs/2611.14488)
**ArXiv:** 2611.14488 [[page](https://arxiv.org/abs/2611.14488)] [[pdf](https://arxiv.org/pdf/2611.14488.pdf)]

**Authors:** Chubin Zhang, Bo An

**Abstract:** Memory-augmented vision-language-action (VLA) policies are meant to remember what happened. Many of them, however, fill memory by time: they sample frames uniformly across the episode and stamp each frame with its time step. Because time passes even when nothing happens, such a memory may not tell waiting from working. We show that it does not. On the counting tasks of RoboMME, we make the robot briefly pause or fidget its wrist. Such a null event changes nothing the task depends on, yet a released memory VLA counts the interval as work done and stops early. We call this failure phantom progress. It cuts success from 47 to 16 of 50 on one task and from 44 to 10 on the other. When the same frames are withheld from memory, success returns, so the cause lies in memory rather than in the interruption itself. We trace the failure to a temporal shortcut: because demonstrations proceed at a steady pace, elapsed time predicts progress. Indeed, rescaling the time stamps alone makes the policy stop early when the clock runs fast and repeat when it runs slow. Surprisingly, removing the time encoding does not cure the failure, because the frames of a null event still occupy slots of the time-sampled memory. Across RoboMME's released designs, vulnerability follows how memory is indexed. Of the fixes the diagnosis suggests, training with injected pauses does not remove the failure and impairs counting, whereas indexing memory by events recovers half or more of the loss without retraining. A memory meant to remember what happened should count events rather than clock ticks.

**Comment:** Criterion 1: Analyzes memory-augmented VLA robotic control policies on RoboMME counting tasks and shows that event-indexed memory recovers at least half of the performance lost to interruption-induced temporal shortcuts without retraining.

**Relevance:** 9
Back to [[topic](#topic-1)] [[top](#topics)]

---
<a id="2002-onlyact-extracting-action-representations-from-foundation-models-for-world-models"></a>

### 2002\. [OnlyAct: Extracting Action Representations from Foundation Models for World Models](https://arxiv.org/abs/2611.14420)
**ArXiv:** 2611.14420 [[page](https://arxiv.org/abs/2611.14420)] [[pdf](https://arxiv.org/pdf/2611.14420.pdf)]

**Authors:** Runjia Li, Minghao Chen, Junyu Xie, ..., Andrea Vedaldi, Tomas Jakab

**Abstract:** Video world models promise general-purpose interactive simulators, but their training is limited by action supervision: action labels are scarce, fragmented across incompatible specifications, and do not scale with the unlabeled video pretraining corpus. A common workaround jointly trains an action encoder to compress consecutive frames into a latent action and a decoder to reconstruct the next frame from the previous frame and that latent, forcing the latent to capture only new information. This has two limitations: learning the action space from scratch can limit generalization to unseen domains, and the bottleneck trades expressiveness for separation, with tighter settings losing nuance and looser ones risking scene leakage under domain shifts. We argue for a different starting point: rather than learning actions from scratch, we start from a pretrained foundation model (V-JEPA 2), whose embeddings of a frame pair already carry substantial action information after large-scale, diverse video pretraining. The remaining challenge is to extract the action while discarding the scene. We propose a label-free contrastive adaptation framework that pulls together frame pairs sharing the same motion under a different scene and pushes apart pairs from the same scene with different motion, reshaping the geometry of the pretrained space without imposing a reconstruction bottleneck. We introduce a unified evaluation protocol along three axes (action expressiveness, scene invariance, and action-conditioned video generation quality) across robot manipulation, camera navigation, game, and driving benchmarks, most of which are unseen during adaptation. Compared with annotation-free latent-action methods, our representations are substantially more expressive while matching their scene invariance, and they yield world models that transfer demonstrated actions to new scenes more faithfully.

**Comment:** Criterion 2: OnlyAct introduces label-free contrastive adaptation of V-JEPA 2 that aligns frame pairs sharing motion across different scenes while separating different motions within the same scene, improving action expressiveness and scene invariance for transferable video world models.

**Relevance:** 9
Back to [[topic](#topic-2)] [[top](#topics)]

<a id="2005-rs-jepa-remote-sensing-joint-embedding-predictive-architecture-with-scan-path-views-for-cross-sensor-image-retrieval"></a>

### 2005\. [RS-JEPA: Remote Sensing Joint-Embedding Predictive Architecture with Scan-Path Views for Cross-Sensor Image Retrieval](https://arxiv.org/abs/2611.14496)
**ArXiv:** 2611.14496 [[page](https://arxiv.org/abs/2611.14496)] [[pdf](https://arxiv.org/pdf/2611.14496.pdf)]

**Authors:** Tanishq Rachamalla, Srishti Kaushik, Muhammad Attique Khan, Gustau Camps-Valls, Swalpa Kumar Roy

**Abstract:** Cross-sensor remote sensing image retrieval aims to find scenes with the same content across heterogeneous sensors, such as SAR and optical or panchromatic (PAN) and multispectral (MS) imagery, without manual labels. Existing self-supervised methods either reconstruct sensor-specific pixel detail, align each image only with its co-registered twin, or rely on large transformers with 118-173M parameters (and up to 398M). We introduce RS-JEPA, a compact joint-embedding predictive framework built on a Vision Mamba encoder shared by both sensors. Since a state-space encoder reads an image as an ordered sequence of patches, we propose scan-path views, which treat two reading orders of the same image as two views with identical pixels and predict the hidden patches in one order from the visible patches in the other.To keep this prediction challenging, we propose Adversarial View Coupling (AVC), which learns the hardest split of the visible patches, and a neighbourhood consensus loss that aligns the scene similarities of both sensors. Extensive experiments on SAR-optical (BEN-14K, CBRSIR-VS) and PAN-MS (DSRSID) benchmarks demonstrate that RS-JEPA achieves the highest F1@5 among twelve re-implemented methods (72.92, 80.35 and 94.43) with a retrieval network of only 3.88M parameters (14.57M in training, 8-16times fewer than prior JEPA retrievers). Like prior methods, RS-JEPA trains on co-registered pairs. Our results highlight reading order as a source of self-supervised views. code: https://github.com/Tanishq251/RSJEPA.

**Comment:** Criterion 2: Introduces self-supervised joint-embedding prediction with scan-path views and Adversarial View Coupling, achieving the highest F1@5 among twelve methods on three cross-sensor image retrieval benchmarks.

**Relevance:** 9
Back to [[topic](#topic-2)] [[top](#topics)]
