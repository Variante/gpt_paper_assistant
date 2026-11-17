---
title: Arxiv Daily
---

# Personalized Daily Arxiv Papers 11/17/2026

<time id="gptpa-last-update" datetime="2026-11-17T15:00:03.278117+00:00">Last update: 2026-11-17T15:00:03.278117+00:00</time>

This project is adapted from [tatsu-lab/gpt_paper_assistant](https://github.com/tatsu-lab/gpt_paper_assistant). The source code of this project is at [Variante/gpt_paper_assistant](https://github.com/Variante/gpt_paper_assistant)

About me on [Bilibili](https://space.bilibili.com/823532). Help keep the website running:

<a href="https://www.buymeacoffee.com/Variante"><img src="https://img.buymeacoffee.com/button-api/?text=Help cover GPT cost&emoji=🎁&slug=Variante&button_colour=4b8b13&font_colour=ffffff&font_family=Cookie&outline_colour=000000&coffee_colour=FFDD00" /></a>


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
1000\. [ThinkWAM: Rethinking Latent Futures for Action Refinement](https://arxiv.org/abs/2611.21258) [[more](#1000-thinkwam-rethinking-latent-futures-for-action-refinement)]  
**Authors:** Haoyun Tang, Baqiao Deng, Enjin Zhou, ..., Zhandong Mei, Huibin Li

1001\. [ScaleVLA: Scaling Vision-Language-Action Models to 552 Billion Parameters](https://arxiv.org/abs/2611.21607) [[more](#1001-scalevla-scaling-vision-language-action-models-to-552-billion-parameters)]  
**Authors:** Ruixi Yu, Dairu Liu, Xuchuan Chen, ..., He Wang, Li Yi


Back to [[top](#topics)]

---
<a id="topic-2"></a>

### Topic 2
2002\. [SADL: On the Demand-Supply Geometry of Representation Collapse](https://arxiv.org/abs/2611.21596) [[more](#2002-sadl-on-the-demand-supply-geometry-of-representation-collapse)]  
**Authors:** Junhan Sun, Yigui Feng, Mingzhe Liu, ..., Hao Zhao, Guofeng Zhang


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
6003\. [Déjà-Events: Multi-view 3D Geometry with Event Cameras via Sparsity-Aware Adaptive Gating](https://arxiv.org/abs/2611.21302) [[more](#6003-dj-events-multi-view-3d-geometry-with-event-cameras-via-sparsity-aware-adaptive-gating)]  
**Authors:** Arda Cinar Demirtas, Pablo Varela Bernal, Marc Pollefeys, Sunghwan Hong


Back to [[top](#topics)]

---
<a id="topic-7"></a>

### Topic 7
7004\. [Lazy-MOSAIC: Accelerating Skill-Centric Manipulation Planning with Learned Skill-Effect World Models](https://arxiv.org/abs/2611.21517) [[more](#7004-lazy-mosaic-accelerating-skill-centric-manipulation-planning-with-learned-skill-effect-world-models)]  
**Authors:** Itamar Mishani, Mukund Krishnakumar, Oren Salzman, Maxim Likhachev


Back to [[top](#topics)]

---
<a id="go-beyond"></a>

### Go beyond


Back to [[top](#topics)]

---
## Full paper list
 <a id="1000-thinkwam-rethinking-latent-futures-for-action-refinement"></a>

### 1000\. [ThinkWAM: Rethinking Latent Futures for Action Refinement](https://arxiv.org/abs/2611.21258)
**ArXiv:** 2611.21258 [[page](https://arxiv.org/abs/2611.21258)] [[pdf](https://arxiv.org/pdf/2611.21258.pdf)]

**Authors:** Haoyun Tang, Baqiao Deng, Enjin Zhou, ..., Zhandong Mei, Huibin Li

**Abstract:** World-action models (WAMs) couple action generation with predictions of future states. An initial action chunk may be flawed. A future prediction that is not conditioned on the chunk may fail to reflect the changes that the proposed actions would induce in visual representations, limiting its usefulness for refinement before execution. We introduce ThinkWAM, a latent WAM that establishes bidirectional feedback before execution through a proposal-rethink-refine loop: the policy first proposes a complete action chunk, the proposal corrects multi-horizon future latents through a learned residual, and the corrected latents are injected into a second pass of the same action head, which generates the executable chunk. An affordance-guided observation-action memory and a near-fine, far-coarse multi-horizon prediction scheme, both motivated by predictive coding, support this loop, and all prediction stays in latent space. ThinkWAM achieves 98.70% success on LIBERO, 84.76% on LIBERO-Plus without OOD fine-tuning, 85.75%/65.85% on RoboTwin 2.0 Clean/Random, and 65.17% on RoboCasa24. ThinkWAM achieves 64.4% mean success across three real-world OOD conditions. The mean two-pass inference latency per action chunk is 153.48 ms on LIBERO, 164.61 ms on RoboTwin, and 101.2 ms on the Galaxea A1Z. Ablations show complementary gains from rethinking and affordance guidance. Diagnostics indicate that proposal-conditioned corrections change both predicted futures and action trajectories, showing that the corrected future representations improve the final action.

**Comment:** Matches criterion 1: ThinkWAM uses a latent world-action model with a proposal–rethink–refine loop to improve robotic action chunks, achieving 98.70% success on LIBERO and 64.4% mean success across three real-world OOD conditions.

**Relevance:** 10
Back to [[topic](#topic-1)] [[top](#topics)]

<a id="1001-scalevla-scaling-vision-language-action-models-to-552-billion-parameters"></a>

### 1001\. [ScaleVLA: Scaling Vision-Language-Action Models to 552 Billion Parameters](https://arxiv.org/abs/2611.21607)
**ArXiv:** 2611.21607 [[page](https://arxiv.org/abs/2611.21607)] [[pdf](https://arxiv.org/pdf/2611.21607.pdf)]

**Authors:** Ruixi Yu, Dairu Liu, Xuchuan Chen, ..., He Wang, Li Yi

**Abstract:** Frontier Vision-Language Models (VLMs) such as GPT-6 Astra have demonstrated remarkable capabilities in robot control, suggesting that broad pretraining provides knowledge valuable for physical action. Yet most Vision-Language-Action (VLA) policies still rely on much smaller VLM backbones, leaving this potential largely untapped. We ask whether frontier-scale representations can provide a stronger foundation for robot learning. We introduce ScaleVLA, which connects a 552B DeepSeek-V4.1-Flash backbone to a trainable action DiT, and systematically compare VLMs from 2B to 552B under a shared training setup. We evaluate three fundamental capabilities of robot foundation models. In sample efficiency, just 10 demonstrations outperform a 2B backbone trained with 50 (87.8% vs. 85.6%). In generalization, LIBERO-PRO success reaches 45.1%, compared with at most 31.3% for smaller backbones. In precise manipulation, real-world charger insertion improves from 4/10 to 9/10 successes, with gains concentrated in contact-sensitive alignment. Despite its scale, ScaleVLA achieves 216 ms median latency per action chunk. Our findings suggest that frontier VLMs can serve not only as reasoning agents, but also as powerful representational foundations for efficient and generalizable robot policies.

**Comment:** Matches criterion 1 by coupling a 552B VLM backbone with a trainable action DiT for robotic manipulation, achieving 45.1% success on LIBERO-PRO and 9/10 successful real-world charger insertions.

**Relevance:** 10
Back to [[topic](#topic-1)] [[top](#topics)]

---
<a id="2002-sadl-on-the-demand-supply-geometry-of-representation-collapse"></a>

### 2002\. [SADL: On the Demand-Supply Geometry of Representation Collapse](https://arxiv.org/abs/2611.21596)
**ArXiv:** 2611.21596 [[page](https://arxiv.org/abs/2611.21596)] [[pdf](https://arxiv.org/pdf/2611.21596.pdf)]

**Authors:** Junhan Sun, Yigui Feng, Mingzhe Liu, ..., Hao Zhao, Guofeng Zhang

**Abstract:** In Joint-Embedding Predictive Architectures (JEPAs), a well-constrained projector output does not by itself prevent representation collapse. We find that, from initialization, the encoder representation is dominated by a component shared by different inputs. A batch-normalized projector removes this component, so regularization after the projector cannot correct it, and the differences between representations shrink until they collapse. We propose SADL (Supply And Demand Loss), which keeps the original training objective and constrains the encoder directly when needed. SADL compares the representation differences that the projector output demands with the differences the encoder can retain and pass on to the projector, and adds a constraint on the encoder only when supply falls short of demand. It works with different Demand Regularizers and introduces no auxiliary networks, additional trainable parameters, or architectural changes. In single-task and jointly trained four-task LeWorldModel (LeWM) and INTACT experiments, SADL lowers the collapse rate over ten random seeds from 50% to 0%. After one epoch of training, encoder effective rank is 3.4-6.4 times that of the corresponding baseline without SADL, and latent prediction error falls to 1/20-1/303 of it. Without additional tuning, SADL stays robust as its loss weight spans orders of magnitude, and with several Demand Regularizers INTACT exceeds 95% mean search-free control success. On four image datasets, SADL also consistently improves accuracy and stability across random seeds. These results show that matching the representation differences a regularizer demands with the differences the encoder can supply leads to faster and more stable representation learning in control and image classification.

**Comment:** Criterion 2: SADL introduces an encoder-level supply–demand loss to prevent representation collapse in self-supervised JEPAs, reducing collapse from 50% to 0% across ten seeds in LeWM and INTACT experiments and improving accuracy and stability on four image datasets.

**Relevance:** 10
Back to [[topic](#topic-2)] [[top](#topics)]

---
<a id="6003-dj-events-multi-view-3d-geometry-with-event-cameras-via-sparsity-aware-adaptive-gating"></a>

### 6003\. [Déjà-Events: Multi-view 3D Geometry with Event Cameras via Sparsity-Aware Adaptive Gating](https://arxiv.org/abs/2611.21302)
**ArXiv:** 2611.21302 [[page](https://arxiv.org/abs/2611.21302)] [[pdf](https://arxiv.org/pdf/2611.21302.pdf)]

**Authors:** Arda Cinar Demirtas, Pablo Varela Bernal, Marc Pollefeys, Sunghwan Hong

**Abstract:** Event cameras keep capturing structure where conventional cameras blur or saturate, which makes them attractive for 3D perception in fast or poorly lit scenes. Event datasets with dense 3D ground truth, however, are far smaller and less diverse than the image collections behind feed-forward 3D reconstruction models. We present Déjà-Events, an event-only model that predicts depth, camera pose and pointmaps from multiple views in a single forward pass. Rather than relying on scarce 3D labels, we distil it from Déjà View, a feed-forward RGB model that refines its tokens by repeatedly applying one shared transformer block. On paired RGB and event recordings, the event student learns to reproduce the frozen teacher's final tokens, which requires no depth or pose ground truth and trains only about 1% of its parameters. Because events are dense at edges and moving texture but nearly absent in textureless regions, we further introduce sparsity-aware adaptive gating, which adapts the shared block's gates to events and adjusts them per patch by event density, so that accuracy holds as refinement steps are added. Trained only on simulated data, Déjà-Events achieves the lowest pointmap error among the event-based methods we evaluate, on EventScape and zero-shot on the real MVSEC and DSEC datasets. On DSEC it halves the error of the strongest baseline (0.15 vs. 0.30 AbsRel) with about a tenth of its parameters. It also has the lowest depth error on EventScape and DSEC and the lowest trajectory error on EventScape.

**Comment:** Criterion 6: Déjà-Events jointly predicts depth, camera poses, and 3D pointmaps from multiple event-camera views in a single forward pass, using sparsity-aware adaptive gating and halving the strongest evaluated baseline’s DSEC error (0.15 versus 0.30 AbsRel).

**Relevance:** 9
Back to [[topic](#topic-6)] [[top](#topics)]

---
<a id="7004-lazy-mosaic-accelerating-skill-centric-manipulation-planning-with-learned-skill-effect-world-models"></a>

### 7004\. [Lazy-MOSAIC: Accelerating Skill-Centric Manipulation Planning with Learned Skill-Effect World Models](https://arxiv.org/abs/2611.21517)
**ArXiv:** 2611.21517 [[page](https://arxiv.org/abs/2611.21517)] [[pdf](https://arxiv.org/pdf/2611.21517.pdf)]

**Authors:** Itamar Mishani, Mukund Krishnakumar, Oren Salzman, Maxim Likhachev

**Abstract:** Long-horizon manipulation tasks require a robot to perform a sequence of physical interactions with its environment, typically combining several different skills. Planning such behaviors efficiently is challenging: given a library of parameterized skills, solutions lie in a vast hybrid space of skill sequences and their parameters, and evaluating each candidate skill's feasibility and effect with a high-fidelity world model such as a physics simulator is computationally expensive. MOSAIC, a recent skill-centric approach, lets the skills guide a non-sequential search toward the regions of this space where they are likely to succeed; however, it pays the full cost of simulation for every parameterized skill it considers. In this work, we introduce Lazy-MOSAIC, a lazy planning framework that instead searches over hypothesized skill outcomes. A cheap learned skill-effect world model predicts, in a fraction of a millisecond, a skill's feasibility and the world state it produces. Lazy-MOSAIC then uses these predictions to focus expensive simulation on promising candidate skills and, when a hypothesis is invalidated, repairs only the affected part of the search, while ensuring that every skill in a returned plan is validated in simulation. We evaluate Lazy-MOSAIC on cluttered-shelf retrieval tasks of increasing difficulty and demonstrate it on a physical robot, showing that it plans substantially faster than baselines with fewer simulator queries, while matching or exceeding their success rate in every setting. Visit skill-mosaic.github.io.

**Comment:** Criterion 7: Lazy-MOSAIC combines learned skill-effect world models, simulator feedback, and targeted search repair for robotic manipulation planning, reducing planning time and simulator queries on cluttered-shelf retrieval tasks and demonstrating the approach on a physical robot.

**Relevance:** 8
Back to [[topic](#topic-7)] [[top](#topics)]
