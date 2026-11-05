---
title: Arxiv Daily
---

# Personalized Daily Arxiv Papers 11/05/2026

<time id="gptpa-last-update" datetime="2026-11-05T15:00:02.772927+00:00">Last update: 2026-11-05T15:00:02.772927+00:00</time>

This project is adapted from [tatsu-lab/gpt_paper_assistant](https://github.com/tatsu-lab/gpt_paper_assistant). The source code of this project is at [Variante/gpt_paper_assistant](https://github.com/Variante/gpt_paper_assistant)

About me on [Bilibili](https://space.bilibili.com/823532). Help keep the website running:

<a href="https://www.buymeacoffee.com/Variante"><img src="https://img.buymeacoffee.com/button-api/?text=Help cover GPT cost&emoji=🐧&slug=Variante&button_colour=790980&font_colour=ffffff&font_family=Bree&outline_colour=000000&coffee_colour=FFDD00" /></a>


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
1004\. [From Pixels to Actions - Towards Native Vision-Language-Action Policies](https://arxiv.org/abs/2611.06918) [[more](#1004-from-pixels-to-actions-towards-native-vision-language-action-policies)]  
**Authors:** Weichen Fan, Haiwen Diao, Penghao Wu, ..., Dahua Lin, Ziwei Liu

1005\. [Open-HumanScale: An Open Recipe for Scaling Human and Robot Experience for Embodied Foundation Models](https://arxiv.org/abs/2611.06817) [[more](#1005-open-humanscale-an-open-recipe-for-scaling-human-and-robot-experience-for-embodied-foundation-models)]  
**Authors:** Juncheng Ma, Jiaxin Li, Jiankai Tu, ..., Tat-Seng Chua, Daquan Zhou

1006\. [Where Is VLA Inference Time Spent? Bottlenecks and Optimization Interactions Across GPUs](https://arxiv.org/abs/2611.06710) [[more](#1006-where-is-vla-inference-time-spent-bottlenecks-and-optimization-interactions-across-gpus)]  
**Authors:** Muhammad Danish Waseem, Ahmed Ali-Eldin Hassan


Back to [[top](#topics)]

---
<a id="topic-2"></a>

### Topic 2
2002\. [Motion-guided Contrastive Regularization For Latent World Models](https://arxiv.org/abs/2611.06901) [[more](#2002-motion-guided-contrastive-regularization-for-latent-world-models)]  
**Authors:** Yiwen Chen, Qiqi Hou, Jisoo Jeong, ..., Huaizu Jiang, Amir Said


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
6007\. [Trainspotting: Learning Object-Centric 3D Reconstruction from Found Photos](https://arxiv.org/abs/2611.06849) [[more](#6007-trainspotting-learning-object-centric-3d-reconstruction-from-found-photos)]  
**Authors:** Hanyu Chen, Noah Snavely


Back to [[top](#topics)]

---
<a id="topic-7"></a>

### Topic 7
7000\. [PyRUA-REAL: Robot Exploration and Adaptive Learning through RSI in the Real World](https://arxiv.org/abs/2611.06768) [[more](#7000-pyrua-real-robot-exploration-and-adaptive-learning-through-rsi-in-the-real-world)]  
**Authors:** Qiang Wang, Jianxin Bi, Shulong Jiang, ..., Xuanyi Li, Daquan Zhou

7001\. [VABAR: A Variational Automation Benchmark for Agentic Robotics](https://arxiv.org/abs/2611.06881) [[more](#7001-vabar-a-variational-automation-benchmark-for-agentic-robotics)]  
**Authors:** Shuangyu Xie, Kaiyuan Chen, Uksang Yoo, ..., Spencer Huang, Ken Goldberg

7003\. [Compounding Physical RSI: Turning Execution Failure into Inherited Capability](https://arxiv.org/abs/2611.06654) [[more](#7003-compounding-physical-rsi-turning-execution-failure-into-inherited-capability)]  
**Authors:** Yixuan Yang, Qiang Chen, Run Shao, ..., Yichao Cao, Xiu Su

7008\. [Rethinking Self-Repair in Code-as-Policy Agents: What Drives Retry Success?](https://arxiv.org/abs/2611.06863) [[more](#7008-rethinking-self-repair-in-code-as-policy-agents-what-drives-retry-success)]  
**Authors:** Siddharth Aphale, Ayushman Singh


Back to [[top](#topics)]

---
<a id="go-beyond"></a>

### Go beyond


Back to [[top](#topics)]

---
## Full paper list
 <a id="1004-from-pixels-to-actions-towards-native-vision-language-action-policies"></a>

### 1004\. [From Pixels to Actions - Towards Native Vision-Language-Action Policies](https://arxiv.org/abs/2611.06918)
**ArXiv:** 2611.06918 [[page](https://arxiv.org/abs/2611.06918)] [[pdf](https://arxiv.org/pdf/2611.06918.pdf)]

**Authors:** Weichen Fan, Haiwen Diao, Penghao Wu, ..., Dahua Lin, Ziwei Liu

**Abstract:** Vision-language-action models are typically built on a simple premise: perception is first compressed by visual encoders, and action reasoning begins only afterwards. Yet manipulation often hinges on precisely the visual signals most vulnerable to this abstraction, e.g., local geometry, boundaries, depth, and subtle state changes. This raises a fundamental question: should action policies inherit a fixed visual endpoint, or directly access evolving visual computation? We introduce NEO-VLA, a structurally encoder-free vision-language-action framework. Images are tokenized by lightweight patch convolutional layers and evolve jointly with language throughout the backbone, allowing a flow-matching action expert to draw from instruction-conditioned visual representations at different depths. This simple change reveals a consistent pattern. Native architectures outperform encoder-based counterparts under controlled comparisons; full fine-tuning reduces but does not erase the gap, and additional scale alone cannot recover it. More strikingly, layer-wise probing shows that the action-critical signals often peak well before the final visual representation. Across diverse manipulation settings and distribution shifts, NEO-VLA consequently achieves stronger and more robust control. Together, these results suggest that embodied control benefits from access to visual representations across the computational hierarchy, rather than a fixed encoder output.

**Comment:** Criterion 1: NEO-VLA combines an encoder-free vision-language-action backbone with a flow-matching action expert that accesses instruction-conditioned visual representations across layers, improving robotic manipulation control across diverse settings and distribution shifts.

**Relevance:** 10
Back to [[topic](#topic-1)] [[top](#topics)]

<a id="1005-open-humanscale-an-open-recipe-for-scaling-human-and-robot-experience-for-embodied-foundation-models"></a>

### 1005\. [Open-HumanScale: An Open Recipe for Scaling Human and Robot Experience for Embodied Foundation Models](https://arxiv.org/abs/2611.06817)
**ArXiv:** 2611.06817 [[page](https://arxiv.org/abs/2611.06817)] [[pdf](https://arxiv.org/pdf/2611.06817.pdf)]

**Authors:** Juncheng Ma, Jiaxin Li, Jiankai Tu, ..., Tat-Seng Chua, Daquan Zhou

**Abstract:** Egocentric human video offers a scalable source of manipulation priors for robot learning, yet translating raw data scale into generalizable robot policies demands rigorous data curation and training recipes. We introduce Open-HumanScale, the first fully open recipe for scaling egocentric pretraining to 50,000 hours, anchored by Open-HumanScale-50K---a curated egocentric video corpus densely annotated with hand motion and natural language. Leveraging this corpus, we systematically investigate pure human pretraining via closed-loop simulation evaluations and uncover an empirical log-linear scaling law linking pretraining data hours to downstream robot success rates. Grounded in these findings, we develop HumanWAM-Vanilla, a world--action model pretrained exclusively on 50k hours of egocentric human video. Upon robot post-training, HumanWAM-Vanilla reaches 92.08% success on RoboTwin 2.0 Clean2Clean and 68.88% on Clean2Random, alongside a 14.17% mean success rate and a 19.00 mean progress score on RoboDojo. To push the capability frontier further, we explore joint human--robot pretraining with HumanWAM-Goal, a visual-goal-conditioned world--action model. Hybrid data scaling unlocks broad zero-shot generalization across both simulation and physical benchmarks, propelling post-trained RoboTwin 2.0 performance to new state-of-the-art levels: 92.70% on Clean2Clean and a substantial leap to 84.60% on Clean2Random. On physical robots, HumanWAM-Goal further exhibits emergent embodied in-context learning, dynamically adapting its manipulation behavior to unseen visual goals without parameter updates. To support reproducible scaling research in embodied AI, we open-source the Open-HumanScale-50K corpus, data curation and pretraining pipelines, alongside the pretrained and post-trained HumanWAM checkpoints, inference code, and simulation evaluation suites.

**Comment:** Criterion 1: HumanWAM-Goal uses joint human–robot pretraining for a visual-goal-conditioned world–action model, achieving 84.60% success on RoboTwin 2.0 Clean2Random and adapting to unseen manipulation goals on physical robots without parameter updates.

**Relevance:** 10
Back to [[topic](#topic-1)] [[top](#topics)]

<a id="1006-where-is-vla-inference-time-spent-bottlenecks-and-optimization-interactions-across-gpus"></a>

### 1006\. [Where Is VLA Inference Time Spent? Bottlenecks and Optimization Interactions Across GPUs](https://arxiv.org/abs/2611.06710)
**ArXiv:** 2611.06710 [[page](https://arxiv.org/abs/2611.06710)] [[pdf](https://arxiv.org/pdf/2611.06710.pdf)]

**Authors:** Muhammad Danish Waseem, Ahmed Ali-Eldin Hassan

**Abstract:** Vision-language-action (VLA) models map camera observations and language instructions to robot actions. However, their large backbones and repeated action generation can make inference take longer than the robot's real-time requirements for its next action. To reduce the load of VLA computations, researchers have proposed multiple optimization approaches, including quantization and pruning. These optimizations can be combined with other general optimizations such as better kernel compilation or using CUDA graphs instead of the eager mode. However, most of these optimization results are usually reported on a single GPU type, typically for a few VLA models, with no analysis of what the real bottlenecks are for a model on a particular hardware or how these optimizations would generalize beyond the reported results. In this paper, we provide a comprehensive measurement study of seven families of inference optimization methods, from CUDA graphs and compilation to token pruning and quantization, on seven state-of-the-art VLAs evaluated on four GPU families ranging from a datacenter H100 to an embedded Jetson Thor. We analyse how CPU launch overhead, memory bandwidth, and kernel efficiency jointly shape optimization gains, and how applying one method changes the benefit of another. From this analysis, we derive VLA-Triage, guidelines for stacking optimizations under latency budgets. The guidelines identify which stages can provide the required savings and which combinations need testing in the deployed execution mode.

**Comment:** Criterion 1: benchmarks seven vision-language-action robot policies across four GPU families and derives VLA-Triage guidelines for meeting real-time, language-conditioned robot-action inference budgets.

**Relevance:** 8
Back to [[topic](#topic-1)] [[top](#topics)]

---
<a id="2002-motion-guided-contrastive-regularization-for-latent-world-models"></a>

### 2002\. [Motion-guided Contrastive Regularization For Latent World Models](https://arxiv.org/abs/2611.06901)
**ArXiv:** 2611.06901 [[page](https://arxiv.org/abs/2611.06901)] [[pdf](https://arxiv.org/pdf/2611.06901.pdf)]

**Authors:** Yiwen Chen, Qiqi Hou, Jisoo Jeong, ..., Huaizu Jiang, Amir Said

**Abstract:** JEPA-style world models learn action-conditioned latent dynamics without reconstructing observations, but distributional regularization alone does not ensure that their representations preserve information needed for control. We identify static collapse, a failure mode in which the encoder preserves episode-specific static content while suppressing manipulation-relevant state changes. Because static content is temporally predictable yet diverse across episodes, this failure can persist despite regularizers such as SIGReg. A controlled study on OGBench-Cube shows that introducing randomized backgrounds substantially degrades planning performance while leaving the underlying manipulation task unchanged. To address this problem, we propose motion-guided contrastive regularization. Our approach compares predicted latent transitions with transitions to candidate observations from the same episode, reducing the usefulness of static background identity for distinguishing action outcomes. Optical-flow similarities provide soft contrastive targets, allowing candidates with similar motion to receive similar supervision. Combined with latent prediction and SIGReg, our objective encourages representations to preserve within-episode state distinctions and similarities between related transitions. Our method improves planning success over LeWorldModel (LeWM) from 52% to 78% on OGBench-Cube with randomized backgrounds and from 14% to 54% on LIBERO-Object, while maintaining comparable performance on four other visual-control benchmarks.

**Comment:** Criterion 2: Introduces motion-guided contrastive regularization for JEPA-style latent world models, using optical-flow similarities as soft targets to prevent static representation collapse and improve planning success from 14% to 54% on LIBERO-Object.

**Relevance:** 10
Back to [[topic](#topic-2)] [[top](#topics)]

---
<a id="6007-trainspotting-learning-object-centric-3d-reconstruction-from-found-photos"></a>

### 6007\. [Trainspotting: Learning Object-Centric 3D Reconstruction from Found Photos](https://arxiv.org/abs/2611.06849)
**ArXiv:** 2611.06849 [[page](https://arxiv.org/abs/2611.06849)] [[pdf](https://arxiv.org/pdf/2611.06849.pdf)]

**Authors:** Hanyu Chen, Noah Snavely

**Abstract:** State-of-the-art feedforward reconstruction models perform poorly on in-the-wild object-centric photo collections. They are trained largely on single-session object captures whose images vary in viewpoint while the object and its environment remain nearly fixed. Public photo archives contain a different form of multi-view data. They hold found photos of the same object or design: photos taken independently by many people across years, environments, cameras, and sometimes physical units. In this work, we introduce Trainspotting, a dataset of 12,068 objects and 311,886 photos from Wikimedia Commons. We reconstruct each object to recover camera poses, multi-view stereo depth, and object masks. We categorize the objects into three regimes: objects fixed at one site, objects that move between sites, and designs shared by many physical units. Evaluating recent feedforward reconstruction models on Trainspotting shows that they fail most severely in the last two regimes. Fine-tuning VGGT-$Ω$ on Trainspotting significantly improves camera pose, depth, and 3D geometry on held-out objects. We observe similar improvements on the independent NAVI in-the-wild benchmark, which contains photos of each object in different environments. Performance on conventional benchmarks remains largely unchanged.

**Comment:** Criterion 6: Trainspotting advances object-centric multi-view reconstruction from independently captured photos, with VGGT-Ω fine-tuning improving camera poses, depth, and 3D geometry on held-out objects and the independent NAVI benchmark.

**Relevance:** 8
Back to [[topic](#topic-6)] [[top](#topics)]

---
<a id="7000-pyrua-real-robot-exploration-and-adaptive-learning-through-rsi-in-the-real-world"></a>

### 7000\. [PyRUA-REAL: Robot Exploration and Adaptive Learning through RSI in the Real World](https://arxiv.org/abs/2611.06768)
**ArXiv:** 2611.06768 [[page](https://arxiv.org/abs/2611.06768)] [[pdf](https://arxiv.org/pdf/2611.06768.pdf)]

**Authors:** Qiang Wang, Jianxin Bi, Shulong Jiang, ..., Xuanyi Li, Daquan Zhou

**Abstract:** Coding agents powered by models such as GPT-6-Astra connect spatial reasoning with executable robot behavior, raising the question of how such agents can produce lasting improvements in robot capabilities from physical interaction. In this work, we study embodied recursive self-improvement (RSI) through real-world exploration, code refinement, and deployment using PyRUA-REAL, a framework in which coding agents retain interaction experience as persistent memory and reusable executable skills that inform subsequent exploration and improvement. Our empirical study places physical manipulation at the center of the improvement process, examining how agents collaborate, what kinds of execution feedback support effective code refinement, and how task specifications and previously acquired skills influence robustness and task transfer. We further investigate the efficiency and capability of the resulting programs when deployed without involving an online agent for decision-making. Across eight real-world tasks, PyRUA-REAL achieves an average success rate of 91.1%. In complementary evaluation on 42 tasks from the RoboDojo benchmark, it achieves an average success rate of 84.4%. This study provides empirical guidance for turning physical exploration into reusable robot skills that support continued improvement and robust, efficient deployment.

**Comment:** Criterion 7: PyRUA-REAL combines physical manipulation exploration, execution-feedback-driven code refinement, and persistent memory of reusable robot skills, achieving 91.1% success across eight real-world tasks and 84.4% across 42 RoboDojo tasks.

**Relevance:** 10
Back to [[topic](#topic-7)] [[top](#topics)]

<a id="7001-vabar-a-variational-automation-benchmark-for-agentic-robotics"></a>

### 7001\. [VABAR: A Variational Automation Benchmark for Agentic Robotics](https://arxiv.org/abs/2611.06881)
**ArXiv:** 2611.06881 [[page](https://arxiv.org/abs/2611.06881)] [[pdf](https://arxiv.org/pdf/2611.06881.pdf)]

**Authors:** Shuangyu Xie, Kaiyuan Chen, Uksang Yoo, ..., Spencer Huang, Ken Goldberg

**Abstract:** For robots to operate reliably in commercial and industrial applications, they must persistently perform specific tasks with known variations in object shape and pose. We call this Variational Automation (VA) to differentiate these settings from fixed automation. Inspired by recent papers on Agentic Robotics, where pretrained LLM and VLM coding agents combine interpretable robot programming with the open-world adaptability of model free policies, we propose a general set of agentic robotics benchmark design criteria that includes accessibility, repeatability, diversity, relevance to agentic robotics and practical applications, physical realizability, resistance to benchmark saturation, and computational efficiency. We instantiate these principles in six manipulation task classes spanning industrial, medical, and infrastructure applications: tool sorting, tool hanging, gear meshing, syringe packing, cable insertion, and cable routing. We introduce the Variational Automation Benchmark for Agentic Robotics (VABAR), an open, easily replicable, simulation-based manipulation benchmark to systematically evaluating agentic robotics systems. Built fully on open-source tools, VABAR features detailed implementations using the Newton physics engine, standardized task variations, difficulty tiers, success metrics, and a library of 33 generalizable skills with human-authored programmatic reference solutions. We provide initial baseline evaluations using an open agentic manipulation harness, Graph-as-Policy (GaP), and Isaac Lab Arena.

**Comment:** Criterion 7: VABAR introduces an open simulation-based benchmark for agentic robotics with six manipulation task classes, 33 generalizable skills, and baseline evaluations using Graph-as-Policy and Isaac Lab Arena.

**Relevance:** 10
Back to [[topic](#topic-7)] [[top](#topics)]

<a id="7003-compounding-physical-rsi-turning-execution-failure-into-inherited-capability"></a>

### 7003\. [Compounding Physical RSI: Turning Execution Failure into Inherited Capability](https://arxiv.org/abs/2611.06654)
**ArXiv:** 2611.06654 [[page](https://arxiv.org/abs/2611.06654)] [[pdf](https://arxiv.org/pdf/2611.06654.pdf)]

**Authors:** Yixuan Yang, Qiang Chen, Run Shao, ..., Yichao Cao, Xiu Su

**Abstract:** Generalist robot policies are trained on static offline datasets, yet physical manipulation unfolds sequentially across stages. Because a weakness in a downstream stage remains unobservable until the policy passes earlier ones, each resolved bottleneck exposes the next hidden failure mode. This staged structure calls for Recursive Self-Improvement (RSI), in which a policy iteratively executes, diagnoses failures, and updates itself. Moreover, because task completion behaves like a product of stage-level pass rates, recursive gains can multiply across generations. In physical environments, three bottlenecks obstruct compounding: terminal execution feedback rarely localizes the stage that caused a failure, failed trajectories do not supply the corrective action, and updating shared parameters can degrade previously mastered skills. We propose Compounding Physical RSI, a self-evolving policy framework that turns execution failures into heritable capability. First, automated capability-gap assessment traces failed rollouts to the task stage where progression breaks down and formulates a structured improvement objective. Second, conditioned on this objective, our remedy stage curates validated, stage-relevant experience and refines the policy via behavior-regularized offline reinforcement learning, without requiring costly online rollouts at each optimization step. Third, to safeguard existing skills, our retention stage constructs weight-space candidates between the inherited reference policy and the refined policy, evaluates them across the task suite, and promotes the best candidate as the reference state for the next generation. Evaluated across all 42 tasks of the RoboDojo-Sim benchmark, CSU-AI-07 ranks first on the leaderboard with a three-seed mean Overall Score of 26.15 and Success Rate of 20.70%, outperforming the next-best entry by 5.92 points and 5.82 percentage points.

**Comment:** Criterion 7: introduces a robotic self-improvement framework combining automated failure diagnosis, behavior-regularized offline reinforcement learning, and skill-preserving policy selection, ranking first across all 42 RoboDojo-Sim tasks.

**Relevance:** 10
Back to [[topic](#topic-7)] [[top](#topics)]

<a id="7008-rethinking-self-repair-in-code-as-policy-agents-what-drives-retry-success"></a>

### 7008\. [Rethinking Self-Repair in Code-as-Policy Agents: What Drives Retry Success?](https://arxiv.org/abs/2611.06863)
**ArXiv:** 2611.06863 [[page](https://arxiv.org/abs/2611.06863)] [[pdf](https://arxiv.org/pdf/2611.06863.pdf)]

**Authors:** Siddharth Aphale, Ayushman Singh

**Abstract:** Code-as-policy agents often use execution feedback to retry failed programs. Additional sampling and task guidance can also improve success, leaving the contribution of failure-specific feedback unclear. We study this question in simulated robot manipulation. A fixed lookup selects feedback templates using failure labels and the API variant. A frozen code model generates a fresh program after each scene reset. On the same 252 initial failures, Qwen2.5-Coder-7B recovers 85.3% within five retries, compared with 68.3% for resampling with the same retry cap. The advantage persists under fixed total-token budgets. Feedback from other episodes and fixed task guidance yield similar recovery point estimates, with no detected benefit from matching the coarse failure label. In a separate study, a fixed guided prompt used from the first attempt achieves an observed overall success rate similar to that of template-based feedback. Template-based feedback also lowers recovery on a nut-insertion task requiring a rim grasp, consistent with an incompatible center-grasp instruction. Evaluations of robot self-repair should include resampling, mismatched feedback, and concise, task-compatible static guidance as baselines.

**Comment:** Criterion 7: This empirical study evaluates a feedback-driven code-as-policy retry framework for simulated robot manipulation, reporting 85.3% recovery within five retries versus 68.3% for resampling and testing whether failure-specific feedback actually drives the improvement.

**Relevance:** 7
Back to [[topic](#topic-7)] [[top](#topics)]
