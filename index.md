---
title: Arxiv Daily
---

# Personalized Daily Arxiv Papers 11/02/2026

<time id="gptpa-last-update" datetime="2026-11-02T15:00:03.465055+00:00">Last update: 2026-11-02T15:00:03.465055+00:00</time>

This project is adapted from [tatsu-lab/gpt_paper_assistant](https://github.com/tatsu-lab/gpt_paper_assistant). The source code of this project is at [Variante/gpt_paper_assistant](https://github.com/Variante/gpt_paper_assistant)

About me on [Bilibili](https://space.bilibili.com/823532). Help keep the website running:

<a href="https://www.buymeacoffee.com/Variante"><img src="https://img.buymeacoffee.com/button-api/?text=Help cover GPT cost&emoji=💗&slug=Variante&button_colour=7f1f16&font_colour=ffffff&font_family=Arial&outline_colour=000000&coffee_colour=FFDD00" /></a>


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
1000\. [SketchMate: Agentic Geometric Reasoning for Zero-Shot Precision Insertion](https://arxiv.org/abs/2610.41844) [[more](#1000-sketchmate-agentic-geometric-reasoning-for-zero-shot-precision-insertion)]  
**Authors:** Jiaming Wang

1001\. [Nexus-VLA: Bridging Human and Robot Experiences via a Shared Hierarchical Semantic Latent Space](https://arxiv.org/abs/2610.42021) [[more](#1001-nexus-vla-bridging-human-and-robot-experiences-via-a-shared-hierarchical-semantic-latent-space)]  
**Authors:** Biaoliang Guan, Tingyu Yuan, Wen Ye, ..., Jihua Zhu, Chaoyang Zhao

1006\. [Hierarchical World Models with Implicit Dynamics](https://arxiv.org/abs/2610.42146) [[more](#1006-hierarchical-world-models-with-implicit-dynamics)]  
**Authors:** Gaoyue Zhou, Yvonne Wu, Zichen Jeff Cui, ..., Lerrel Pinto, Yann LeCun


Back to [[top](#topics)]

---
<a id="topic-2"></a>

### Topic 2
2005\. [Rethinking Asymmetry in Self-Supervised 3D Geometry Learning](https://arxiv.org/abs/2610.42092) [[more](#2005-rethinking-asymmetry-in-self-supervised-3d-geometry-learning)]  
**Authors:** Gyeongjin Kang, Freeman Cheng, Botao Ye, ..., Ming-Hsuan Yang, Eunbyung Park


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
7002\. [META-Robot: Self-Cognition and Capability Scaling in Agentic Robots](https://arxiv.org/abs/2610.41868) [[more](#7002-meta-robot-self-cognition-and-capability-scaling-in-agentic-robots)]  
**Authors:** Xuanzhi Feng, Zeyu Liu, Weiheng Gao, ..., Jie Zhang, Song Guo

7003\. [REA: A Routed Embodied Agent Runtime for Asynchronous Brain-Cerebellum Control](https://arxiv.org/abs/2610.42053) [[more](#7003-rea-a-routed-embodied-agent-runtime-for-asynchronous-brain-cerebellum-control)]  
**Authors:** Pengfei Zhou, Yixing Ma, Zhengfeiyang Zhang, ..., Wangbo Zhao, Yang You

7004\. [RoboArchitect: Automating the Design of Robotic Agents with Persistent Memory](https://arxiv.org/abs/2610.42141) [[more](#7004-roboarchitect-automating-the-design-of-robotic-agents-with-persistent-memory)]  
**Authors:** Shengguang Wu, Suning Huang, Cherie Ho, ..., Hao Zhu, Serena Yeung-Levy

7007\. [When the Simulator Fails, Not the Policy: Auditing Contact Artifacts in Simulated Robot Benchmarks](https://arxiv.org/abs/2610.41857) [[more](#7007-when-the-simulator-fails-not-the-policy-auditing-contact-artifacts-in-simulated-robot-benchmarks)]  
**Authors:** Alex Zhou, Junhao Wu, Xueyang Zhou, ..., Zhenghao Zhang, Yongchao Chen


Back to [[top](#topics)]

---
<a id="go-beyond"></a>

### Go beyond


Back to [[top](#topics)]

---
## Full paper list
 <a id="1000-sketchmate-agentic-geometric-reasoning-for-zero-shot-precision-insertion"></a>

### 1000\. [SketchMate: Agentic Geometric Reasoning for Zero-Shot Precision Insertion](https://arxiv.org/abs/2610.41844)
**ArXiv:** 2610.41844 [[page](https://arxiv.org/abs/2610.41844)] [[pdf](https://arxiv.org/pdf/2610.41844.pdf)]

**Authors:** Jiaming Wang

**Abstract:** Inserting an unseen part requires sub-millimetre precision and the correct orientation of keyed features. Current zero-shot agentic manipulation methods built on vision-language models (VLMs) produce spatial outputs that are far too coarse for this. We propose to let a VLM build a sketch model of the task, i.e., a few geometric primitives and two mate frames, which a VLM agent measures with geometric tools in multi-view depth and which a standard planner and compliant controller execute. Because a sketch describes the parts independently of their placement, the robot needs to build it only once per task. A sketch memory stores each agent run and transfers the stored sketch to later placements by point-cloud registration, so that the VLM is called only for unfamiliar parts. After grasping the part, the robot inspects both mating faces at close range and corrects the sketch by shape matching. On 69 simulated insertion episodes, our method inserts 88% of the parts at the assembled pose in a strict kinematic test, while ReKep and MOKA with the same VLM insert none. On new placements of eight tasks, reusing one stored run inserts as many parts as calling the agent in each episode, and it produces the sketch in 3.4 s instead of about 90 s.

**Comment:** Matches criteria 1 and 7 through a VLM agent that combines geometric tools, reusable sketch memory, and robotic control for zero-shot precision insertion, achieving 88% success across 69 simulated episodes versus 0% for ReKep and MOKA.

**Relevance:** 10
Back to [[topic](#topic-1)] [[top](#topics)]

<a id="1001-nexus-vla-bridging-human-and-robot-experiences-via-a-shared-hierarchical-semantic-latent-space"></a>

### 1001\. [Nexus-VLA: Bridging Human and Robot Experiences via a Shared Hierarchical Semantic Latent Space](https://arxiv.org/abs/2610.42021)
**ArXiv:** 2610.42021 [[page](https://arxiv.org/abs/2610.42021)] [[pdf](https://arxiv.org/pdf/2610.42021.pdf)]

**Authors:** Biaoliang Guan, Tingyu Yuan, Wen Ye, ..., Jihua Zhu, Chaoyang Zhao

**Abstract:** Large-scale egocentric human videos provide rich and scalable manipulation experience for robot policy learning, yet substantial embodiment gaps in visual appearance, kinematic structure, and action spaces hinder effective knowledge transfer. Existing methods often decouple cross-embodiment knowledge alignment from robot policy learning, making it difficult for human manipulation knowledge to directly and fully contribute to robot action learning. To address this issue, we propose Nexus-VLA, a hybrid policy learning framework that enables cross-embodiment knowledge transfer through end-to-end human-robot joint training, allowing human manipulation experience to directly contribute to robot policy learning. We first introduce a hierarchical cross-embodiment semantic alignment paradigm that progressively establishes shared manipulation semantics from task intent to fine-grained interactions, enabling more complete use of the manipulation knowledge contained in human videos. Building on this hierarchy, we design a shared manipulation latent representation that jointly encodes multi-level shared knowledge into an internal representation directly used for continuous action generation, improving policy learning efficiency while effectively leveraging human knowledge. Across RoboCasa-GR1, OOD settings, and real-world robot experiments, Nexus-VLA achieves an 80.5% task success rate on RoboCasa-GR1 and demonstrates strong advantages in human-data utilization, learning efficiency, and cross-scenario generalization.

**Comment:** Matches criterion 1: Nexus-VLA uses end-to-end human–robot joint training and a shared hierarchical semantic latent representation for continuous robot action generation, achieving 80.5% task success on RoboCasa-GR1.

**Relevance:** 10
Back to [[topic](#topic-1)] [[top](#topics)]

<a id="1006-hierarchical-world-models-with-implicit-dynamics"></a>

### 1006\. [Hierarchical World Models with Implicit Dynamics](https://arxiv.org/abs/2610.42146)
**ArXiv:** 2610.42146 [[page](https://arxiv.org/abs/2610.42146)] [[pdf](https://arxiv.org/pdf/2610.42146.pdf)]

**Authors:** Gaoyue Zhou, Yvonne Wu, Zichen Jeff Cui, ..., Lerrel Pinto, Yann LeCun

**Abstract:** Planning with world models enables test-time adaptation for embodied agents, offering superior generalization over direct policy learning. However, long-horizon tasks remain hindered by cumulative errors and search complexity. Existing hierarchical methods often use rigid macro-actions or latent skills, leading to brittle representations and limited fine-grained control. We introduce ID-WM (Implicit Dynamics World Models), a hierarchical world model that decouples state-space reachability from local control. Our framework utilizes a high-level model to sample feasible transitions across the state manifold and a low-level world model as an inverse dynamics solver to ground these transitions into precise actions. This allows the planner to identify reachable subgoals without being constrained by low-level execution details. By prioritizing local physical feasibility over global behavioral patterns, ID-WM synthesizes novel long-horizon plans from short segments, transcending the specific behavioral modes seen during training. We demonstrate its effectiveness across three navigation and manipulation suites, achieving a 38.5% improvement over state-of-the-art policy and diffusion-based planners. Crucially, by leveraging this hierarchical structure to mitigate compounding errors, our approach yields a 135% performance gain over single-level world model MPC, particularly in complex, unseen configurations for long-horizon tasks. Videos can be viewed at https://id-wm.github.io

**Comment:** Criterion 1: ID-WM uses a hierarchical world model to sample feasible transitions and ground them into actions through inverse dynamics for navigation and manipulation, reporting a 135% performance gain over single-level world-model MPC.

**Relevance:** 9
Back to [[topic](#topic-1)] [[top](#topics)]

---
<a id="2005-rethinking-asymmetry-in-self-supervised-3d-geometry-learning"></a>

### 2005\. [Rethinking Asymmetry in Self-Supervised 3D Geometry Learning](https://arxiv.org/abs/2610.42092)
**ArXiv:** 2610.42092 [[page](https://arxiv.org/abs/2610.42092)] [[pdf](https://arxiv.org/pdf/2610.42092.pdf)]

**Authors:** Gyeongjin Kang, Freeman Cheng, Botao Ye, ..., Ming-Hsuan Yang, Eunbyung Park

**Abstract:** Self-supervised learning relies on asymmetry between what a model observes and what it must explain. Recent methods learn 3D geometry from unposed images by predicting cameras and scene structure, then rendering held-out views. To avoid trivial solutions, they impose this asymmetry in the input space, restricting how views interact through separate branches or masked attention. We revisit this design with geometric accuracy, rather than rendering quality, as the main objective. When a held-out view lies between context views, the context already constrains the geometry it observes. We therefore impose the asymmetry in the 3D representation space, excluding held-out views only from the reconstructed scene while a shared backbone uses all views for correspondence. This improves depth and pose estimation over input-space designs. We further exploit view density: sparse and dense observations depict the same scene but provide different amounts of correspondence. Aligning sparse-context features with dense-context features supplies a geometric training signal alongside rendering, without a pretrained geometric teacher. We additionally align features with a frozen vision foundation model, which brings further gains. None of these signals requires geometric annotation, and the resulting model improves geometry estimation over prior self-supervised methods. Controlled experiments isolate the contribution of each component, and self-supervised pretraining also improves subsequent supervised training.

**Comment:** Criteria 2 and 6: The method introduces 3D-representation-space asymmetry and sparse-to-dense feature alignment for self-supervised geometry learning from unposed images, improving depth and camera-pose estimation without geometric annotations.

**Relevance:** 9
Back to [[topic](#topic-2)] [[top](#topics)]

---
<a id="7002-meta-robot-self-cognition-and-capability-scaling-in-agentic-robots"></a>

### 7002\. [META-Robot: Self-Cognition and Capability Scaling in Agentic Robots](https://arxiv.org/abs/2610.41868)
**ArXiv:** 2610.41868 [[page](https://arxiv.org/abs/2610.41868)] [[pdf](https://arxiv.org/pdf/2610.41868.pdf)]

**Authors:** Xuanzhi Feng, Zeyu Liu, Weiheng Gao, ..., Jie Zhang, Song Guo

**Abstract:** LLM-based robotic agent systems can generate programs for robot control, yet their generalization to new tasks often depends on elaborate scaffolding. Expanding their skill repertoires alone offers limited guidance on how to assess new capabilities or identify what remains beyond their reach. We introduce META-Robot, a framework that couples self cognition with capability scaling by expanding the criteria through which an agent evaluates and develops its skills. At its core, action-level predicates make execution outcomes explicit and available for LLM reasoning. The agent uses these assessments to identify its own limitations, extend relevant evaluation criteria, and guide the structured expansion of its skills. Proprioceptive feedback further grounds this process in the robot's physical state, enriching the evidence available for self-cognition. Experiments in simulation and on real robots demonstrate stronger overall task performance, generalization to new tasks, continued expansion of skills and evaluation criteria, and effective use of non-visual proprioceptive signals.

**Comment:** Criterion 7: META-Robot introduces an agentic framework that uses action-level predicates and proprioceptive feedback to assess execution, identify capability limitations, and expand skills, demonstrating improved task performance and generalization in simulation and on real robots.

**Relevance:** 10
Back to [[topic](#topic-7)] [[top](#topics)]

<a id="7003-rea-a-routed-embodied-agent-runtime-for-asynchronous-brain-cerebellum-control"></a>

### 7003\. [REA: A Routed Embodied Agent Runtime for Asynchronous Brain-Cerebellum Control](https://arxiv.org/abs/2610.42053)
**ArXiv:** 2610.42053 [[page](https://arxiv.org/abs/2610.42053)] [[pdf](https://arxiv.org/pdf/2610.42053.pdf)]

**Authors:** Pengfei Zhou, Yixing Ma, Zhengfeiyang Zhang, ..., Wangbo Zhao, Yang You

**Abstract:** Foundation models are becoming remarkably reliable at high-level reasoning, yet they remain too slow and costly to serve as fine-grained controllers. Conversely, fast policies like VLAs and whole-body controllers offer more efficient reflexes, but struggle to generalize. While pairing a deliberative brain (foundation models) with a reactive cerebellum (fast policies) seems natural for embodied systems, two obstacles remain: methods are built and scored on incompatible stacks, and their asynchronous coordination is difficult. For example, an agent that calls the VLA in the agentic loop can stall the fast execution, while a VLA that calls the agent on errors tends to re-trigger fallbacks repeatedly or completely miss silent failures. To address this, we propose REA (Routed Embodied Agent), a unified runtime where a deliberative System 2, an embodied router System 1 and reactive System 0 policies cooperate under a single golden rule - no running policy waits on the brain. Instead of querying the brain at each failure, REA employs a route memory that stores each policy's measured success ahead of time, a boundary-triggered judge gate that consults the brain on policy execution, and a shared-budget escalation ladder for bounded recovery. Evaluated on 3,000 test episodes across 60 LIBERO-Pro tasks, REA reaches 95.9% success and beats the best single-agent baseline in its own policy pool (94.7%) while reducing control steps by 25% and control time by 17%. In a separate 100-episode RoboTwin 2.0 study, REA achieves 84% success with VLA policies and 86% with VLA+code, compared with 80% for the strongest single VLA baseline MEDex. These results demonstrate learned and programmatic policies operating within the same runtime. REA also transfers to BEHAVIOR-1K and a real Unitree G1 humanoid, achieving the highest observed success among the compared methods.

**Comment:** Criterion 7: REA coordinates foundation-model reasoning and reactive VLA policies through route memory, asynchronous execution, and bounded recovery, achieving 95.9% success across 60 LIBERO-Pro robotic manipulation tasks.

**Relevance:** 10
Back to [[topic](#topic-7)] [[top](#topics)]

<a id="7004-roboarchitect-automating-the-design-of-robotic-agents-with-persistent-memory"></a>

### 7004\. [RoboArchitect: Automating the Design of Robotic Agents with Persistent Memory](https://arxiv.org/abs/2610.42141)
**ArXiv:** 2610.42141 [[page](https://arxiv.org/abs/2610.42141)] [[pdf](https://arxiv.org/pdf/2610.42141.pdf)]

**Authors:** Shengguang Wu, Suning Huang, Cherie Ho, ..., Hao Zhu, Serena Yeung-Levy

**Abstract:** Recent work shows that a frozen robot policy performs far better inside an agent harness, where control code decides when to query the policy, consult vision tools, or command the arm directly. Yet today these harnesses are built by hand and redesigned for each new robot or benchmark. We show that harness design can be automated, offering a strong alternative to finetuning on a novel setup. In RoboArchitect, a coding model observes a robot's rollouts, diagnoses failures, and iteratively revises the harness around the frozen policy. The harness evolves on a set of episodes for its optimization and is evaluated on held-out scenes and tasks. Without updating any model weights, RoboArchitect raises average success from 9.7% to 43.3% on unseen scenes (4.5x) and from 12.7% to 40.2% on unseen tasks (3.2x), across multiple benchmarks and policies. The gains are largest where the benchmark is far from the policy's training data: wrapping a general-purpose policy, the evolved harness yields a specialist that often outperforms even the finetuned version. Intuitive control strategies are automatically designed, such as lifting a jammed arm clear or reaching in horizontally when an obstacle blocks the way from above. The harness also evolves its own persistent memory, carrying learned experience, such as effective grasp depths, into later decisions. These results suggest that substantial headroom in robotic manipulation lies in the agent harness, and its design can be effectively automated and optimized without weight updates.

**Comment:** Criterion 7: RoboArchitect automatically improves robotic manipulation agent harnesses through rollout-based failure diagnosis, iterative code revision, and persistent memory, raising success on unseen scenes from 9.7% to 43.3% without updating model weights.

**Relevance:** 10
Back to [[topic](#topic-7)] [[top](#topics)]

<a id="7007-when-the-simulator-fails-not-the-policy-auditing-contact-artifacts-in-simulated-robot-benchmarks"></a>

### 7007\. [When the Simulator Fails, Not the Policy: Auditing Contact Artifacts in Simulated Robot Benchmarks](https://arxiv.org/abs/2610.41857)
**ArXiv:** 2610.41857 [[page](https://arxiv.org/abs/2610.41857)] [[pdf](https://arxiv.org/pdf/2610.41857.pdf)]

**Authors:** Alex Zhou, Junhao Wu, Xueyang Zhou, ..., Zhenghao Zhang, Yongchao Chen

**Abstract:** Simulation benchmarks score robot policies by success rate, and every failed episode counts against the policy. Many of these failures are the simulator's fault, not the policy's; we find the cause and fix it. Testing a vision language action model and a world action model in simulation, we find that the policy often places the object and the simulator then ejects it, and that the benchmark's own scripted demonstrations fail the same way. We present SimuGuard, an auditing plugin for LIBERO, RoboTwin, RoboCasa and RoboDojo. It leaves the evaluation unchanged. At every physics step, it records the state and flags an object that starts moving at a contact faster than the robot's contact or gravity account for. It replays a recorded episode bit for bit, and it reruns the episode with one simulation parameter changed. With it we trace these episodes on RoboTwin to the collision detection of the physics engine: for geometry one to two centimeters apart, the engine reports a penetration of about two centimeters, and the solver throws the object out at several meters per second. The cause is one constant: the distance algorithm discards triangles below a fixed area threshold, which valid triangles fall below at the centimeter scale. With that constant changed and nothing else, episodes with an artifact fall from 12 in 100 to none, and the success rate on five container tasks rises from 89% to 98%. Success rates in simulation thus count failures that the simulator causes. Benchmarks on PhysX 5.3.1 or earlier should rerun in a repaired engine. Where the engine cannot change, they should report the success rate together with the range it spans when every flagged episode fails or passes: 75.8% to 98.5% around a reported 83.1% for LingBot-VA.

**Comment:** Criterion 7: SimuGuard supports reliable robotic-agent evaluation by auditing contact artifacts in manipulation benchmarks, with a simulator fix raising success on five RoboTwin container tasks from 89% to 98%.

**Relevance:** 8
Back to [[topic](#topic-7)] [[top](#topics)]
