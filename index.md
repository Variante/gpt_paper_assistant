---
title: Arxiv Daily
---

# Personalized Daily Arxiv Papers 11/03/2026

<time id="gptpa-last-update" datetime="2026-11-03T15:00:03.279712+00:00">Last update: 2026-11-03T15:00:03.279712+00:00</time>

This project is adapted from [tatsu-lab/gpt_paper_assistant](https://github.com/tatsu-lab/gpt_paper_assistant). The source code of this project is at [Variante/gpt_paper_assistant](https://github.com/Variante/gpt_paper_assistant)

About me on [Bilibili](https://space.bilibili.com/823532). Help keep the website running:

<a href="https://www.buymeacoffee.com/Variante"><img src="https://img.buymeacoffee.com/button-api/?text=Help cover GPT cost&emoji=🎈&slug=Variante&button_colour=7a4460&font_colour=ffffff&font_family=Cookie&outline_colour=000000&coffee_colour=FFDD00" /></a>


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
1000\. [VerifLM: Physically Verified Image-Edit-Guided Goal Pose Generation for Robotic Manipulation](https://arxiv.org/abs/2611.03268) [[more](#1000-veriflm-physically-verified-image-edit-guided-goal-pose-generation-for-robotic-manipulation)]  
**Authors:** Xuanyou Chen, Jinyu Chen, Glen Chou

1001\. [TAMPAgent: Agentic Belief-Space Planning for Mobile Manipulation](https://arxiv.org/abs/2611.03584) [[more](#1001-tampagent-agentic-belief-space-planning-for-mobile-manipulation)]  
**Authors:** Utkarsh A. Mishra, Alperen Degirmenci, Yongxin Chen, ..., Tucker Hermans, Caelan Garrett

1002\. [Seeing Through the Illusion: Test-Time Scaling for World Action Models](https://arxiv.org/abs/2611.03585) [[more](#1002-seeing-through-the-illusion-test-time-scaling-for-world-action-models)]  
**Authors:** Cheikh Ahmed, Mahdi Mostajabdaveh, Milad Yazdani, Zirui Zhou

1003\. [D&A: Decoupling Description and Assessment Enables VLMs as Zero-Shot Robotic Reward Models](https://arxiv.org/abs/2611.03622) [[more](#1003-da-decoupling-description-and-assessment-enables-vlms-as-zero-shot-robotic-reward-models)]  
**Authors:** Marius Memmel, Shawn Yin, Urvi Rutia, ..., Anqi Li, Abhishek Gupta


Back to [[top](#topics)]

---
<a id="topic-2"></a>

### Topic 2


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
5004\. [Intent2CAD: How Semantic-Parametric Supervision Shapes Text-to-CAD Generation](https://arxiv.org/abs/2611.03449) [[more](#5004-intent2cad-how-semantic-parametric-supervision-shapes-text-to-cad-generation)]  
**Authors:** Jinbo Yang, Mingyue Yuan, Shikai Jing

5005\. [CoLa3D: Composable Latent 3D Decomposition](https://arxiv.org/abs/2611.03533) [[more](#5005-cola3d-composable-latent-3d-decomposition)]  
**Authors:** Xudong Cai, Yongcai Wang, Deying Li, Chuanxia Zheng


Back to [[top](#topics)]

---
<a id="topic-6"></a>

### Topic 6


Back to [[top](#topics)]

---
<a id="topic-7"></a>

### Topic 7
7006\. [Pixel: A Co-Designed Full-Stack Robot System for Learning General-Purpose Mobile Manipulation](https://arxiv.org/abs/2611.03648) [[more](#7006-pixel-a-co-designed-full-stack-robot-system-for-learning-general-purpose-mobile-manipulation)]  
**Authors:** Tony Tao, Jason Jingzhou Liu, Ruslan Salakhutdinov, Deepak Pathak


Back to [[top](#topics)]

---
<a id="go-beyond"></a>

### Go beyond


Back to [[top](#topics)]

---
## Full paper list
 <a id="1000-veriflm-physically-verified-image-edit-guided-goal-pose-generation-for-robotic-manipulation"></a>

### 1000\. [VerifLM: Physically Verified Image-Edit-Guided Goal Pose Generation for Robotic Manipulation](https://arxiv.org/abs/2611.03268)
**ArXiv:** 2611.03268 [[page](https://arxiv.org/abs/2611.03268)] [[pdf](https://arxiv.org/pdf/2611.03268.pdf)]

**Authors:** Xuanyou Chen, Jinyu Chen, Glen Chou

**Abstract:** Many manipulation goals are easier to visualize than to specify numerically through explicit geometric constraints. We present VerifLM, a framework that converts image-edited goal configurations into physically verified 6-DoF object poses for planning. Using an edited image, VerifLM recovers an initial relative pose between an active object and a passive object, then prompts a vision-language model to generate axis-wise sampling constraints and a task-specific validation program. Candidate poses are evaluated in a physics simulator, and the VLM selects the most task-consistent retained candidate. To operate without predefined object meshes or physical parameters, VerifLM reconstructs metric object geometry from RGB-D observations and constructs multiple hypothetical simulations with varying physics and geometry. Parallel-world validation evaluates candidates across geometry--physics variations and ranks them by the number of worlds in which they satisfy the validation program. We evaluate VerifLM on nine manipulation tasks. VerifLM achieves the highest task success on eight of nine tasks, with the largest performance gain on goals requiring precise orientation and contact relationships.

**Comment:** Matches criterion 1 by using a VLM to generate pose-sampling constraints and physics-validation programs for robotic manipulation planning, achieving the highest task success on eight of nine tasks.

**Relevance:** 10
Back to [[topic](#topic-1)] [[top](#topics)]

<a id="1001-tampagent-agentic-belief-space-planning-for-mobile-manipulation"></a>

### 1001\. [TAMPAgent: Agentic Belief-Space Planning for Mobile Manipulation](https://arxiv.org/abs/2611.03584)
**ArXiv:** 2611.03584 [[page](https://arxiv.org/abs/2611.03584)] [[pdf](https://arxiv.org/pdf/2611.03584.pdf)]

**Authors:** Utkarsh A. Mishra, Alperen Degirmenci, Yongxin Chen, ..., Tucker Hermans, Caelan Garrett

**Abstract:** Open-world robotic mobile manipulation requires acting in physical environments with only partial information. While recent Large Vision-Language Model (LVLM) agents show strong open-world reasoning capabilities, existing architectures that rely on direct tool calling struggle with geometric uncertainty and thus are limited to fully-observable tabletop environments. In contrast, Task and Motion Planning (TAMP) systems can solve long-horizon tasks under partial observability but incur high computational costs and require closed worlds. We propose \textbf{TAMPAgent}, an agentic planning framework that unifies the reasoning of LVLMs with the structured constraint satisfaction of TAMP systems. We endow an LVLM agent with a belief-space planning model operating over an extensible API of perception, planning, and control tools. The agent formulates a partial task plan composed of parameterized observation and execution actions, which is subsequently completed using classical search algorithms. After extracting the plan's geometric constraints, the agent leverages its tool API to bind valid parameter values, satisfying feasibility requirements and producing an executable motion plan. Upon execution, the agent continuously updates its belief history and repeats this closed-loop process by interleaving observation and manipulation actions. Evaluated across complex household mobile manipulation tasks on Rainbow Y1 and Unitree G1 platforms, TAMPAgent significantly outperforms direct tool-calling agentic baselines, demonstrating that integrating TAMP constraint satisfaction into agentic planning loops is essential for unstructured, partially observable environments.

**Comment:** Criteria 1 and 7: TAMPAgent integrates LVLM reasoning with belief-space task-and-motion planning and perception, planning, and control tools in a closed manipulation loop, outperforming direct tool-calling agents on household tasks using Rainbow Y1 and Unitree G1 platforms.

**Relevance:** 10
Back to [[topic](#topic-1)] [[top](#topics)]

<a id="1002-seeing-through-the-illusion-test-time-scaling-for-world-action-models"></a>

### 1002\. [Seeing Through the Illusion: Test-Time Scaling for World Action Models](https://arxiv.org/abs/2611.03585)
**ArXiv:** 2611.03585 [[page](https://arxiv.org/abs/2611.03585)] [[pdf](https://arxiv.org/pdf/2611.03585.pdf)]

**Authors:** Cheikh Ahmed, Mahdi Mostajabdaveh, Milad Yazdani, Zirui Zhou

**Abstract:** World action models (WAMs) jointly generate actions and imagined futures, but a plausible prediction need not be realized by its co-generated action. We introduce WISE (\emph{World Action Inference Scaling through Evaluation}), a pre-execution selector that scores each sampled rollout on action validity and temporal regularity, inverse-dynamics consistency between the proposed action and the imagined motion, and instruction-conditioned task progress, and executes the highest-scoring candidate while keeping the WAM frozen. With four candidates per decision, WISE raises closed-loop success from 6.0\% to 24.0\% on RoboLab with Cosmos~3 Edge and from 75.1\% to 78.3\% mean success on LIBERO with DreamZero, and exceeds the best existing selector in aggregate by 4.0 and 0.9 points. Component ablations and agreement analyses show that the three scores rank candidates differently and that task progress carries the largest weight in the fused decision.

**Comment:** Criterion 1: WISE improves frozen world action models for robotic control by ranking sampled action–future rollouts using action validity, inverse-dynamics consistency, and task progress, increasing RoboLab closed-loop success from 6.0% to 24.0% and LIBERO mean success from 75.1% to 78.3%.

**Relevance:** 10
Back to [[topic](#topic-1)] [[top](#topics)]

<a id="1003-da-decoupling-description-and-assessment-enables-vlms-as-zero-shot-robotic-reward-models"></a>

### 1003\. [D&A: Decoupling Description and Assessment Enables VLMs as Zero-Shot Robotic Reward Models](https://arxiv.org/abs/2611.03622)
**ArXiv:** 2611.03622 [[page](https://arxiv.org/abs/2611.03622)] [[pdf](https://arxiv.org/pdf/2611.03622.pdf)]

**Authors:** Marius Memmel, Shawn Yin, Urvi Rutia, ..., Anqi Li, Abhishek Gupta

**Abstract:** Vision-Language Models (VLMs) hold promise for zero-shot video task-progress estimation in robot learning, yet existing frameworks frequently suffer from confirmation bias, hallucinating progress in failed executions due to implicit success assumptions or relative temporal ranking designs. Naive prompting of frontier models collapses into a binary success detector that fails to capture nuanced, non-linear progress and failures. While post-training approaches can mitigate this through explicit training, they demand expensive real-world robotics data and require computationally intensive post-training for every individual base model. To resolve these limitations, we introduce D&A (Describe & Assess): our approach decouples progress estimation into an objective, task-grounded textual description of scene dynamics followed by an assessment phase using explicit rubrics. By compelling the VLM to synthesize an explicit, text-based definition of absolute success before scoring, the framework anchors evaluations to an external, global criterion rather than the video's internal timeline. Empirical results demonstrate that our method calibrates zero-shot estimates and mitigates confirmation bias, providing fine-grained, partial progress metrics that accurately penalize failed executions while correctly interpolating successful runs. Being purely prompt-based, D&A generalizes across tasks, embodiments, and benchmarks---matching or surpassing post-trained reward models---and its reward estimates enable efficient downstream reinforcement learning.

**Comment:** Criterion 1: D&A decouples task-grounded scene description from rubric-based assessment to produce zero-shot VLM reward estimates that mitigate confirmation bias, match or surpass post-trained reward models, and enable downstream robotic reinforcement learning.

**Relevance:** 10
Back to [[topic](#topic-1)] [[top](#topics)]

---
<a id="5004-intent2cad-how-semantic-parametric-supervision-shapes-text-to-cad-generation"></a>

### 5004\. [Intent2CAD: How Semantic-Parametric Supervision Shapes Text-to-CAD Generation](https://arxiv.org/abs/2611.03449)
**ArXiv:** 2611.03449 [[page](https://arxiv.org/abs/2611.03449)] [[pdf](https://arxiv.org/pdf/2611.03449.pdf)]

**Authors:** Jinbo Yang, Mingyue Yuan, Shikai Jing

**Abstract:** Text-to-CAD generation aims to make CAD modeling more accessible by generating executable CAD programs from natural-language descriptions. However, practical CAD workflows require more than executable geometry: generated programs should also expose reusable structure that supports later modification. We introduce intent-oriented text-to-CAD generation, where design intent is expressed as recoverable code-level signals: semantic features, reusable parameters, and lightweight parametric relations. Rather than changing the model architecture or decoding procedure, we investigate how supervision representation granularity affects both CAD generation and zero-shot intent-preserving editing. To enable this study, we construct IntentCAD-100K, a 100K paired text--code dataset built by converting low-level CAD construction traces into executable CadQuery programs. We further construct IntentCAD-Edit-1K to evaluate zero-shot intent-preserving editing from the same representation perspective. Intent2CAD uses a two-stage conservative lifting pipeline: semantic lifting rewrites reliable primitive groups into feature-level operations, while parametric lifting exposes lightweight parametric relations. Under controlled fine-tuning, semantic lifting mainly improves generation reliability and feature recovery, reducing IR from 2.39% to 0.22% and achieving 99.96% API Recovery, while parametric lifting makes reusable relation structure explicit and further improves relation-aware zero-shot editing. Datasets and code are available at https://github.com/JinboYang94/Intent2CAD.

**Comment:** Criterion 5: Intent2CAD improves text-to-3D CAD program generation through semantic and parametric supervision, with semantic lifting reducing the reported IR from 2.39% to 0.22% and achieving 99.96% API Recovery.

**Relevance:** 9
Back to [[topic](#topic-5)] [[top](#topics)]

<a id="5005-cola3d-composable-latent-3d-decomposition"></a>

### 5005\. [CoLa3D: Composable Latent 3D Decomposition](https://arxiv.org/abs/2611.03533)
**ArXiv:** 2611.03533 [[page](https://arxiv.org/abs/2611.03533)] [[pdf](https://arxiv.org/pdf/2611.03533.pdf)]

**Authors:** Xudong Cai, Yongcai Wang, Deying Li, Chuanxia Zheng

**Abstract:** We introduce CoLa3D, a simple and effective model for composable 3D scene decomposition that, given a single whole-scene latent from either a single image or an existing scene mesh as input, directly outputs a set of object meshes with consistent spatial layout that can be assembled back into a coherent scene. CoLa3D introduces a novel paradigm for 3D scene decomposition within a shared whole-scene latent, unlike alternatives that perform per-object generation in canonical spaces and then glue the objects back together with fragile pose and scale estimation. The network is a lightweight, flexible, and scalable query-based transformer that accepts various types of queries, including 2D masks, 3D points, and learnable queries, and learns to decompose the scene into objects in a compact latent space. On both synthetic and real-world datasets, CoLa3D achieves state-of-the-art performance on composable 3D scene decomposition, substantially improving global scene-level consistency and object-level quality while running 2 times faster and using less than 1% of SAM 3D's training data. Visible depth on par with MoGe2 further confirms that latent decomposition preserves accurate visible scene geometry while producing physically complete, decomposable object meshes.

**Comment:** Criterion 5: CoLa3D uses a query-based transformer to generate spatially consistent, physically complete object meshes from a shared single-image scene latent, achieving state-of-the-art scene decomposition while running twice as fast and using less than 1% of SAM 3D's training data.

**Relevance:** 9
Back to [[topic](#topic-5)] [[top](#topics)]

---
<a id="7006-pixel-a-co-designed-full-stack-robot-system-for-learning-general-purpose-mobile-manipulation"></a>

### 7006\. [Pixel: A Co-Designed Full-Stack Robot System for Learning General-Purpose Mobile Manipulation](https://arxiv.org/abs/2611.03648)
**ArXiv:** 2611.03648 [[page](https://arxiv.org/abs/2611.03648)] [[pdf](https://arxiv.org/pdf/2611.03648.pdf)]

**Authors:** Tony Tao, Jason Jingzhou Liu, Ruslan Salakhutdinov, Deepak Pathak

**Abstract:** General-purpose mobile manipulation requires a combination of capable robot hardware, scalable demonstration collection, expressive whole-body control, and effective policy learning. Yet these components are often developed independently, creating mismatches that limit the capabilities of the complete system. We present Pixel, a co-designed full-stack system for learning general-purpose mobile manipulation. Pixel combines a low-cost wheeled humanoid with human-aligned kinematics and compliant grippers, multiple teleoperation and robot-free human data-collection interfaces, a unified whole-body controller, and a common policy learning pipeline across human and robot demonstrations. A hierarchical learning stack further enables language-conditioned visuomotor control and long-horizon task execution. We evaluate the system across hardware, data collection, and policy learning, and demonstrate mobile-manipulation behaviors requiring dexterity, active perception, navigation, and whole-body coordination. We hope that Pixel provides an integrated and extensible foundation for studying how robot hardware, data, control, and learning can be designed together for robot learning. More results at https://pixel-robot.com/pixel

**Comment:** Criterion 7: Pixel introduces an integrated robot-learning platform combining teleoperation and robot-free demonstration collection, a shared human/robot policy-learning pipeline, and hierarchical language-conditioned control, evaluated on dexterous mobile manipulation and whole-body coordination.

**Relevance:** 8
Back to [[topic](#topic-7)] [[top](#topics)]
