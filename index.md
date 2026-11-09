---
title: Arxiv Daily
---

# Personalized Daily Arxiv Papers 11/09/2026

<time id="gptpa-last-update" datetime="2026-11-09T15:00:03.717841+00:00">Last update: 2026-11-09T15:00:03.717841+00:00</time>

This project is adapted from [tatsu-lab/gpt_paper_assistant](https://github.com/tatsu-lab/gpt_paper_assistant). The source code of this project is at [Variante/gpt_paper_assistant](https://github.com/Variante/gpt_paper_assistant)

About me on [Bilibili](https://space.bilibili.com/823532). Help keep the website running:

<a href="https://www.buymeacoffee.com/Variante"><img src="https://img.buymeacoffee.com/button-api/?text=Help cover GPT cost&emoji=🥤&slug=Variante&button_colour=5b1e17&font_colour=ffffff&font_family=Cookie&outline_colour=000000&coffee_colour=FFDD00" /></a>


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
1000\. [ODEAction: From Predicting Chunks to Evolving Actions](https://arxiv.org/abs/2611.09672) [[more](#1000-odeaction-from-predicting-chunks-to-evolving-actions)]  
**Authors:** Qishen Wang, Pengxiang Ding, Shifeng Bao, ..., Yonghao Dang, Cheng Chi

1001\. [SURE:Spatiotemporal Uncertainty Representation for Efficient Vision-Language-Action Learning](https://arxiv.org/abs/2611.09684) [[more](#1001-surespatiotemporal-uncertainty-representation-for-efficient-vision-language-action-learning)]  
**Authors:** Jingye Zhang, Choo Sin Wai, Dake Zhong, ..., Peng Jiao, Haoqian Wang

1003\. [Seeing What Matters: Benchmarking Task-Conditioned Actionable Spatial Belief for Manipulation](https://arxiv.org/abs/2611.09794) [[more](#1003-seeing-what-matters-benchmarking-task-conditioned-actionable-spatial-belief-for-manipulation)]  
**Authors:** Yina Jian, Mingkun Xu


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


Back to [[top](#topics)]

---
<a id="topic-6"></a>

### Topic 6


Back to [[top](#topics)]

---
<a id="topic-7"></a>

### Topic 7
7002\. [On Agentic Manipulation Policies](https://arxiv.org/abs/2611.09892) [[more](#7002-on-agentic-manipulation-policies)]  
**Authors:** Chen Chen, Omid Bagherieh, Naveen Nikamanth Appiah Balaji, ..., Matthias Müller, Vladlen Koltun


Back to [[top](#topics)]

---
<a id="go-beyond"></a>

### Go beyond


Back to [[top](#topics)]

---
## Full paper list
 <a id="1000-odeaction-from-predicting-chunks-to-evolving-actions"></a>

### 1000\. [ODEAction: From Predicting Chunks to Evolving Actions](https://arxiv.org/abs/2611.09672)
**ArXiv:** 2611.09672 [[page](https://arxiv.org/abs/2611.09672)] [[pdf](https://arxiv.org/pdf/2611.09672.pdf)]

**Authors:** Qishen Wang, Pengxiang Ding, Shifeng Bao, ..., Yonghao Dang, Cheng Chi

**Abstract:** Vision-language-action (VLA) models typically generate actions in chunks, organizing continuous robot execution into separately predicted segments and thus introducing potential discontinuities at action-chunk transitions. We propose ODEAction, a chunk-free action generation paradigm that replaces action-chunk prediction with Continuous Action Evolution. Given visual observations, language instructions, and robot proprioceptive information, ODEAction learns an action transition field defined over action space, where a transition vector characterizes the evolution rate of the current action with respect to evolution time. During execution, the model takes the current action as its initial condition and integrates the learned field using an ordinary differential equation (ODE), recursively generating subsequent actions with explicit dependencies. ODEAction is implemented as an action head and does not depend on a specific vision-language backbone architecture. Experiments on LIBERO, EBench, RoboTwin, and real robots show that ODEAction improves both task performance and trajectory quality over matched action-chunking baselines, with particularly pronounced gains on fine manipulation tasks.

**Comment:** Matches criterion 1: ODEAction replaces action-chunk prediction in vision-language-action policies with an ODE-integrated action transition field, improving manipulation performance and trajectory quality on LIBERO, EBench, RoboTwin, and real robots.

**Relevance:** 10
Back to [[topic](#topic-1)] [[top](#topics)]

<a id="1001-surespatiotemporal-uncertainty-representation-for-efficient-vision-language-action-learning"></a>

### 1001\. [SURE:Spatiotemporal Uncertainty Representation for Efficient Vision-Language-Action Learning](https://arxiv.org/abs/2611.09684)
**ArXiv:** 2611.09684 [[page](https://arxiv.org/abs/2611.09684)] [[pdf](https://arxiv.org/pdf/2611.09684.pdf)]

**Authors:** Jingye Zhang, Choo Sin Wai, Dake Zhong, ..., Peng Jiao, Haoqian Wang

**Abstract:** Vision-language-action (VLA) policies have become a promising paradigm for end-to-end robot control, but their effectiveness still depends heavily on large volumes of demonstration data. In manipulation tasks, this dependence is compounded by structured spatiotemporal uncertainty in the action space: multiple nearby actions or trajectories may successfully solve the same task, while semantically equivalent demonstrations often unfold at different speeds. Standard pointwise supervision treats each demonstrated action as an exact target, imposing an unnecessary learning burden under such uncertainty. We present SURE, a data-efficient VLA training framework that explicitly models both spatial and temporal uncertainty. SURE augments a VLA backbone with a lightweight transposed-convolutional probabilistic action decoder that captures spatial uncertainty through local correlations among neighboring action bins. It further introduces Soft-DTW alignment supervision to accommodate local timing variations between predicted and demonstrated action chunks. Together, these two components reduce sensitivity to demonstrator-specific poses and execution timing, enabling more data-efficient action learning. We evaluate SURE on LIBERO, RoboTwin 2.0, and real-world manipulation tasks. Even with only 10% of the demonstration data or 20% of the training steps, SURE achieves manipulation success rates that match or surpass baseline VLAs.

**Comment:** Matches criterion 1: SURE improves vision-language-action manipulation learning through a probabilistic action decoder and Soft-DTW temporal alignment, matching or surpassing baseline VLAs with only 10% of demonstration data or 20% of training steps.

**Relevance:** 10
Back to [[topic](#topic-1)] [[top](#topics)]

<a id="1003-seeing-what-matters-benchmarking-task-conditioned-actionable-spatial-belief-for-manipulation"></a>

### 1003\. [Seeing What Matters: Benchmarking Task-Conditioned Actionable Spatial Belief for Manipulation](https://arxiv.org/abs/2611.09794)
**ArXiv:** 2611.09794 [[page](https://arxiv.org/abs/2611.09794)] [[pdf](https://arxiv.org/pdf/2611.09794.pdf)]

**Authors:** Yina Jian, Mingkun Xu

**Abstract:** When should a robot look at the cup, and when at the portafilter? Spatial-reasoning benchmarks evaluate whether a model can describe a scene, but manipulation needs something more specific: the ability to acquire the particular evidence that makes the current command executable. A view that is informative for one task can be useless for another, even at the same workstation. We introduce Seeing What Matters, a benchmark for task-conditioned actionable spatial belief in simulated tabletop manipulation. Across three domains--coffee workstation, drawer/cabinet, and microwave/appliance--the benchmark provides 860 frozen evaluation cases; expanding across methods, budgets, and inference repeats yields more than 115,000 evaluator episodes under two protocols: closed-loop actionability and same-object functional out-of-distribution generalization. Each case pairs a set of candidate RGB views with a manipulation command, shared scripted skills, and hidden postconditions, so the benchmark directly measures whether a view-selection policy acquires the observations its command actually needs. We also provide a diagnostic reference pipeline that maintains a spatial-functional belief state and ranks views by missing evidence. At primary budget settings, this pipeline reaches 98.8% closed-loop and 98.9% functional-OOD success; ablating the functional or command-binding components collapses performance to near zero. Seven vision-language models, evaluated on the same cases, reach only 26-43%. Replacing the reference pipeline's coverage metadata with frozen RGB-derived scores preserves 98.5%/98.9% success; shuffling the visual scores collapses it to 15%/11%, confirming that the result depends on image content. These gaps suggest that spatial belief for manipulation should be evaluated not by scene-level completeness, but by whether it acquires the evidence the current command requires.

**Comment:** Matches criteria 1 and 7 by introducing 860 simulated tabletop-manipulation cases evaluating task-conditioned view selection in a closed-loop manipulation protocol, where seven VLMs achieve only 26–43% success versus 98.8% for a spatial-functional-belief reference pipeline.

**Relevance:** 9
Back to [[topic](#topic-1)] [[top](#topics)]

---
<a id="7002-on-agentic-manipulation-policies"></a>

### 7002\. [On Agentic Manipulation Policies](https://arxiv.org/abs/2611.09892)
**ArXiv:** 2611.09892 [[page](https://arxiv.org/abs/2611.09892)] [[pdf](https://arxiv.org/pdf/2611.09892.pdf)]

**Authors:** Chen Chen, Omid Bagherieh, Naveen Nikamanth Appiah Balaji, ..., Matthias Müller, Vladlen Koltun

**Abstract:** Recent work on robot manipulation has focused on specialist policies trained on demonstrations. We show that generalist AI models can control manipulators more reliably than specialist policies such as vision-language-action models (VLAs) and world-action models (WAMs). We systematically analyze agentic manipulation policies (AMPs), in which a generalist AI model commands the manipulator directly, adapting its actions as the task unfolds. We present AMP1, a reference harness that runs unchanged across single-arm, bimanual, and mobile manipulation. AMP1 achieves state-of-the-art results on the RoboLab-120 and CaP-Gym benchmarks, with no demonstrations or task-specific training. On RoboLab-120, AMP1 reaches an 85.6\% success rate zero-shot, compared to 49.2\% and 42.9\% for leading VLA and WAM policies. The agent succeeds zero-shot in 104 of 108 real-world trials spanning twelve tasks and three robot instances, reducing failure rates by 20x and 15x relative to strong VLA and WAM baselines. The basic harness supplies no grasping, pushing, or placing skills; the agent develops its own strategies, such as searching for hidden objects, using objects as tools, and calibrating its cameras. With the addition of a persistent memory store, the agent can also learn from experience: an augmented harness with a memory representation that is carried across episodes lowers the failure rate on RoboDojo bimanual manipulation tasks by 4.6x relative to zero-shot deployment. Our analysis suggests that AMPs are substantially more general and robust than specialist manipulation policies, providing a promising foundation for robot manipulation.

**Comment:** Criterion 7: AMP1 provides an agentic framework for direct robot manipulation, achieving 85.6% zero-shot success on RoboLab-120, while persistent cross-episode memory reduces failure rates on RoboDojo bimanual tasks by 4.6×.

**Relevance:** 10
Back to [[topic](#topic-7)] [[top](#topics)]
