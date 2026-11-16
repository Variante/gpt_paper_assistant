---
title: Arxiv Daily
---

# Personalized Daily Arxiv Papers 11/16/2026

<time id="gptpa-last-update" datetime="2026-11-16T15:00:03.734482+00:00">Last update: 2026-11-16T15:00:03.734482+00:00</time>

This project is adapted from [tatsu-lab/gpt_paper_assistant](https://github.com/tatsu-lab/gpt_paper_assistant). The source code of this project is at [Variante/gpt_paper_assistant](https://github.com/Variante/gpt_paper_assistant)

About me on [Bilibili](https://space.bilibili.com/823532). Help keep the website running:

<a href="https://www.buymeacoffee.com/Variante"><img src="https://img.buymeacoffee.com/button-api/?text=Help cover GPT cost&emoji=🐤&slug=Variante&button_colour=098b5c&font_colour=ffffff&font_family=Bree&outline_colour=000000&coffee_colour=FFDD00" /></a>


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
1001\. [Objectives as Policies: Vision-Language Model Cost Programs for Contact-Rich Manipulation](https://arxiv.org/abs/2611.18298) [[more](#1001-objectives-as-policies-vision-language-model-cost-programs-for-contact-rich-manipulation)]  
**Authors:** Zihao Lu, Trong Gia Hung Nguyen, Fuquan Wang, Yifan Zhu

1002\. [Vision-Language-Action Models: A Life-Cycle Survey of Design, Training and Deployment](https://arxiv.org/abs/2611.18372) [[more](#1002-vision-language-action-models-a-life-cycle-survey-of-design-training-and-deployment)]  
**Authors:** Suvarna Kadam, Soran Parsa


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
7000\. [Training Reasoning Models in Simulation to Natively Use Real Robots](https://arxiv.org/abs/2611.18562) [[more](#7000-training-reasoning-models-in-simulation-to-natively-use-real-robots)]  
**Authors:** Jiaming Tang, Zhuoyang Zhang, Junxian Guo, ..., Zhijian Liu, Song Han


Back to [[top](#topics)]

---
<a id="go-beyond"></a>

### Go beyond


Back to [[top](#topics)]

---
## Full paper list
 <a id="1001-objectives-as-policies-vision-language-model-cost-programs-for-contact-rich-manipulation"></a>

### 1001\. [Objectives as Policies: Vision-Language Model Cost Programs for Contact-Rich Manipulation](https://arxiv.org/abs/2611.18298)
**ArXiv:** 2611.18298 [[page](https://arxiv.org/abs/2611.18298)] [[pdf](https://arxiv.org/pdf/2611.18298.pdf)]

**Authors:** Zihao Lu, Trong Gia Hung Nguyen, Fuquan Wang, Yifan Zhu

**Abstract:** Vision-language models (VLMs) have shown strong semantic and spatial reasoning capabilities in 3D environments. Recently, connecting VLMs with model-predictive controllers (MPC) has emerged as a paradigm for combining strong semantic reasoning with physics-based planning. However, connecting these two for contact-rich robotic manipulation requires a task representation that bridges the two and captures how physical interactions can be reasoned by VLMs. In this work, we introduce Objectives as Policies (OaP), a framework that connects a pretrained VLM to a shared sampling-based MPC through an editable objective program. Each program specifies a sequence of stages for completing a task, combining running and terminal costs from a cost library with stage completion conditions. Given a natural-language task instruction and scene observation, we build a simulation-based digital twin through real-to-sim, and the VLM generates an objective program that is executed by the MPC in the simulation. The program is iteratively updated by the VLM based on MPC rollouts, before being deployed in the real world. We evaluate our framework in both simulation and the real world on four tabletop manipulation tasks, and show that it outperforms strong baselines. Additional videos and examples are available on the project website: https://zihaolu001.github.io/oap-project/

**Comment:** Criterion 1: Objectives as Policies connects a pretrained VLM to sampling-based MPC through staged objective programs refined using simulated rollouts, outperforming strong baselines on four contact-rich tabletop manipulation tasks in simulation and the real world.

**Relevance:** 10
Back to [[topic](#topic-1)] [[top](#topics)]

<a id="1002-vision-language-action-models-a-life-cycle-survey-of-design-training-and-deployment"></a>

### 1002\. [Vision-Language-Action Models: A Life-Cycle Survey of Design, Training and Deployment](https://arxiv.org/abs/2611.18372)
**ArXiv:** 2611.18372 [[page](https://arxiv.org/abs/2611.18372)] [[pdf](https://arxiv.org/pdf/2611.18372.pdf)]

**Authors:** Suvarna Kadam, Soran Parsa

**Abstract:** Vision-language-action (VLA) models adapt web-pretrained vision-language models to output robot actions, so that web knowledge carries over to physical control. In under 4 years the field has moved from RT-2, the first VLA, to open models that fine-tune on a single consumer GPU and to proprietary generalists that clean unseen homes. This survey reviews 88 works from December 2022 to October 2026, mainly on manipulation and navigation, plus humanoid control, world-action models and evaluation. We follow a VLA through its life cycle, from model design to training and deployment, along seven axes. Model design covers (a) inputs and memory, what the model observes; (b) architecture, how its pipeline is built; (c) action output, how it produces motor commands; and (d) intermediate predictions, what else it predicts, from reasoning steps to future observations. Training covers (e) pretraining data, the mix of robot, web, human and simulated data, and (f) post-training, adaptation to the target robot. Deployment covers (g) real-time execution, keeping pace with the robot. We analyse 22 models on these axes and report their training data, model size, inference speed and hardware. Three trends emerge: (i) training data has broadened, from about 10,000 hours of teleoperated robot data ($π_0$) to 23,000 community-collected episodes (SmolVLA) and 20,000 hours of egocentric human video (GR00T N1.7); (ii) model sizes have split: most recent models have 3-7 billion parameters, while a new group under one billion runs on consumer GPUs; and (iii) generalist VLAs are emerging that handle both manipulation and navigation, such as UniVLA and Qwen-VLA. We also compare VLAs with world-action models, which predict future observations as well as actions. We review evidence that some VLAs with near-perfect benchmark scores fail when objects or the camera move slightly, and open up further research questions.

**Comment:** Criterion 1: This survey directly covers adapting pretrained VLMs into robot-action policies for manipulation, comparing 22 VLA models across architecture, action outputs, training, and real-time deployment.

**Relevance:** 8
Back to [[topic](#topic-1)] [[top](#topics)]

---
<a id="7000-training-reasoning-models-in-simulation-to-natively-use-real-robots"></a>

### 7000\. [Training Reasoning Models in Simulation to Natively Use Real Robots](https://arxiv.org/abs/2611.18562)
**ArXiv:** 2611.18562 [[page](https://arxiv.org/abs/2611.18562)] [[pdf](https://arxiv.org/pdf/2611.18562.pdf)]

**Authors:** Jiaming Tang, Zhuoyang Zhang, Junxian Guo, ..., Zhijian Liu, Song Han

**Abstract:** Robot-use agents (RUAs) have emerged as a promising approach to general-purpose robot control. However, strong performance has so far been achieved only by frontier proprietary models such as GPT-6 Astra and Claude Fable 5.2, whereas open-weight models remain substantially less capable. We show that an open-weight 27B reasoning model can be trained to use real robots natively through reinforcement learning, entirely in simulation and without demonstrations. Like a computer-use agent (CUA), the model observes the environment, reasons in a chain of thought, and generates the next end-effector pose, without any harness. We build a Post-training Engine for Agentic Robotics (PEAR) and use it to train the model on simulated robot tasks at scale with robot-use reinforcement learning (RURL), mirroring how frontier models acquire computer use. We find that the skills acquired in simulation generalize well beyond the training tasks: they transfer to an unseen robot, to the real world, and to novel instructions. After training on LIBERO and RoboDojo in simulation, the model succeeds in 60% of our trials on a real dual-arm robot, whereas the base model succeeds in none. On LIBERO, training raises the success rate from 1.6% to 74.4%, approaching GPT-6 Astra and GPT-6.1 Sol at maximum reasoning effort. Compared with $π_{0.5}$, a vision-language-action model fine-tuned on demonstrations, our model is substantially more robust to changes in the task itself. On LIBERO-Pro, $π_{0.5}$ succeeds in only 0.75% of episodes when assigned a new task in a familiar scene whereas our model succeeds in 57.4%. Finally, the model retains its general capabilities, with no measurable degradation on nine reasoning, coding and vision benchmarks. These results highlight robot-use reinforcement learning as a promising component of foundation-model post-training.

**Comment:** Criterion 7: PEAR is an agentic robotics post-training framework that trains a 27B reasoning model through demonstration-free reinforcement learning in simulation, achieving 74.4% success on LIBERO and transferring to a real dual-arm robot with 60% success.

**Relevance:** 10
Back to [[topic](#topic-7)] [[top](#topics)]
