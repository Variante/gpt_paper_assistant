---
title: Arxiv Daily
---

# Personalized Daily Arxiv Papers 11/10/2026

<time id="gptpa-last-update" datetime="2026-11-10T15:00:03.557601+00:00">Last update: 2026-11-10T15:00:03.557601+00:00</time>

This project is adapted from [tatsu-lab/gpt_paper_assistant](https://github.com/tatsu-lab/gpt_paper_assistant). The source code of this project is at [Variante/gpt_paper_assistant](https://github.com/Variante/gpt_paper_assistant)

About me on [Bilibili](https://space.bilibili.com/823532). Help keep the website running:

<a href="https://www.buymeacoffee.com/Variante"><img src="https://img.buymeacoffee.com/button-api/?text=Help cover GPT cost&emoji=🍗&slug=Variante&button_colour=5a4f69&font_colour=ffffff&font_family=Bree&outline_colour=000000&coffee_colour=FFDD00" /></a>


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
1001\. [VLA Adaptation with Real and Simulated Data: A Case Study on Robotic Chess](https://arxiv.org/abs/2611.12676) [[more](#1001-vla-adaptation-with-real-and-simulated-data-a-case-study-on-robotic-chess)]  
**Authors:** Matěj Kripner, Milan Straka

1003\. [SWAM: Scaling World Action Models with Cross-Modal Alignment](https://arxiv.org/abs/2611.12870) [[more](#1003-swam-scaling-world-action-models-with-cross-modal-alignment)]  
**Authors:** Yu Zhou, Xinyuan Xia, Xuankun Yang, ..., Cewu Lu, Yanwei Li

1005\. [Learning Heterogeneous Tactile Representation for General Dexterous Manipulation](https://arxiv.org/abs/2611.12806) [[more](#1005-learning-heterogeneous-tactile-representation-for-general-dexterous-manipulation)]  
**Authors:** Bing Han, Yichuan Peng, Haoyu Dong, ..., Nan Xue, Wei Ning


Back to [[top](#topics)]

---
<a id="topic-2"></a>

### Topic 2
2007\. [$Σ$-LeWM: Learning World Models via S-Information for Robot Control](https://arxiv.org/abs/2611.12732) [[more](#2007-lewm-learning-world-models-via-s-information-for-robot-control)]  
**Authors:** Xinrui Zu, Michael Przystupa, Zhao Yang, Shujian Yu, Kevin Sebastian Luck


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
5002\. [Plant Orbiter: 3D Plant Reconstruction and Generation with Video Diffusion Orbit Cameras and 3D Gaussian Splatting](https://arxiv.org/abs/2611.12739) [[more](#5002-plant-orbiter-3d-plant-reconstruction-and-generation-with-video-diffusion-orbit-cameras-and-3d-gaussian-splatting)]  
**Authors:** Zane K. J. Hartley, Yong En Kok, Lewis A. G. Stuart, Andrew P. French, Michael P. Pound


Back to [[top](#topics)]

---
<a id="topic-6"></a>

### Topic 6
6000\. [Spaghetti Splatting: Differentiating Curvilinear Dirac Masses via Convolution](https://arxiv.org/abs/2611.12716) [[more](#6000-spaghetti-splatting-differentiating-curvilinear-dirac-masses-via-convolution)]  
**Authors:** George Shramko, Shrisudhan Govindarajan, Rudy Lam, ..., David B. Lindell, Andrea Tagliasacchi

6006\. [Click3R: Interactive Stereo 3D Reconstruction with Sparse Correspondence Clicks](https://arxiv.org/abs/2611.12875) [[more](#6006-click3r-interactive-stereo-3d-reconstruction-with-sparse-correspondence-clicks)]  
**Authors:** Hao Gu, Jin Yao, Zezhou Cheng


Back to [[top](#topics)]

---
<a id="topic-7"></a>

### Topic 7
7004\. [World In-Context Learning for Agentic Robot Manipulation](https://arxiv.org/abs/2611.12885) [[more](#7004-world-in-context-learning-for-agentic-robot-manipulation)]  
**Authors:** Heng Fang, Zehui Guo, Yuehan Zhang, ..., Dingkang Liang, Xiang Bai

7008\. [Library of Hands: Fix the Hands, Not the World Model](https://arxiv.org/abs/2611.12678) [[more](#7008-library-of-hands-fix-the-hands-not-the-world-model)]  
**Authors:** Hasaan Ahmad, Qizhen Ying, Yuxuan Sun, Tianjin Huang, Guangming Wang


Back to [[top](#topics)]

---
<a id="go-beyond"></a>

### Go beyond


Back to [[top](#topics)]

---
## Full paper list
 <a id="1001-vla-adaptation-with-real-and-simulated-data-a-case-study-on-robotic-chess"></a>

### 1001\. [VLA Adaptation with Real and Simulated Data: A Case Study on Robotic Chess](https://arxiv.org/abs/2611.12676)
**ArXiv:** 2611.12676 [[page](https://arxiv.org/abs/2611.12676)] [[pdf](https://arxiv.org/pdf/2611.12676.pdf)]

**Authors:** Matěj Kripner, Milan Straka

**Abstract:** Vision-language-action (VLA) models are trained as general-purpose foundation models for robot learning, and require fine-tuning for each downstream task and robot. This raises practical questions such as how much data and training updates are needed, how checkpoints should be selected, and whether simulation can be leveraged to alleviate the cost of real data collection. We study these questions on a task that requires both precise manipulation and spatial understanding of the scene. In this task, an SO-101 arm moves a chess king between two squares that are given only by their names, such as e4 and a5. We fine-tune $π_{0.5}$ with various dataset sizes and update budgets, using teleoperated demonstrations as well as simulated demonstrations from a randomized scripted expert that imitates the variability of human demonstrations. In simulation, the success rate scales with both data and updates. Importantly, we find that a short learning-rate cool-down adds up to 31 points and is necessary for comparing checkpoints, and that validation loss and action error are unreliable for model selection. On the real arm, we fine-tune on up to 256 teleoperated demonstrations and evaluate ten policies in 25 trials each. Partial results indicate that pretraining on the simulated demonstrations helps substantially, most clearly when real data is scarce. We also present a negative result, in which training on the policy's own rollouts with filtered behavior cloning, RECAP-style advantage conditioning, corrections from an automatic coach, or residual reinforcement learning did not yield a measurable improvement over plain behavior cloning in our setting. We release the simulator, the expert and the real demonstrations.

**Comment:** Criterion 1: Fine-tunes the π0.5 vision-language-action model for language-specified chess-piece manipulation on an SO-101 arm, finding that simulated-demonstration pretraining helps substantially when real demonstrations are scarce.

**Relevance:** 10
Back to [[topic](#topic-1)] [[top](#topics)]

<a id="1003-swam-scaling-world-action-models-with-cross-modal-alignment"></a>

### 1003\. [SWAM: Scaling World Action Models with Cross-Modal Alignment](https://arxiv.org/abs/2611.12870)
**ArXiv:** 2611.12870 [[page](https://arxiv.org/abs/2611.12870)] [[pdf](https://arxiv.org/pdf/2611.12870.pdf)]

**Authors:** Yu Zhou, Xinyuan Xia, Xuankun Yang, ..., Cewu Lu, Yanwei Li

**Abstract:** World action models jointly predict video and actions conditioned on language. Existing studies focus primarily on architectures and training mechanisms, typically treating language-video alignment acquired through video pretraining as an established foundation. However, introducing action learning may impair this alignment. Principled guidance on data and supervision design for maintaining language-video alignment during action learning remains limited. To investigate this, we curate a pool of approximately 47,000 hours of general videos, egocentric videos, UMI demonstrations, and robot demonstrations. Building on this, we systematically study how language-video supervision, language-video-action supervision, and the joint use of these two forms affect WAM performance. In particular, we examine how language annotation quality affects pretrained language-video capabilities, and how action supervision from UMI and robot demonstrations extends these capabilities to language-video-action modeling. To better scale the model, we further conduct detailed studies on joint training recipe and model architecture. Under a fixed training budget, mixing egocentric videos and UMI demonstrations improves control generalization. Model architecture affects how effectively these experiences are used, while scaling pretraining under the selected configuration brings further gains. The resulting recipe, post-trained on RoboTwin Clean demonstrations only, achieves success rates of 88.10% and 61.56% under Easy (Clean-to-Clean) and Hard (Clean-to-Random) conditions, respectively. Our model achieves 40.00% SR across three real-world tasks, compared with 23.33% for the strongest baseline, indicating better generalization under unseen conditions. We release the curated data, training code, and model weights.

**Comment:** Criterion 1: SWAM jointly predicts language-conditioned video and robot actions, with its cross-modal training recipe achieving 61.56% success on RoboTwin Hard and 40.00% across three real-world tasks.

**Relevance:** 10
Back to [[topic](#topic-1)] [[top](#topics)]

<a id="1005-learning-heterogeneous-tactile-representation-for-general-dexterous-manipulation"></a>

### 1005\. [Learning Heterogeneous Tactile Representation for General Dexterous Manipulation](https://arxiv.org/abs/2611.12806)
**ArXiv:** 2611.12806 [[page](https://arxiv.org/abs/2611.12806)] [[pdf](https://arxiv.org/pdf/2611.12806.pdf)]

**Authors:** Bing Han, Yichuan Peng, Haoyu Dong, ..., Nan Xue, Wei Ning

**Abstract:** Tactile sensing is an indispensable modality for contact-rich dexterous manipulation, a capability that hints at the ultimate frontiers of embodied intelligence. However, commercially available tactile sensors are still at a relatively early stage of development, with each type of sensor offering irreplaceable strengths alongside its own drawbacks. A key challenge is thus to free tactile understanding from sensor-specific physics, so that embodied models can effectively leverage heterogeneous tactile data. To this end, we develop LingBot-Touch, a tactile foundation model characterized by the following properties. (i) Our model yields a unified representation across different tactile sensors (i.e., piezoresistive, magnetic, and visuotactile sensors), and is easily extensible to new sensor types. (ii) The resulting tactile tokens are plug-and-play: they can be seamlessly integrated into policy models across paradigms, from diffusion policy to vision-language-action (VLA) models such as $π_{0.5}$ and LingBot-VLA, consistently enhancing manipulation dexterity. (iii) Extensive experiments, spanning simulation and real-robot deployments across multiple hardware platforms, demonstrate that our tactile representation substantially benefits complex manipulation tasks while maintaining strong computational efficiency. We believe the findings in this work demonstrate the potential of leveraging cross-sensor data to scale tactile learning, and offer encouraging insights for the design of future hardware and embodied models.

**Comment:** Criterion 1: LingBot-Touch integrates unified tactile representations into diffusion policies and vision-language-action models such as LingBot-VLA, improving dexterous manipulation across simulation and real-robot deployments.

**Relevance:** 9
Back to [[topic](#topic-1)] [[top](#topics)]

---
<a id="2007-lewm-learning-world-models-via-s-information-for-robot-control"></a>

### 2007\. [$Σ$-LeWM: Learning World Models via S-Information for Robot Control](https://arxiv.org/abs/2611.12732)
**ArXiv:** 2611.12732 [[page](https://arxiv.org/abs/2611.12732)] [[pdf](https://arxiv.org/pdf/2611.12732.pdf)]

**Authors:** Xinrui Zu, Michael Przystupa, Zhao Yang, Shujian Yu, Kevin Sebastian Luck

**Abstract:** World models have increasingly been applied as a model-based control method for robotics tasks such as manipulation and locomotion. While early world model architectures relied on decoders to prevent mode collapse and learn a predictable latent space, recent works on world models, such as Joint Embedding Predictive Architecture or LeWorldModel, increasingly aim to learn embeddings without a grounding decoder, utilising only a latent forward dynamics predictor and a stabilising entropy term. In this paper, we set out to demonstrate that the further introduction of two additional latent losses to LeWorldModel, a backward dynamics predictor and an inverse dynamics action predictor, enforces further structure on the latent embedding, leading to further gains in both performance and planning efficiency. These are critical properties for improving the applicability of such methods to real-time model-based control of real-world robots. We demonstrate these gains on standard world model benchmarks and two simulated robot manipulation tasks.

**Comment:** Criterion 2: Adds backward-dynamics and inverse-action prediction losses to decoder-free LeWorldModel representation learning, improving performance and planning efficiency on standard world-model benchmarks and two simulated robot-manipulation tasks.

**Relevance:** 8
Back to [[topic](#topic-2)] [[top](#topics)]

---
<a id="5002-plant-orbiter-3d-plant-reconstruction-and-generation-with-video-diffusion-orbit-cameras-and-3d-gaussian-splatting"></a>

### 5002\. [Plant Orbiter: 3D Plant Reconstruction and Generation with Video Diffusion Orbit Cameras and 3D Gaussian Splatting](https://arxiv.org/abs/2611.12739)
**ArXiv:** 2611.12739 [[page](https://arxiv.org/abs/2611.12739)] [[pdf](https://arxiv.org/pdf/2611.12739.pdf)]

**Authors:** Zane K. J. Hartley, Yong En Kok, Lewis A. G. Stuart, Andrew P. French, Michael P. Pound

**Abstract:** Modern digital phenotyping pipelines require the precise measurement of a range of 3D morphological plant traits. The accurate, high throughput acquisition of these traits helps breeders and researchers accelerate the development of new varieties, techniques and products. Recent advances in 3D technologies have enabled a rapid transition from 2D to 3D pipelines, but high throughput acquisition of digital 3D twins of plants remains a significant challenge due to complex morphology, thin structures and occlusions. In this paper we present Plant Orbiter, a new approach to generating 3D plants that leverages recent advances in video diffusion to produce high quality 3D Gaussian Splats of plants from a single image. We show that our method more accurately recreates plant morphology than other state-of-the-art methods, achieving the highest 3D F-score against all baselines, while being more practical for high throughput pipelines due to easier alignment and reduced cost of camera placement. Combined with a 2D diffusion model, we show that our method also achieves state-of-the-art results in the generation of 3D plant models from a text prompt.

**Comment:** Criterion 5: Plant Orbiter uses video diffusion to generate 3D Gaussian splats from a single image, achieves the highest 3D F-score against its baselines, and supports text-to-3D plant generation.

**Relevance:** 10
Back to [[topic](#topic-5)] [[top](#topics)]

---
<a id="6000-spaghetti-splatting-differentiating-curvilinear-dirac-masses-via-convolution"></a>

### 6000\. [Spaghetti Splatting: Differentiating Curvilinear Dirac Masses via Convolution](https://arxiv.org/abs/2611.12716)
**ArXiv:** 2611.12716 [[page](https://arxiv.org/abs/2611.12716)] [[pdf](https://arxiv.org/pdf/2611.12716.pdf)]

**Authors:** George Shramko, Shrisudhan Govindarajan, Rudy Lam, ..., David B. Lindell, Andrea Tagliasacchi

**Abstract:** In production systems, 3D geometry is represented by primitives such as points, lines, and triangles, as rendering hardware is optimized to generate images from them very efficiently. Learning such primitives from images by volume rendering is challenging, as they are Dirac masses defined at infinitesimal locations in space. EWA splatting, Poisson reconstruction, and 3D Gaussian Splatting overcome this by convolving Dirac masses with a low-frequency kernel, a recipe that generalizes along two axes: the kernel and the primitive. As points and triangles have already received significant attention, we investigate this recipe for curvilinear primitives, and introduce Spaghetti Splatting. We approximate curves by polylines whose convolution converges to that of the continuous curve, and volume render each segment so that connected segments composite without overcounting at their joints. For fast training and inference, we linearize the camera projection per segment, which also accommodates kernels beyond Gaussians, such as Beta kernels. As a point is a curve of zero length, our representation is a strict superset of 3D Gaussian Splatting. It remains competitive with the state-of-the-art on general scenes, outperforms methods specialized to line-like geometry, natively supports anti-aliasing, and produces long connected strands that are directly amenable to physics simulation.

**Comment:** Criterion 6: Introduces differentiable curvilinear primitives that strictly generalize 3D Gaussian Splatting, remaining competitive on general scenes while outperforming methods specialized to line-like geometry.

**Relevance:** 10
Back to [[topic](#topic-6)] [[top](#topics)]

<a id="6006-click3r-interactive-stereo-3d-reconstruction-with-sparse-correspondence-clicks"></a>

### 6006\. [Click3R: Interactive Stereo 3D Reconstruction with Sparse Correspondence Clicks](https://arxiv.org/abs/2611.12875)
**ArXiv:** 2611.12875 [[page](https://arxiv.org/abs/2611.12875)] [[pdf](https://arxiv.org/pdf/2611.12875.pdf)]

**Authors:** Hao Gu, Jin Yao, Zezhou Cheng

**Abstract:** Despite impressive progress, existing feed-forward large 3D reconstruction models produce a single, irrevocable prediction. When these models fail, there is no mechanism to correct their mistakes. To address this limitation, we introduce Click3R, a framework for interactive geometric correction of feed-forward models using sparse human correspondence clicks. Given an image pair and a small set of cross-view point correspondences provided by a user, Click3R injects geometric constraints into the 3D reconstruction networks through a lightweight point correspondence adapter, enabling the model to resolve ambiguities in challenging scenarios such as repetitive textures, symmetric structures, large viewpoint changes, or textureless regions. To evaluate interactive reconstruction, we curate a challenging benchmark specifically designed to expose failure modes of existing feed-forward 3D reconstruction methods. Experiments show that Click3R reduces reconstruction error with as few as a single correspondence click, improving performance on both standard and our curated benchmarks. Our results highlight that sparse human interaction provides an effective and practical mechanism for correcting geometric reconstruction errors.

**Comment:** Criterion 6: Click3R advances sparse-view 3D reconstruction by injecting user-provided cross-view correspondences through a lightweight adapter, reducing stereo reconstruction error with as little as one correspondence click.

**Relevance:** 9
Back to [[topic](#topic-6)] [[top](#topics)]

---
<a id="7004-world-in-context-learning-for-agentic-robot-manipulation"></a>

### 7004\. [World In-Context Learning for Agentic Robot Manipulation](https://arxiv.org/abs/2611.12885)
**ArXiv:** 2611.12885 [[page](https://arxiv.org/abs/2611.12885)] [[pdf](https://arxiv.org/pdf/2611.12885.pdf)]

**Authors:** Heng Fang, Zehui Guo, Yuehan Zhang, ..., Dingkang Liang, Xiang Bai

**Abstract:** Trained action policies write manipulation skill into network weights, tying what is learned to the distribution it is learned from. In-context learning (ICL) avoids this coupling but has not matched trained policies, because a demonstration recorded in another scene carries no feedback from the world the robot currently faces. We observe that what an agent learns in context is semantic, recording task steps and object roles rather than visual appearance, so experience gathered in one world transfers to another. Based on this insight, we propose World In-Context Learning (WICL), a paradigm in which a frozen model starts from a demonstration, rehearses the task in a world such as a simulator, and writes the feedback into its context rather than its weights. We build RoboScript, a modular infrastructure that implements WICL for agentic manipulation. From a single demonstration, RoboScript learns each task through rehearsal in simulation, freezes the learned knowledge as a Task Memory, and deploys the frozen memory on new scenes with data isolation that ensures every deployment input is available on a real robot. On all 50 tasks of RoboTwin 2.0, RoboScript reaches 83.4% mean success across clean and randomized settings with one demonstration per task and no gradient updates, exceeding the strongest trained action policy by over 20 points. The method generalizes across visual distributions, improves through reflection, scales with the foundation model, and transfers to a physical dual-arm robot through cross-embodiment memory adaptation. With a world in the loop, a robot can learn a manipulation task in its context as well as it learns one in its weights. All code is available at https://github.com/ArcheOne-AI/RoboScript

**Comment:** Criterion 7: RoboScript implements World In-Context Learning through simulation rehearsal, feedback, and reusable Task Memory, achieving 83.4% mean success across all 50 RoboTwin 2.0 tasks with one demonstration per task and no gradient updates, while also transferring to a physical dual-arm robot.

**Relevance:** 10
Back to [[topic](#topic-7)] [[top](#topics)]

<a id="7008-library-of-hands-fix-the-hands-not-the-world-model"></a>

### 7008\. [Library of Hands: Fix the Hands, Not the World Model](https://arxiv.org/abs/2611.12678)
**ArXiv:** 2611.12678 [[page](https://arxiv.org/abs/2611.12678)] [[pdf](https://arxiv.org/pdf/2611.12678.pdf)]

**Authors:** Hasaan Ahmad, Qizhen Ying, Yuxuan Sun, Tianjin Huang, Guangming Wang

**Abstract:** A world model predicts how a scene changes after an action, and a planner uses it to search for actions that reach a goal. If the camera moves or the controller changes, the same command moves the robot differently: when its commands are turned by 90 degrees, a frozen planner with the LeWM world model that succeeds in 77.5% of episodes succeeds in only 7.8%. Current fixes retrain the model, need training data, or forget a set-up when it returns. We propose Library of Hands (LoH), a training-free method that keeps the world model frozen and corrects only the map from commands to applied actions, which we call the hand. LoH reads the hand through the frozen model, with no extra sensor: it finds the action that best explains the change between two camera images (a reading). It stores every hand it identifies, so a returning set-up is recognised rather than re-learnt. In closed-loop tests, LoH more than doubles success on three simulated tasks, from 28.6% to 62.3%, and loses only 0.6 percentage points when nothing changes. It also works, unchanged, with two other world models and on two of three models pretrained on real robot data, and its gain grows with the quality of its readings.

**Comment:** Criterion 7: Library of Hands combines frozen world-model planning with feedback-based action-map identification and persistent memory of controller configurations, increasing closed-loop success across three simulated robot tasks from 28.6% to 62.3%.

**Relevance:** 8
Back to [[topic](#topic-7)] [[top](#topics)]
