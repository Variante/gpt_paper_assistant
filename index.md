---
title: Arxiv Daily
---

# Personalized Daily Arxiv Papers 11/18/2026

<time id="gptpa-last-update" datetime="2026-11-18T15:00:02.177428+00:00">Last update: 2026-11-18T15:00:02.177428+00:00</time>

This project is adapted from [tatsu-lab/gpt_paper_assistant](https://github.com/tatsu-lab/gpt_paper_assistant). The source code of this project is at [Variante/gpt_paper_assistant](https://github.com/Variante/gpt_paper_assistant)

About me on [Bilibili](https://space.bilibili.com/823532). Help keep the website running:

<a href="https://www.buymeacoffee.com/Variante"><img src="https://img.buymeacoffee.com/button-api/?text=Help cover GPT cost&emoji=🐢&slug=Variante&button_colour=83593e&font_colour=ffffff&font_family=Poppins&outline_colour=000000&coffee_colour=FFDD00" /></a>


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
1000\. [Making Future-Predictive Representations Actionable](https://arxiv.org/abs/2611.22994) [[more](#1000-making-future-predictive-representations-actionable)]  
**Authors:** Fangchuan Wang, Hao Zhang, Qiyan Zhao, ..., Xu-Yao Zhang, Yuqing Gao

1006\. [TCP4D: A Unified Tool-Center-Point Tracking and Generation Model for Robot Manipulation](https://arxiv.org/abs/2611.23143) [[more](#1006-tcp4d-a-unified-tool-center-point-tracking-and-generation-model-for-robot-manipulation)]  
**Authors:** Haosong Peng, Xiangyang Liu, Zhengshen Zhang, ..., Yufeng Zhan, Wenchao Xu


Back to [[top](#topics)]

---
<a id="topic-2"></a>

### Topic 2
2004\. [SpikWM: Efficient End-to-End Spiking World Modeling for Visual Planning](https://arxiv.org/abs/2611.22981) [[more](#2004-spikwm-efficient-end-to-end-spiking-world-modeling-for-visual-planning)]  
**Authors:** Shuhan Ye, Hongbin Yu, Chong Wang, ..., Jiangbo Qian, Xudong Jiang


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
5002\. [BRepA: Graph-Based B-Rep Pretraining for Unified CAD Understanding and Generation](https://arxiv.org/abs/2611.22903) [[more](#5002-brepa-graph-based-b-rep-pretraining-for-unified-cad-understanding-and-generation)]  
**Authors:** Dekai Zhu, Zhaiyu Chen, Stefan Gavranovic, Slobodan Ilic

5005\. [Any2Building: Unifying Point Clouds and Images for Building Reconstruction via Shared Spatial Queries](https://arxiv.org/abs/2611.23187) [[more](#5005-any2building-unifying-point-clouds-and-images-for-building-reconstruction-via-shared-spatial-queries)]  
**Authors:** Hongye Hou, Zhuoxiao Li, Liu Xiangyu, ..., Tongyan Hua, Wufan Zhao

5008\. [RFW-S: Everyday Deformable Objects with Independent Simulation Verification](https://arxiv.org/abs/2611.23056) [[more](#5008-rfw-s-everyday-deformable-objects-with-independent-simulation-verification)]  
**Authors:** Jiarui Guo, Saining Zhang, Jiaying Wan, ..., Hao Zhao, Chaojian Li


Back to [[top](#topics)]

---
<a id="topic-6"></a>

### Topic 6
6003\. [$π^4$: A Foundation Model for 4D Scene Geometry and Motion](https://arxiv.org/abs/2611.23144) [[more](#6003-4-a-foundation-model-for-4d-scene-geometry-and-motion)]  
**Authors:** Yifan Wang, Weiguang Zhao, Ni Yang, ..., Weinan Zhang, Chunhua Shen

6007\. [Refin3R: Geometry-Guided Refinement for Feed-Forward 3D Reconstruction](https://arxiv.org/abs/2611.23051) [[more](#6007-refin3r-geometry-guided-refinement-for-feed-forward-3d-reconstruction)]  
**Authors:** Jongmin Lee, Jaeyeong Kim, Seungryong Kim


Back to [[top](#topics)]

---
<a id="topic-7"></a>

### Topic 7
7001\. [PAIR: Bringing Household Robots into the Smart Home](https://arxiv.org/abs/2611.23070) [[more](#7001-pair-bringing-household-robots-into-the-smart-home)]  
**Authors:** Junseong Pyo, Hyeongjin Kim, Jongwook Han, ..., Sungjib Lim, Yohan Jo


Back to [[top](#topics)]

---
<a id="go-beyond"></a>

### Go beyond


Back to [[top](#topics)]

---
## Full paper list
 <a id="1000-making-future-predictive-representations-actionable"></a>

### 1000\. [Making Future-Predictive Representations Actionable](https://arxiv.org/abs/2611.22994)
**ArXiv:** 2611.22994 [[page](https://arxiv.org/abs/2611.22994)] [[pdf](https://arxiv.org/pdf/2611.22994.pdf)]

**Authors:** Fangchuan Wang, Hao Zhang, Qiyan Zhao, ..., Xu-Yao Zhang, Yuqing Gao

**Abstract:** Recent world action models (WAMs) drop future prediction at inference, assuming that representations learned through auxiliary future supervision still benefit action generation. However, our layerwise analysis of Fast-WAM reveals a depth-dependent mismatch: while probes decode future state transitions more accurately from deeper visual layers, the gradients and visual focuses of the future and action objectives increasingly diverge with depth, leaving much of this predictive information unused by the action pathway. To address this issue, we introduce RelayWAM, which adds an explicit transition space between the video and action backbones. Conditioned on the task instruction, robot state, and current action estimate, recurrent transition layers extract step-relevant predictive cues across depth and relay them directly into the action attention, without generating future frames at inference. Experiments show that RelayWAM substantially amplifies the benefit of future supervision, translating into stronger generalization under distribution shifts in both simulation and real-world robot tasks. On LIBERO-Plus, RelayWAM achieves 72.03% mean success, compared to 56.42% without future supervision and 51.50% for Fast-WAM.

**Comment:** Criterion 1: RelayWAM uses instruction-conditioned recurrent transition layers to relay future-predictive video representations into robot action attention, achieving 72.03% mean success on LIBERO-Plus versus 51.50% for Fast-WAM.

**Relevance:** 10
Back to [[topic](#topic-1)] [[top](#topics)]

<a id="1006-tcp4d-a-unified-tool-center-point-tracking-and-generation-model-for-robot-manipulation"></a>

### 1006\. [TCP4D: A Unified Tool-Center-Point Tracking and Generation Model for Robot Manipulation](https://arxiv.org/abs/2611.23143)
**ArXiv:** 2611.23143 [[page](https://arxiv.org/abs/2611.23143)] [[pdf](https://arxiv.org/pdf/2611.23143.pdf)]

**Authors:** Haosong Peng, Xiangyang Liu, Zhengshen Zhang, ..., Yufeng Zhan, Wenchao Xu

**Abstract:** Robot manipulation videos provide a rich visual experience, yet recovering robot action and gripper-state annotations from RGB observations still remains challenging. Using the embodiment-agnostic tool-center-point (TCP) pose as the action space, we present TCP4D, a unified model in which a shared 4D scene representation both tracks observed TCP motion and generates future TCP trajectories. Given a monocular video and initial 2D TCP queries, the tracker directly recovers metric TCP poses in SE(3) and gripper states alongside scene geometry and camera parameters. The generator samples local features from the same 4D representation along the recovered TCP tracks, then combines them with robot observations and language instructions to predict future TCP states. Our new cross-embodiment augmentation pipeline converts existing RGB videos into demonstrations for alternative embodiments by retargeting the recovered motion and synthesizing the corresponding observations, without recorded source-robot states. Experiments show that the tracker achieves absolute trajectory errors of 7.60 and 16.07 mm on DROID and RoboLab and adapts to a new embodiment with only ten demonstrations. The generator achieves 69.19% mean success on RoboTwin and outperforms well-established policy models on two real-world tasks, while joint training on source and augmented demonstrations improves target-robot success by 41.5% over augmented-only training.

**Comment:** Criterion 1: TCP4D generates language-conditioned TCP trajectories and augments robot-learning data through motion retargeting and synthesized observations, with joint source/augmented training improving target-robot success by 41.5% over augmented-only training.

**Relevance:** 8
Back to [[topic](#topic-1)] [[top](#topics)]

---
<a id="2004-spikwm-efficient-end-to-end-spiking-world-modeling-for-visual-planning"></a>

### 2004\. [SpikWM: Efficient End-to-End Spiking World Modeling for Visual Planning](https://arxiv.org/abs/2611.22981)
**ArXiv:** 2611.22981 [[page](https://arxiv.org/abs/2611.22981)] [[pdf](https://arxiv.org/pdf/2611.22981.pdf)]

**Authors:** Shuhan Ye, Hongbin Yu, Chong Wang, ..., Jiangbo Qian, Xudong Jiang

**Abstract:** Goal-conditioned planning with image observations and goals requires predictions that distinguish visual states and action consequences. Spiking neural networks offer sparse computation, but single-step spikes can obscure motion cues and merge actions with different physical outcomes. We propose the Spiking Latent World Model (SpikWM), an end-to-end joint-embedding predictive architecture (JEPA) with one simulation step per image and one neural update per latent position. Spiking Regional Encoding retains spatial cues through regional pooling. Bi-Spike Action Encoding combines positive and negative binary responses, while Causal Membrane Prediction carries subthreshold evidence across the latent history with learned group-wise time constants. The encoder and predictor are trained jointly on reward-free image--action trajectories using next-embedding prediction and SIGReg. With a common planner, SpikWM achieves mean success rates of $88.67\%$, $97.33\%$, $99.33\%$, and $86.67\%$ on Push-T, Two-Room, Reacher, and OGBench-Cube, respectively, outperforming two spiking baselines. Push-T ablations support each module, and controlled replays examine responses to appearance and physical state changes. Under a matched planning budget, SpikWM reduces estimated arithmetic energy by $75.6\%$ relative to LeWM. Code will be released at https://github.com/ShawnYE618/SpikWM.

**Comment:** Criterion 2: SpikWM introduces a self-supervised spiking JEPA with regional image encoding and causal membrane prediction, learning visual representations through next-embedding prediction and SIGReg on reward-free image–action trajectories, with ablations on Push-T.

**Relevance:** 8
Back to [[topic](#topic-2)] [[top](#topics)]

---
<a id="5002-brepa-graph-based-b-rep-pretraining-for-unified-cad-understanding-and-generation"></a>

### 5002\. [BRepA: Graph-Based B-Rep Pretraining for Unified CAD Understanding and Generation](https://arxiv.org/abs/2611.22903)
**ArXiv:** 2611.22903 [[page](https://arxiv.org/abs/2611.22903)] [[pdf](https://arxiv.org/pdf/2611.22903.pdf)]

**Authors:** Dekai Zhu, Zhaiyu Chen, Stefan Gavranovic, Slobodan Ilic

**Abstract:** Boundary representation (B-Rep) of CAD models encodes geometric primitives and their relations as a graph structure, enabling efficient shape modeling within CAD software. Learning on B-Reps for downstream tasks, such as segmentation and novel shape generation, requires specialized neural network designs. Moreover, performance improvements and accelerated training can be achieved with effective pretraining strategies, which is particularly important for tasks with limited labeled B-Rep data. In this paper, we introduce BRepA, a unified self-supervised pretraining framework that explicitly reconstructs both the geometry and topology of B-Rep objects via graph-based modeling. By learning from large-scale unlabeled CAD data only once, BRepA produces representations that generalize across tasks and significantly enhance both discriminative and generative learning. On segmentation tasks, BRepA demonstrates remarkable data efficiency. For instance, with only 1% labeled samples, BRepA pretraining improves mIoU by 32.52% compared to models trained from scratch. On generative tasks, aligning diffusion model representations with pretrained BRepA features substantially accelerates convergence, achieving nearly 2 times faster training while improving generation quality under the same computational budget, despite the diffusion model being roughly 20 times larger than BRepA. Overall, our results show that geometry- and topology-aware B-Rep pretraining provides a powerful foundation for both understanding and generating CAD structures, serving as a versatile representation backbone for diverse downstream applications. We will release the code to facilitate future research.

**Comment:** Criterion 5: BRepA aligns CAD diffusion models with geometry- and topology-aware pretrained B-Rep features, achieving nearly 2× faster training while improving 3D generation quality under the same computational budget.

**Relevance:** 9
Back to [[topic](#topic-5)] [[top](#topics)]

<a id="5005-any2building-unifying-point-clouds-and-images-for-building-reconstruction-via-shared-spatial-queries"></a>

### 5005\. [Any2Building: Unifying Point Clouds and Images for Building Reconstruction via Shared Spatial Queries](https://arxiv.org/abs/2611.23187)
**ArXiv:** 2611.23187 [[page](https://arxiv.org/abs/2611.23187)] [[pdf](https://arxiv.org/pdf/2611.23187.pdf)]

**Authors:** Hongye Hou, Zhuoxiao Li, Liu Xiangyu, ..., Tongyan Hua, Wufan Zhao

**Abstract:** Abstracting structured buildings from limited observations is essential for 3D urban modeling and digital twins. Partial point clouds and overhead images of the same building provide incomplete 3D surface measurements and visual cues to its footprint, roof geometry, and rooftop structures, respectively. This motivates a unified framework that aligns geometric and visual cues in 3D and reconstructs buildings from either or both modalities. Building on this connection, we present Any2Building, a unified model for Level of Detail 2 (LoD2) building reconstruction from partial point clouds, overhead images, or both, with optional coarse geometric priors. At its core is an occupancy-grounded shared spatial queries representation that organizes available observations within a common 3D grid. Supervised by complete building surface voxels, these queries connect observational evidence to target geometry, enabling a shared generator to learn structural priors across tasks and providing spatial guidance for recovering missing structure. A trained voxel conditional surface-generation and low-poly remeshing pipeline then converts the predicted structure into compact LoD2 building models. To support this framework, we introduce BuildingWorld-Image, comprising approximately one million paired samples of partial point clouds, overhead images, and LoD2 building models across 18 cities. Experiments on multiple benchmarks demonstrate that Any2Building outperforms existing methods in geometric reconstruction accuracy across unimodal and multimodal point-cloud completion, single-image building generation, and prior-assisted generation, while maintaining detailed and robust recovery when point cloud input modality is degraded.

**Comment:** Matches criterion 5 through image-to-3D building generation using occupancy-grounded shared spatial queries and a voxel-conditional surface-generation pipeline, outperforming existing methods in geometric accuracy on single-image and multimodal reconstruction benchmarks.

**Relevance:** 8
Back to [[topic](#topic-5)] [[top](#topics)]

<a id="5008-rfw-s-everyday-deformable-objects-with-independent-simulation-verification"></a>

### 5008\. [RFW-S: Everyday Deformable Objects with Independent Simulation Verification](https://arxiv.org/abs/2611.23056)
**ArXiv:** 2611.23056 [[page](https://arxiv.org/abs/2611.23056)] [[pdf](https://arxiv.org/pdf/2611.23056.pdf)]

**Authors:** Jiarui Guo, Saining Zhang, Jiaying Wan, ..., Hao Zhao, Chaojian Li

**Abstract:** Large-scale robotic simulation requires diverse deformable objects with executable physical properties and realistic interaction behaviors. However, existing 3D asset collections primarily emphasize visual diversity, while scalable physical construction and verification remain challenging. We introduce RFW-S, a large-scale dataset of 83,365 simulation-verified assets spanning 485 everyday categories across five structural families: soft solids, cloth sheets, shells, garments, and bags. We develop an automated pipeline integrating reference-guided 3D generation, structure-aware physical modeling, and LLM-assisted parameter estimation and calibration. To ensure physical reliability, we introduce an independent simulation verification framework that evaluates object behaviors under interactions distinct from those used for parameter calibration. Extensive experiments demonstrate the effectiveness of independent verification and characterize deformable responses under different physical configurations. We further evaluate robotic grasping in terms of both task success and object integrity, and demonstrate the physical realism of our simulated objects through sim-to-real replay of fragile-object manipulation. RFW-S provides a scalable foundation for physically grounded simulation and embodied robotic learning.

**Comment:** Criterion 5: RFW-S combines reference-guided 3D generation with independent simulation verification of 83,365 deformable assets, evaluating their physical fidelity through robotic grasping and sim-to-real fragile-object manipulation.

**Relevance:** 8
Back to [[topic](#topic-5)] [[top](#topics)]

---
<a id="6003-4-a-foundation-model-for-4d-scene-geometry-and-motion"></a>

### 6003\. [$π^4$: A Foundation Model for 4D Scene Geometry and Motion](https://arxiv.org/abs/2611.23144)
**ArXiv:** 2611.23144 [[page](https://arxiv.org/abs/2611.23144)] [[pdf](https://arxiv.org/pdf/2611.23144.pdf)]

**Authors:** Yifan Wang, Weiguang Zhao, Ni Yang, ..., Weinan Zhang, Chunhua Shen

**Abstract:** $π^4$ is a unified foundation model for 4D scene perception. From diverse, uncalibrated RGB observations in the wild---a single image, an unordered collection, or a video sequence---it recovers dense geometry, camera poses, and metric scale, together with 2D/3D point tracks and moving-object masks for multi-image inputs. It couples a geometry branch with a geometry-conditioned motion branch, which predicts image-space correspondences that are lifted into 3D using reconstructed target-view geometry. Across diverse benchmarks, $π^4$ achieves the best overall average rank in geometry evaluation, alongside competitive performance in single- and multiview 2D/3D point tracking and moving-object segmentation. With a shared backbone, the full geometry-and-motion system contains approximately 0.5B parameters---less than half the size of most baselines. It produces joint reconstructions and tracks in one forward pass, including large-scale scenes with thousands of input images despite training with contexts of at most 64 views. We further present the data and training recipe underlying these capabilities, together with empirical studies of key design choices. Code and models are available at https://github.com/yyfz/Pi4.

**Comment:** Criterion 6: π⁴ jointly reconstructs dense 3D geometry, camera poses, and metric scale from uncalibrated RGB observations, scaling to scenes with thousands of input images despite training with at most 64 views.

**Relevance:** 9
Back to [[topic](#topic-6)] [[top](#topics)]

<a id="6007-refin3r-geometry-guided-refinement-for-feed-forward-3d-reconstruction"></a>

### 6007\. [Refin3R: Geometry-Guided Refinement for Feed-Forward 3D Reconstruction](https://arxiv.org/abs/2611.23051)
**ArXiv:** 2611.23051 [[page](https://arxiv.org/abs/2611.23051)] [[pdf](https://arxiv.org/pdf/2611.23051.pdf)]

**Authors:** Jongmin Lee, Jaeyeong Kim, Seungryong Kim

**Abstract:** Feed-forward 3D reconstruction models predict dense point maps from multiple images in a single pass. Without explicit geometric constraints, however, their predictions are often inconsistent across views and lack fine-grained precision. Recent methods incorporate geometric priors such as camera pose or depth, but use them merely as auxiliary inputs, process each view independently, or refine the geometry only where the priors are available. We present Refin3R, which refines feed-forward point maps with Structure-from-Motion (SfM) points, propagating SfM geometry over the image grid. In its dual-branch architecture, a 2D branch extends SfM geometry beyond its coverage using geometry-aware backbone features, while a 3D branch reasons over the SfM points of all views, capturing multi-view structure that the image grid alone cannot represent. The two branches exchange information bidirectionally to jointly refine each view, yielding dense and accurate point maps. Across ScanNet++, ETH3D, Tanks\&Temples, and out-of-domain human and surgical scenes, Refin3R consistently outperforms feed-forward and geometry-guided methods in point map accuracy while running faster than the strongest geometry-guided baseline.

**Comment:** Criterion 6: Refin3R introduces interacting 2D and 3D branches that propagate SfM geometry across multi-view point maps, improving reconstruction accuracy on ScanNet++, ETH3D, and Tanks&Temples.

**Relevance:** 8
Back to [[topic](#topic-6)] [[top](#topics)]

---
<a id="7001-pair-bringing-household-robots-into-the-smart-home"></a>

### 7001\. [PAIR: Bringing Household Robots into the Smart Home](https://arxiv.org/abs/2611.23070)
**ArXiv:** 2611.23070 [[page](https://arxiv.org/abs/2611.23070)] [[pdf](https://arxiv.org/pdf/2611.23070.pdf)]

**Authors:** Junseong Pyo, Hyeongjin Kim, Jongwook Han, ..., Sungjib Lim, Yohan Jo

**Abstract:** Household robots often need to operate appliances, and smart appliances offer both control panels and device APIs. Existing embodied AI simulation environments, however, model appliances solely as physical objects. We introduce PAIR, a smart-home simulation environment grounded in the Matter protocol~\footnote{https://csa-iot.org/all-solutions/matter/}, the industry standard for smart-home devices. PAIR provides 84 smart-device assets whose functional control panels and APIs act on a shared device state, with ambient conditions that evolve in response to both. From a survey of 1,443 requests, we construct a benchmark of 40 tasks that all require object manipulation yet remain solvable without API access, with 340 human demonstrations for policy training. Whether run alone, with a high-level planner, or with an API-enabled planner, two fine-tuned VLA policies complete none of the 120 evaluation episodes. Robots rarely get far enough to use the API: even with oracle plans and API access, over 90\% of failed rollouts stall at navigating, grasping, or operating a door. With a perfect oracle in place of the robot's body, the best planner still gets through only 57.3\% of its plan. When choosing between pressing a button and calling an API, several planners go by whether they can read the panel, not by API latency.\footnote{We will release our code and data upon publication.}

**Comment:** Criterion 7: PAIR introduces a smart-home robot-learning environment with 84 API-controllable device assets, 40 manipulation tasks, and 340 training demonstrations, accompanied by evaluations of VLA policies and tool-using planners.

**Relevance:** 10
Back to [[topic](#topic-7)] [[top](#topics)]
