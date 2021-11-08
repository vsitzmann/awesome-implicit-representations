# Awesome Implicit Neural Representations [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
A curated list of resources on implicit neural representations, inspired by [awesome-computer-vision](https://github.com/jbhuang0604/awesome-computer-vision).

## Hiring graduate students!
I am looking for graduate students to join my new lab at MIT CSAIL in July 2022. 
If you are excited about neural implicit representations, neural rendering, neural scene representations, and their applications
in vision, graphics, and robotics, apply [here](https://gradapply.mit.edu/eecs/apply/login/)! In the webform, you can choose me as "Potential Adviser", 
and in your SoP, please describe how our research interests are well-aligned. The deadline is Dec 15th!

## Disclaimer
This list does __not aim to be exhaustive__, as implicit neural representations are a rapidly growing research field with
hundreds of papers to date. Instead, it lists the papers that I give my students to read, which introduce key concepts & foundations of 
implicit neural representations across applications. I will therefore generally __not merge pull requests__. 
This is not an evaluation of the quality or impact of a paper, but rather the result of my and my students' research interests.

However, if you see potential for another list that is broader or narrower in scope, get in touch, and I'm happy 
to link to it right here and contribute to it as well as I can!

Disclosure: I am an author on the following papers.
* [Scene Representation Networks: Continuous 3D-Structure-Aware Neural Scene Representations](https://vsitzmann.github.io/srns/)
* [MetaSDF: MetaSDF: Meta-Learning Signed Distance Functions](https://vsitzmann.github.io/metasdf/)
* [Implicit Neural Representations with Periodic Activation Functions](https://vsitzmann.github.io/siren/)
* [Inferring Semantic Information with 3D Neural Scene Representations](https://www.computationalimaging.org/publications/semantic-srn/)
* [Light Field Networks: Neural Scene Representations with Single-Evaluation Rendering](vsitzmann.github.io/lfns/)


## Table of contents
- [What are implicit neural representations?](#what-are-implicit-neural-representations) 
- [Why are they interesting?](#why-are-they-interesting) 
- [Colabs](#colabs) 
- [Papers](#papers) 
- [Talks](#Talks)

## What are implicit neural representations?
Implicit Neural Representations (sometimes also referred to as coordinate-based representations) are a novel way to parameterize
signals of all kinds. Conventional signal representations are usually discrete - for instance, images are discrete grids
of pixels, audio signals are discrete samples of amplitudes, and 3D shapes are usually parameterized as grids of voxels,
point clouds, or meshes. In contrast, Implicit Neural Representations parameterize a signal as a *continuous function* that 
maps the domain of the signal (i.e., a coordinate, such as a pixel coordinate for an image) to whatever is at that coordinate
(for an image, an R,G,B color). Of course, these functions are usually not analytically tractable - it is impossible to 
"write down" the function that parameterizes a natural image as a mathematical formula. Implicit Neural Representations
thus approximate that function via a neural network.

## Why are they interesting?
Implicit Neural Representations have several benefits: First, they are not coupled to spatial resolution anymore, the way, for instance,
an image is coupled to the number of pixels. This is because they are continuous functions! 
Thus, the memory required to parameterize the signal is *independent* of spatial
resolution, and only scales with the complexity of the underyling signal. Another corollary of this is that implicit
representations have "infinite resolution" - they can be sampled at arbitrary spatial resolutions. 

This is immediately useful for a number of applications, such as super-resolution, or in parameterizing signals in 3D and higher dimensions,
where memory requirements grow intractably fast with spatial resolution.
Further, generalizing across neural implicit representations amounts to learning a prior over a space of functions, implemented
via learning a prior over the weights of neural networks - this is commonly referred to as meta-learning and is an extremely exciting
intersection of two very active research areas!
Another exciting overlap is between neural implicit representations and the study of symmetries in neural network architectures -
for intance, creating a neural network architecture that is 3D rotation-equivariant immediately yields a viable path to rotation-equivariant generative models via neural implicit representations.

Another key promise of implicit neural representations lie in algorithms that directly operate in the space
of these representations. In other words: What's the "convolutional neural network" equivalent of a neural network
operating on images represented by implicit representations?

# Colabs
This is a list of Google Colabs that immediately allow you to jump in and toy around with implicit neural representations!
* [Implicit Neural Representations with Periodic Activation Functions](https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb)
shows how to fit images, audio signals, and even solve simple Partial Differential Equations with the SIREN architecture.
* [Neural Radiance Fields (NeRF)](https://colab.research.google.com/github/bmild/nerf/blob/master/tiny_nerf.ipynb)
shows how to fit a neural radiance field, allowing novel view synthesis of a single 3D scene.
* [MetaSDF & MetaSiren](https://colab.research.google.com/github/vsitzmann/metasdf/blob/master/MetaSDF.ipynb) shows how 
  you can leverage gradient-based meta-learning to generalize across neural implicit representations.

# Papers
## Implicit Neural Representations of Geometry
The following three papers first (and concurrently) demonstrated that implicit neural representations outperform grid-, point-, and mesh-based 
representations in parameterizing geometry and seamlessly allow for learning priors over shapes.
* [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://arxiv.org/abs/1901.05103) (Park et al. 2019) 
* [Occupancy Networks: Learning 3D Reconstruction in Function Space](https://arxiv.org/abs/1812.03828) (Mescheder et al. 2019)
* [IM-Net: Learning Implicit Fields for Generative Shape Modeling](https://arxiv.org/abs/1812.02822) (Chen et al. 2018)

Since then, implicit neural representations have achieved state-of-the-art-results in 3D computer vision:
* [Sal: Sign agnostic learning of shapes from raw data](https://github.com/matanatz/SAL) (Atzmon et al. 2019) shows how we may learn SDFs from raw data (i.e., without ground-truth signed distance values)
* [Implicit Geometric Regularization for Learning Shapes](https://github.com/amosgropp/IGR) (Gropp et al. 2020) shows how we may learn SDFs from raw data (i.e., without ground-truth signed distance values)
* [Local Implicit Grid Representations for 3D Scenes](https://geometry.stanford.edu/papers/jsmhnf-lligrf3s-20/jsmhnf-lligrf3s-20.pdf), [Convolutional Occupancy Networks](https://arxiv.org/abs/2003.04618), [Deep Local Shapes: Learning Local SDF Priors for Detailed 3D Reconstruction](https://arxiv.org/abs/2003.10983)
concurrently proposed hybrid voxelgrid/implicit representations to fit large-scale 3D scenes.
* [Implicit Neural Representations with Periodic Activation Functions](https://vsitzmann.github.io/siren/) (Sitzmann et al. 2020) 
demonstrates how we may parameterize room-scale 3D scenes via a single implicit neural representation by leveraging sinusoidal activation functions.
* [Neural Unsigned Distance Fields for Implicit Function Learning](https://arxiv.org/pdf/2010.13938.pdf) (Chibane et al. 2020) 
proposes to learn unsigned distance fields from raw point clouds, doing away with the requirement of water-tight surfaces.

## Implicit representations of Geometry and Appearance 
### From 2D supervision only (“inverse graphics”)
3D scenes can be represented as 3D-structured neural scene representations, i.e., neural implicit representations that map a 
3D coordinate to a representation of whatever is at that 3D coordinate. This then requires the formulation of a neural renderer,
in particular, a ray-marcher, which performs rendering by repeatedly sampling the neural implicit representation along a ray.
* [Scene Representation Networks: Continuous 3D-Structure-Aware Neural Scene Representations](https://vsitzmann.github.io/srns/) proposed to learn an implicit representations
 of 3D shape and geometry given only 2D images, via a differentiable ray-marcher, and generalizes across 3D scenes for 
 reconstruction from a single image via hyper-networks. This was demonstrated for single-object scenes, but also for simple room-scale scenes (see talk).
* [Differentiable volumetric rendering: Learning implicit 3d representations without 3d supervision](https://github.com/autonomousvision/differentiable_volumetric_rendering) (Niemeyer et al. 2020), 
replaces LSTM-based ray-marcher in SRNs with a fully-connected neural network & analytical gradients, enabling easy extraction of the final 3D geometry.
* [Neural Radiance Fields (NeRF)](https://www.matthewtancik.com/nerf) (Mildenhall et al. 2020) proposes positional encodings, volumetric rendering & ray-direction conditioning for high-quality reconstruction of 
single scenes, and has spawned a large amount of follow-up work on volumetric rendering of 3D implicit representations. 
For a curated list of NeRF follow-up work specifically, see [awesome-NeRF](https://github.com/yenchenlin/awesome-NeRF)
* [SDF-SRN: Learning Signed Distance 3D Object Reconstruction from Static Images](https://github.com/chenhsuanlin/signed-distance-SRN) (Lin et al. 2020), 
demonstrates how we may train Scene Representation Networks from a single observation only.
* [Pixel-NERF](https://alexyu.net/pixelnerf/) (Yu et al. 2020) proposes to condition a NeRF on local features lying on camera rays,
 extracted from contact images, as proposed in PiFU (see "from 3D supervision").
* [Multiview neural surface reconstruction by disentangling geometry and appearance](https://lioryariv.github.io/idr/) (Yariv et al. 2020)
demonstrates sphere-tracing with positional encodings for reconstruction of complex 3D scenes, and proposes a surface normal and view-direction
dependent rendering network for capturing view-dependent effects.

One may also encode geometry and appearance of a 3D scene via its 360-degree, 4D light field. This obviates the need for 
ray-marching and enables real-time rendering and fast training with minimal memory footprint, but requires additional machinery to ensure
multi-view consistency.
* [Light Field Networks: Neural Scene Representations with Single-Evaluation Rendering](vsitzmann.github.io/lfns/) (Sitzmann et al. 2021) 
proposes to represent 3D scenes via their 360-degree light field parameterized as a neural implicit representation.

### From 3D supervision
* [Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization](https://shunsukesaito.github.io/PIFu/) (Saito et al. 2019)
Pifu first introduced the concept of conditioning an implicit representation on local features extracted from context images. Follow-up work 
achieves photo-realistic, real-time re-rendering.
* [Texture Fields: Learning Texture Representations in Function Space](https://autonomousvision.github.io/texture-fields/) (Oechsle et al.)

### For dynamic scenes
* [Occupancy flow: 4d reconstruction by learning particle dynamics](https://avg.is.tuebingen.mpg.de/publications/niemeyer2019iccv) 
(Niemeyer et al. 2019) first proposed to learn a space-time neural implicit representation by representing a 4D warp field 
with an implicit neural representation.

The following papers concurrently proposed to leverage a similar approach for the reconstruction of dynamic scenes 
from 2D observations only via Neural Radiance Fields.
* [D-NeRF: Neural Radiance Fields for Dynamic Scenes](https://arxiv.org/abs/2011.13961)
* [Deformable Neural Radiance Fields](https://nerfies.github.io/)
* [Neural Radiance Flow for 4D View Synthesis and Video Processing](https://yilundu.github.io/nerflow/)
* [Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes](http://www.cs.cornell.edu/~zl548/NSFF/)
* [Space-time Neural Irradiance Fields for Free-Viewpoint Video](https://video-nerf.github.io/)
* [Non-Rigid Neural Radiance Fields: Reconstruction and Novel View Synthesis of a Deforming Scene from Monocular Video](https://gvv.mpi-inf.mpg.de/projects/nonrigid_nerf/)

## Symmetries in Implicit Neural Representations
* [Vector Neurons: A General Framework for SO(3)-Equivariant Networks](https://cs.stanford.edu/~congyue/vnn/) (Deng et al. 2021) 
makes conditional implicit neural representations equivariant to SO(3), enabling the learning of a rotation-equivariant
  shape space and subsequent reconstruction of 3D geometry of single objects in unseen poses.


## Hybrid implicit / explicit (condition implicit on local features)
The following four papers concurrently proposed to condition an implicit neural representation on local features stored in a voxelgrid:
* [Implicit Functions in Feature Space for 3D ShapeReconstruction and Completion](https://virtualhumans.mpi-inf.mpg.de/papers/chibane20ifnet/chibane20ifnet.pdf)
* [Local Implicit Grid Representations for 3D Scenes](https://geometry.stanford.edu/papers/jsmhnf-lligrf3s-20/jsmhnf-lligrf3s-20.pdf)
* [Convolutional Occupancy Networks](https://arxiv.org/abs/2003.04618)
* [Deep Local Shapes: Learning Local SDF Priors for Detailed 3D Reconstruction](https://arxiv.org/abs/2003.10983)

This has since been leveraged for inverse graphics as well:
* [Neural Sparse Voxel Fields](https://github.com/facebookresearch/NSVF) Applies a similar concept to neural radiance fields.
* [Pixel-NERF](https://alexyu.net/pixelnerf/) (Yu et al. 2020) proposes to condition a NeRF on local features lying on camera rays,
  extracted from contact images, as proposed in PiFU (see "from 3D supervision").

The following papers condition a deep signed distance function on local patches:
* [Local Deep Implicit Functions for 3D Shape](https://ldif.cs.princeton.edu/)
* [PatchNets: Patch-Based Generalizable Deep Implicit 3D Shape Representations](http://gvv.mpi-inf.mpg.de/projects/PatchNets/)

### For point cloud registration
* [DPDist: Comparing Point Clouds Using Deep Point Cloud Distance](https://github.com/dahliau/DPDist/) (Urbach et al. 2020) first proposed to use a pre-trained deep implicit representation for training a registration network over sparse point clouds directly. They train a deep implicit neural representation on local features extracts from a [3DmFV](https://github.com/sitzikbs/3DmFV-Net) representation.

## Representation learning with implicit neural representations for downstream tasks
* [Inferring Semantic Information with 3D Neural Scene Representations](https://www.computationalimaging.org/publications/semantic-srn/) leverages
features learned by Scene Representation Networks for weakly supervised semantic segmentation of 3D objects.

## Generalization & Meta-Learning with Neural Implicit Representations
* DeepSDF, Occupancy Networks, IM-Net concurrently proposed conditioning via concatenation.
* [Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization](https://shunsukesaito.github.io/PIFu/) (Saito et al. 2019)
proposed to locally condition implicit representations on ray features extracted from context images.
* [Scene Representation Networks: Continuous 3D-Structure-Aware Neural Scene Representations](https://vsitzmann.github.io/srns/) (Sitzmann et al. 2019) proposed meta-learning via hypernetworks.
* [MetaSDF: MetaSDF: Meta-Learning Signed Distance Functions](https://vsitzmann.github.io/metasdf/) (Sitzmann et al. 2020) proposed gradient-based meta-learning for implicit neural representations
* [SDF-SRN: Learning Signed Distance 3D Object Reconstruction from Static Images](https://github.com/chenhsuanlin/signed-distance-SRN) (Lin et al. 2020) show how to learn 3D implicit representations from single-image supervision only.
* [Learned Initializations for Optimizing Coordinate-Based Neural Representations](https://www.matthewtancik.com/learnit) (Tancik et al. 2020) explored gradient-based meta-learning for NeRF.

## Fitting high-frequency detail with positional encoding & periodic nonlinearities
* [Neural Radiance Fields (NeRF)](https://www.matthewtancik.com/nerf) (Mildenhall et al. 2020) proposed positional encodings.
* [Implicit Neural Representations with Periodic Activation Functions](https://vsitzmann.github.io/siren/) (Sitzmann et al. 2020) proposed implicit representations with periodic nonlinearities.
* [Fourier features let networks learn high frequency functions in low dimensional domains](https://people.eecs.berkeley.edu/~bmild/fourfeat/) (Tancik et al. 2020) explores positional encodings in an NTK framework.

## Implicit Neural Representations of Images
* [Compositional Pattern-Producing Networks: Compositional pattern producing networks: A novel abstraction of development](https://link.springer.com/content/pdf/10.1007/s10710-007-9028-8.pdf) (Stanley et al. 2007) 
first proposed to parameterize images implicitly via neural networks.
* [Implicit Neural Representations with Periodic Activation Functions](https://vsitzmann.github.io/siren/) (Sitzmann et al. 2020) proposed to generalize across implicit representations of images via hypernetworks.
* [X-Fields: Implicit Neural View-, Light- and Time-Image Interpolation](https://xfields.mpi-inf.mpg.de/) (Bemana et al. 2020) parameterizes the Jacobian of pixel position with respect to view, time, illumination, etc. to naturally interpolate images.
* [Learning Continuous Image Representation with Local Implicit Image Function](https://github.com/yinboc/liif) (Chen et al. 2020) proposed a hypernetwork-based GAN for images.

## Composing implicit neural representations
The following papers propose to assemble scenes from per-object 3D implicit neural representations.
* [GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields](https://arxiv.org/abs/2011.12100) (Niemeyer et al. 2021) 
* [Object-centric Neural Rendering](https://arxiv.org/pdf/2012.08503.pdf) (Guo et al. 2020)

## Implicit Representations for Partial Differential Equations & Boundary Value Problems
* [Implicit Geometric Regularization for Learning Shapes](https://github.com/amosgropp/IGR) (Gropp et al. 2020) learns SDFs by enforcing constraints of the Eikonal equation via the loss.
* [Implicit Neural Representations with Periodic Activation Functions](https://vsitzmann.github.io/siren/) (Sitzmann et al. 2020) proposes to leverage the periodic sine as an 
activation function, enabling the parameterization of functions with non-trivial higher-order derivatives and the solution of complicated PDEs.
* [AutoInt: Automatic Integration for Fast Neural Volume Rendering](https://davidlindell.com/publications/autoint) (Lindell et al. 2020)
* [MeshfreeFlowNet: Physics-Constrained Deep Continuous Space-Time Super-Resolution Framework](http://www.maxjiang.ml/proj/meshfreeflownet) (Jiang et al. 2020) performs super-resolution for spatio-temporal flow functions using local implicit representaitons, with auxiliary PDE losses.

## Generative Adverserial Networks with Implicit Representations
### For 3D
* [Generative Radiance Fields for 3D-Aware Image Synthesis](https://autonomousvision.github.io/graf/) (Schwarz et al. 2020)
* [pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis](https://arxiv.org/abs/2012.00926) (Chan et al. 2020)
* [Unconstrained Scene Generation with Locally Conditioned Radiance Fields](https://arxiv.org/pdf/2104.00670.pdf) (DeVries et al. 2021) Leverage a hybrid implicit-explicit representation, 
  by generating a 2D feature grid floorplan with a classic convolutional GAN, and then conditioning a 3D neural implicit representation on these features.
  This enables generation of room-scale 3D scenes.

### For 2D
For 2D image synthesis, neural implicit representations enable the generation of high-resolution images, while also 
allowing the principled treatment of symmetries such as rotation and translation equivariance.
* [Adversarial Generation of Continuous Images](https://arxiv.org/abs/2011.12026) (Skorokhodov et al. 2020) 
* [Learning Continuous Image Representation with Local Implicit Image Function](https://github.com/yinboc/liif) (Chen et al. 2020) 
* [Image Generators with Conditionally-Independent Pixel Synthesis](https://arxiv.org/abs/2011.13775) (Anokhin et al. 2020)
* [Alias-Free GAN](https://nvlabs.github.io/alias-free-gan/) (Karras et al. 2021)

## Image-to-image translation
* [Spatially-Adaptive Pixelwise Networks for Fast Image Translation](https://arxiv.org/pdf/2012.02992.pdf) (Shaham et al. 2020)
leverages a hybrid implicit-explicit representation for fast high-resolution image2image translation.
  
## Articulated representations
* [NASA: Neural Articulated Shape Approximation](https://virtualhumans.mpi-inf.mpg.de/papers/NASA20/NASA.pdf) (Deng et al. 2020) 
represents an articulated object as a composition of local, deformable implicit elements.

# Talks
* [Vincent Sitzmann: Implicit Neural Scene Representations (Scene Representation Networks, MetaSDF, Semantic Segmentation with Implicit Neural Representations, SIREN)](https://www.youtube.com/watch?v=__F9CCqbWQk&amp;t=1s)
* [Andreas Geiger: Neural Implicit Representations for 3D Vision (Occupancy Networks, Texture Fields, Occupancy Flow, Differentiable Volumetric Rendering, GRAF)](https://www.youtube.com/watch?v=F9mRv4v80w0)
* [Gerard Pons-Moll: Shape Representations: Parametric Meshes vs Implicit Functions](https://www.youtube.com/watch?v=_4E2iEmJXW8)
* [Yaron Lipman: Implicit Neural Representations](https://www.youtube.com/watch?v=rUd6qiSNwHs&list=PLat4GgaVK09e7aBNVlZelWWZIUzdq0RQ2&index=11) 

# Links
* [awesome-NeRF](https://github.com/yenchenlin/awesome-NeRF) - List of implicit representations specifically on neural radiance fields (NeRF)

## License
License: MIT

