# Awesome Implicit Neural Representations [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
A curated list of resources on implicit neural representations, inspired by [awesome-computer-vision](https://github.com/jbhuang0604/awesome-computer-vision).
Work-in-progress.

This list does not aim to be exhaustive, as implicit neural representations are a rapidly evolving & growing research field with
hundreds of papers to date. 

Instead, this list aims to list papers introducing key concepts & foundations of implicit neural representations across
applications. It's a great reading list if you want to get started in this area!

For most papers, there is a short summary of the most important contributions.

Disclosure: I am an author on the following papers:
* [Scene Representation Networks: Continuous 3D-Structure-Aware Neural Scene Representations](https://vsitzmann.github.io/srns/)
* [MetaSDF: MetaSDF: Meta-Learning Signed Distance Functions](https://vsitzmann.github.io/metasdf/)
* [Implicit Neural Representations with Periodic Activation Functions](https://vsitzmann.github.io/siren/)
* [Inferring Semantic Information with 3D Neural Scene Representations](https://www.computationalimaging.org/publications/semantic-srn/)

## What are implicit neural representations?
Implicit Neural Representations (sometimes also referred to coordinate-based representations) are a novel way to parameterize
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

However, in the future, the key promise of implicit neural representations lie in algorithms that directly operate in the space
of these representations. In other words: What's the "convolutional neural network" equivalent of a neural network
operating on images represented by implicit representations? Questions like these offer a path towards a class of algorithms
that are independent of spatial resolution!

# Papers
## Implicit Neural Representations of Geometry
The following three papers first (and concurrently) demonstrated that implicit neural representations outperform grid-, point-, and mesh-based 
representations in parameterizing geometry and seamlessly allow for learning priors over shapes.
* [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://arxiv.org/abs/1901.05103) (Park et al. 2019) 
* [Occupancy Networks: Learning 3D Reconstruction in Function Space](https://arxiv.org/abs/1812.03828) (Mescheder et al. 2019)
* [IM-Net: Learning Implicit Fields for Generative Shape Modeling](https://arxiv.org/abs/1812.02822) (Chen et al. 2018)


* [Sal: Sign agnostic learning of shapes from raw data](https://github.com/matanatz/SAL) (Atzmon et al. 2019) shows how we may learn SDFs from raw data (i.e., without ground-truth sigend distance values)
* [Implicit Geometric Regularization for Learning Shapes](https://github.com/amosgropp/IGR) (Gropp et al. 2020) shows how we may learn SDFs from raw data (i.e., without ground-truth sigend distance values)
* [Local Implicit Grid Representations for 3D Scenes](https://geometry.stanford.edu/papers/jsmhnf-lligrf3s-20/jsmhnf-lligrf3s-20.pdf), [Convolutional Occupancy Networks](https://arxiv.org/abs/2003.04618), [Deep Local Shapes: Learning Local SDF Priors for Detailed 3D Reconstruction](https://arxiv.org/abs/2003.10983)
concurrently proposed hybrid voxelgrid/implicit representations to fit large-scale 3D scenes.
* [Implicit Neural Representations with Periodic Activation Functions](https://vsitzmann.github.io/siren/) (Sitzmann et al. 2020) 
demonstrates how we may parameterize room-scale 3D scenes via a single implicit neural representation by leveraging sinusoidal activation functions.

## Implicit representations of Geometry and Appearance 
### From 2D supervision only (“inverse graphics”)
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

#### With view-dependent effects
* [Neural Radiance Fields (NeRF)](https://www.matthewtancik.com/nerf) (Mildenhall et al. 2020)
* [Multiview neural surface reconstruction by disentangling geometry and appearance](https://lioryariv.github.io/idr/) (Yariv et al. 2020)

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

## Hybrid implicit / voxelgrid
The following three papers concurrently proposed to condition an implicit neural representation on local features stored in a voxelgrid:
* [Local Implicit Grid Representations for 3D Scenes](https://geometry.stanford.edu/papers/jsmhnf-lligrf3s-20/jsmhnf-lligrf3s-20.pdf)
* [Convolutional Occupancy Networks](https://arxiv.org/abs/2003.04618)
* [Deep Local Shapes: Learning Local SDF Priors for Detailed 3D Reconstruction](https://arxiv.org/abs/2003.10983)


* [Neural Sparse Voxel Fields](https://github.com/facebookresearch/NSVF) Applies a similar concept to neural radiance fields.

## Representation learning with implicit neural representations for downstream tasks
* [Inferring Semantic Information with 3D Neural Scene Representations](https://www.computationalimaging.org/publications/semantic-srn/) leverages
features learned by Scene Representation Networks for weakly supervised semantic segmentation of 3D objects.

## On Meta-Learning with Neural Implicit Representations
* DeepSDF, Occupancy Networks, IM-Net concurrently proposed conditioning via concatenation.
* [Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization](https://shunsukesaito.github.io/PIFu/) (Saito et al. 2019)
proposed to locally condition implicit representations on ray features extracted from context images.
* [Scene Representation Networks: Continuous 3D-Structure-Aware Neural Scene Representations](https://vsitzmann.github.io/srns/) (Sitzmann et al. 2019) proposed meta-learning via hypernetworks.
* [MetaSDF: MetaSDF: Meta-Learning Signed Distance Functions](https://vsitzmann.github.io/metasdf/) (Sitzmann et al. 2020) proposed gradient-based meta-learning for implicit neural representations
* [SDF-SRN: Learning Signed Distance 3D Object Reconstruction from Static Images](https://github.com/chenhsuanlin/signed-distance-SRN) (Lin et al. 2020) show how to learn 3D implicit representations from single-image supervision only.
* [Learned Initializations for Optimizing Coordinate-Based Neural Representations](https://www.matthewtancik.com/learnit) (Tancik et al. 2020) explored gradient-based meta-learning for NeRF.

## On fitting high-frequency detail with positional encoding & periodic nonlinearities
* [Neural Radiance Fields (NeRF)](https://www.matthewtancik.com/nerf) (Mildenhall et al. 2020) proposed positional encodings.
* [Implicit Neural Representations with Periodic Activation Functions](https://vsitzmann.github.io/siren/) (Sitzmann et al. 2020) proposed implicit representations with periodic nonlinearities.
* [Fourier features let networks learn high frequency functions in low dimensional domains](https://people.eecs.berkeley.edu/~bmild/fourfeat/) (Tancik et al. 2020) explores positional encodings in an NTK framework.

## Implicit Neural Representations of Images
* [Compositional Pattern-Producing Networks: Compositional pattern producing networks: A novel abstraction of development](https://link.springer.com/content/pdf/10.1007/s10710-007-9028-8.pdf) (Stanley et al. 2007) 
first proposed to parameterize images implicitly via neural networks.
* [Implicit Neural Representations with Periodic Activation Functions](https://vsitzmann.github.io/siren/) (Sitzmann et al. 2020) proposed to generalize across implicit representations of images via hypernetworks.
* [Learning Continuous Image Representation with Local Implicit Image Function](https://github.com/yinboc/liif) (Chen et al. 2020) proposed a hypernetwork-based GAN for images.

## Composing implicit neural representations
The following papers propose to assemble scenes from per-object 3D implicit neural representations.
* [GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields](https://arxiv.org/abs/2011.12100)
* [Object-centric Neural Rendering](https://arxiv.org/pdf/2012.08503.pdf)

## Implicit Representations for Partial Differential Equations & Boundary Value Problems
* [Implicit Geometric Regularization for Learning Shapes](https://github.com/amosgropp/IGR) (Gropp et al. 2020) learns SDFs by enforcing constraints of the Eikonal equation via the loss.
* [Implicit Neural Representations with Periodic Activation Functions](https://vsitzmann.github.io/siren/) (Sitzmann et al. 2020) proposes to leverage the periodic sine as an 
activation function, enabling the parameterization of functions with non-trivial higher-order derivatives and the solution of complicated PDEs.
* [AutoInt: Automatic Integration for Fast Neural Volume Rendering](https://davidlindell.com/publications/autoint) (Lindell et al. 2020)

## Generative Adverserial Networks with Implicit Representations
* [Generative Radiance Fields for 3D-Aware Image Synthesis](https://autonomousvision.github.io/graf/) (Schwarz et al. 2020)
* [Learning Continuous Image Representation with Local Implicit Image Function](https://github.com/yinboc/liif) (Chen et al. 2020) 
* [Image Generators with Conditionally-Independent Pixel Synthesis](https://arxiv.org/abs/2011.13775) (Anokhin et al. 2020)
* [pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis](https://arxiv.org/abs/2012.00926) (Chan et al. 2020)

# Talks
* [Vincent Sitzmann: Implicit Neural Scene Representations (Scene Representation Networks, MetaSDF, Semantic Segmentation with Implicit Neural Representations, SIREN)](https://www.youtube.com/watch?v=__F9CCqbWQk&amp;t=1s)
* [Andreas Geiger: Neural Implicit Representations for 3D Vision (Occupancy Networks, Texture Fields, Occupancy Flow, Differentiable Volumetric Rendering, GRAF)](https://www.youtube.com/watch?v=F9mRv4v80w0)

# Links
* [awesome-NeRF](https://github.com/yenchenlin/awesome-NeRF) - List of implicit representations specifically on neural radiance fields (NeRF)

## License
License: MIT

