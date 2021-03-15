# IFSS-Net

# Paper Title: 
IFSS-Net: Interactive Few-Shot Siamese Network for Faster Muscle Segmentation and Propagation in Volumetric Ultrasound

# Paper Links:
https://pubmed.ncbi.nlm.nih.gov/33560982/
https://ieeexplore.ieee.org/document/9350651
https://arxiv.org/abs/2011.13246


# Abstract:
We present an accurate, fast and efficient method for segmentation and muscle mask propagation in 3D freehand ultrasound data, towards accurate volume quantification. A deep Siamese 3D Encoder-Decoder network that captures the evolution of the muscle appearance and shape for contiguous slices is deployed. We use it to propagate a reference mask annotated by a clinical expert. To handle longer changes of the muscle shape over the entire volume and to provide an accurate propagation, we devise a Bidirectional Long Short Term Memory module. Also, to train our model with a minimal amount of training samples, we propose a strategy combining learning from few annotated 2D ultrasound slices with sequential pseudo-labelling of the unannotated slices. We introduce a decremental update of the objective function to guide the model convergence in the absence of large amounts of annotated data. After training with a few volumes, the decremental update strategy switches from a weak supervised training to a few-shot setting. Finally, to handle the class-imbalance between foreground and background muscle pixels, we propose a parametric Tversky loss function that learns to penalize adaptively the false positives and the false negatives. We validate our approach for the segmentation, label propagation, and volume computation of the three low-limb muscles on a dataset of 61600 images from 44 subjects. We achieve a Dice score coefficient of over 95 % and a volumetric error of 1.6035±0.587%.


# Citation:
@ARTICLE{9350651,
  author={D. A. {Chanti} and V. G. {Duque} and M. {Crouzier} and A. {Nordez} and L. {Lacourpaille} and D. {Mateus}},
  journal={IEEE Transactions on Medical Imaging}, 
  title={IFSS-Net: Interactive Few-Shot Siamese Network for Faster Muscle Segmentation and Propagation in Volumetric Ultrasound}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2021.3058303}}



# Code Description:
> This code is a draft of the implementation of the main model provided in the paper: IFSS-Net DOI: 10.1109/TMI.2021.3058303

> It contains three files, data_utilities, IFSSNet_utilities and IFSS-Net.

> The code represent a prototype code, it is not fully clean or optimized, but it can be easilty used and repreduced.
  
> Some of the functions in IFSSNet_utilities are borrowed from cited papers in our article.

## Environment and Dependencies
  1. TensorFlow 1.14 and slim
  2. OpenCV
  3. matplotlib
  4. slim.
  5. some other packages as seaborn, numpy, etc
