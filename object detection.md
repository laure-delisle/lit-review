# Object detection (OD)

Object detection requires localization.

Classification + localization = there is a fixed number of continuous (=regression) outputs, for instance a set number of objects we'll classify and locate (coordinates are the continuous outputs). You need two losses: softmax loss for the classification scores, and an L2 loss for the box coordinates.

note: this is a **multi-task loss**, we have two scalars we want to both minimize. Using an additional hyperparameter to weight those losses, we take a weighted sum of the two loss functions to get a final scalar loss. Gradient is then computed wrt to the weighted sum of those two losses. Setting that hyperparameter is difficult, it affects the loss which in return cannot be used to adjust the value of that hyperparameter. A strategy for this is to use another performance metric to make the cross-validation choice. This can be applied to other tasks, for instance human pose estimation, where the output consists of 14 (x,y) joint positions each with a regression loss.

note: "regression loss" means something different than cross-entropy or softmax: L2 euclidian, L1, smooth L1 or any loss adapated to a continuous output.

An approach is to train the network, freeze it then train the fully connected layers for those two different tasks. However when doing transfer learning, better performance is usually attained by fine-tuning the whole system jointly. A compromise is to freeze the network, trained the last few connected layers for the two tasks until convergence, then unfreeze the rest of the network and fine-tune jointly.

Object detection = there is a varying number (unknown in advance) of objects to locate and classify

## Extract Regions Of Interest (ROI)
Sequential steps of region extraction, region proposal, evaluation of ROI relevance.

### 1- Region extraction

Construct region tree using hierarchical segmentation engine, either with contours or occlusion boundaries. In this step, we segment the whole image and attribute a region to each pixel, in a hierarchical manner.  

- **From contours to regions: An empirical evaluation** [[url](https://vision.ics.uci.edu/papers/ArbelaezMFM_CVPR_2009/ArbelaezMFM_CVPR_2009.pdf)](_Arbelaez et al. CVPR 2009_)  
Oriented Watershed Transform (OWT) to form initial regions from contours, followed by construction of an Ultrametric Contour Map (UCM) defining a hierarchical segmentation.  
![Contours to regions](./img/object_detection/contours_to_regions.png)

- **Recovering occlusion boundaries from an image** [[url](https://www.ri.cmu.edu/pub_files/pub4/hoiem_derek_2007_3/hoiem_derek_2007_3.pdf)](_Hoeim et al. ICCV 2007_)  
Segmentation using occlusion boundaries.  
[[https://github.com/laure-delisle/lit-review/blob/master/img/object_detection/occlusion_recovery.png|alt=Occlusion recovery]]

### 2- Region proposal

From hierachical segmentation, we construct regions of interest, either by agglomerative grouping, bagging or using RPN (new approach by Faster R-CNN).

- **Category independent proposal** [[ur](http://dhoiem.cs.illinois.edu/publications/eccv2010_CategoryIndependentProposals_ian.pdf)](_Endres et Hoeim, ECCV 2010_)  
Hierarchical segmentation with agglomerative grouping, based on boundary strength. Then groups newly obtained regions with seeding: starting from a region, appearance and boundaries around the seed are used to identify other regions that might belong to the same object. Uses _Hoeim 2007_  
![Category independent proposal](./img/object_detection/category_independent_proposal.png)


- **Recognition using regions** [[url](http://www-bcf.usc.edu/~limjj/paper/glam_cvpr09.pdf)]
(_Gu et al., 2009 CVPR_), used for both object detection and semantic segmentation.  
Produces a "robust bag" of overlaid regions (Region=set of image cues (color, texture, shape)), learns region weights using a max-margin framework. Uses _Arbelaez 2009_  
![Region tree](./img/object_detection/region_tree.png)

- **Selective Search** [[url](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)](_Uijlings et al., IJCV 2013_)   
Uses hierarchical grouping combined with a diverse set of complementary strategies. Initializes the regions using _Felzenszwalb and Huttenlocher 2004_ Really fast (~1k proposal per sec on a CPU), noisy but with a high recall (if there is an object in the image it is likely it is covered by a proposal).
http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf

great slides: http://vision.stanford.edu/teaching/cs231b_spring1415/slides/ssearch_schuyler.pdf

- **Region Proposal Networks** (_Faster R-CNN_)   
Produces a set of rectangular region proposals each with an objectness score, from an _n x n_ window input taken from a feature map.  
After mapping to a lower dimension feature map, this _n x n_ window is fed to two sibling _1 x 1_ conv networks: **box-regression** layer (reg) and **box-classification** layer (cls). At each sliding-window location, we simultaneously predict _k_ region proposals. The reg layer has 4k outputs encoding the coordinates of k boxes, and the cls layer outputs 2k scores that estimate probability of object or not object for each proposal. The k proposals are parameterized relative to k reference boxes, which we call **anchors**. An anchor is centered at the sliding window in question, and is associated with a scale and aspect ratio. By default RPN use 3 scales and 3 aspect ratios, yielding k = 9 anchors at each sliding position. For a convolutional feature map of a size W × H, there are W x H x k anchors in total.   
note: this method is translation invariant (in terms of anchors and function prediction the ROI based on the anchor).   
![Region Proposal Networks](./img/object_detection/region_proposal_networks.png)


### 3- Evaluate how good/relevant the extracted regions are

- **Measuring the objectness of image windows** [[url](http://calvin.inf.ed.ac.uk/wp-content/uploads/Publications/alexe12pami.pdf)](_Alexe et al., TPAMI 2012_)  
Distinguish objects with a well-defined boundary in space, such as cows and telephones, from amorphous
background elements, such as grass and road. The measure combines in a Bayesian framework several image cues measuring characteristics of objects:  
. well-defined closed boundary in space [Edge density, Superpixels straddling],  
. different appearance from its surroundings [Color contrast],  
. unique within the image and stands out as salient [Multi-scale saliency].   
note: Region Proposal Networks (RPN) output this along the ROI (same step).  
![Objectness](./img/object_detection/objectness.png)



## Detect objects in ROI

### R-CNN family
In 2014, Ross Girshick (student at UC Berkeley) publishes R-CNN (Regions with CNN-features). In the following years, he (then at Microsoft Research) improves on the original idea by publishing Fast R-CNN in 2015 and Faster R-CNN in 2016. After that, he went on to work for FAIR.

- **R-CNN**   
1/ Extract regions of interest (eg: Selective Search), R-CNN is agnostic to region proposal method
2/ Warp them a square
3/ Feed forward through a CNN -> classifies (softmax with a log loss) as one of the object class or background 
4/ Classify using SVM (hinge loss) + Bounding box regression (least squares)
Training data: each object marked with a bounding-box and labelled with a category

- **Fast R-CNN**
1/ Pass through the CNN to get a HRes feature map
2/ Extract and warp regions of interest from the feature map



## Feature extraction

### Feature Pyramid Networks for Object Detection [[url](https://arxiv.org/pdf/1612.03144.pdf)]
> _Lin et al., FAIR and Cornell, 2017 CVPR_

**Observation**: Low-resolution are semantically strong features, high-resolution are semantically weak features. Need to combine both.

**Results**: SOTA on COCO using FPN with Faster R-CNN detector. Can be trained end-to-end.

For generic feature extraction. Uses a pyramidal structure with:
- top-down pathway
- lateral skip connections
- prediction at each level 

Prediction: 3×3 convolution is appended on each merged map to generate the final feature map, which is to reduce the aliasing effect of upsampling

![FPN structure](./img/object_detection/fpn_structure.png)

**Related work**  
- _Adelson 1984 - Pyramid methods in image processing_  
Pyramids are scale-invariant.  
(+) this enables a model to detect objects across large range of scales (scan across positions and pyramid levels).  
(-) requires dense scale sampling.

- _He 2016 - Deep Residuals..., Shrivastava 2016 - Training region-based..._  
Convnets + Featurizing each level of an image pyramid.  
(+) multi-scale feature representation with all levels semantically strong, even HR.  
(-) time and memory consuming, impractical in real world.

- _Liu 2016 - SSD: Single shot multibox detector_  
Feature hierarchy layer by layer from Convnet subsampling process.  
(+) feature maps of different spatial resolutions.  
(-) large semantic gaps due to different depths: low-level HR can be harmful for OD-> SSD uses top-down to avoid low-level features, which are needed to detect small objects.

## Semantic segmentation

Task of labelling each pixel with a category label, without differenciating instances.

- Naive approach - Sliding window: extract patch, classify the patch, apply the label to the center pixel. This would be very computationally intensive (one patch per pixel) that doesn't share the info from overlapping patches.

- Convolutional approach with dimension conservation, and last layer classification, outputs a tensor _h x w x c_ with c the number of categories we want to segment. Loss function: cross-entropy loss on every pixel of the output, then sum or average over image. Assumption: we know the categories This requires training data that is very expensive to acquire. This is also computationally and memory expensive as numerous convolutions are needed are each layer due to the HRes of feature maps.

- Fully convolutional with downsampling + upsampling (U-net). Downsampling is done by pooling + strided convolutions. Upsampling can be done by "unpooling" (nearest neighbor, "bed of nails"), but spatial information is lost, we don't know where that feature vectore came from in the local receptive field after maxpooling. A way to address this is to use "max unpooling" (use positions from the pooling layer), it helps preserve some of the spatial information that was lost during maxpooling. 

[image unpooling]
[image max unpooling]

Another approach is a **learnable upsampling** such as **Transpose Convolution**. We multiply the scalar value of each pixel in the feature map with the values in the filter, i.e. we weight the filter to obtain a weighted copy of the filter as output. We  move the weighted filter using the stride as the movement ratio between the input (feature map) and the output. In case of overlap, we sum the outputs. We use those learned convolution filter weights to upsample the image and increase the spatial size. This is also called deconvolution, upconvolution, fractionally strided convolution, backward strided convolution. Summing in case of overlap creates a **checker-board effect**. A way to fix this is by changing the stride to avoid overlapping (stride = filter size) or to make it uniform (stride = filter size / 2)
Note: a convolution can always be written as a matrix multiplication.

[image transpose convolution]
[image convolution as matrix multiplication]


## References
- **R-CNN** (region) [[url](https://arxiv.org/pdf/1311.2524.pdf])], _Girshick et al., UC Berkeley, 2014 CVPR_

- **Fast R-CNN** [[url](https://arxiv.org/pdf/1504.08083)], _Girshick et al., Microsoft Research, 2015 ICCV_

- **Faster R-CNN** [[url](https://arxiv.org/abs/1506.01497)], _Ren et al., USTChina and Microsoft Research, 2015 NIPS_

- **YOLO** (you only look once) [[url](https://arxiv.org/pdf/1506.02640.pdf)], _Redmon et al., UW Allen Institute for AI and FAIR, 2016 CVPR_

- **R-FCN** (fully conv net) [[url](https://arxiv.org/pdf/1605.06409.pdf)], _Dai et al., Microsoft Research, 2016 NIPS_

- **SSD** [[url](https://arxiv.org/pdf/1512.02325.pdf)], _Liu et al, UNC Chapel Hill Zoox Google and U Michigan, 2016 ECCV_ 
