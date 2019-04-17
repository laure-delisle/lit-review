# Object detection (OD)

Object detection requires localization.

**First step**: extract regions of interest.

**1-** Segmentation

- Region extraction: construct region tree using hierarchical segmentation engine  
**From contours to regions: An empirical evaluation** [[url](https://vision.ics.uci.edu/papers/ArbelaezMFM_CVPR_2009/ArbelaezMFM_CVPR_2009.pdf)](_Arbelaez et al. CVPR 2009_)  
Oriented Watershed Transform (OWT) to form initial regions from contours, followed by construction of an Ultrametric Contour Map (UCM) defining a hierarchical segmentation.  
[[https://github.com/laure-delisle/lit-review/blob/master/docs/img/contours_to_regions.png|alt=Contours to regions]]

- **Recovering occlusion boundaries from an image** [[url](https://www.ri.cmu.edu/pub_files/pub4/hoiem_derek_2007_3/hoiem_derek_2007_3.pdf)](_Hoeim et al. ICCV 2007_)  
Segmentation using occlusion boundaries.  
[[../img/object_detection/occlusion_recovery.png|alt=Occlusion recovery]]

**2-** Region proposal

- **Category independent proposal** [[ur](http://dhoiem.cs.illinois.edu/publications/eccv2010_CategoryIndependentProposals_ian.pdf)](_Endres et Hoeim, ECCV 2010_)  
Hierarchical segmentation with agglomerative grouping, based on boundary strength. Then groups newly obtained regions with seeding: starting from a region, appearance and boundaries around the seed are used to identify other regions that might belong to the same object. Uses _Hoeim 2007_  
[[https://github.com/laure-delisle/lit-review/blob/master/docs/img/category_independent_proposal.png|alt=Category independent proposal]]


- **Recognition using regions** [[url](http://www-bcf.usc.edu/~limjj/paper/glam_cvpr09.pdf)]
(_Gu et al., 2009 CVPR_), used for both object detection and semantic segmentation.  
Produces a "robust bag" of overlaid regions (Region=set of image cues (color, texture, shape)), learns region weights using a max-margin framework. Uses _Arbelaez 2009_  
[[https://github.com/laure-delisle/lit-review/blob/master/docs/img/region_tree.png|alt=Region tree]]

- **Region Proposal Networks** [[url](https://arxiv.org/pdf/1506.01497)]
(_Gu et al., 2009 CVPR_), used for both object detection and semantic segmentation.  
Produces a "robust bag" of overlaid regions (Region=set of image cues (color, texture, shape)), learns region weights using a max-margin framework. Uses _Arbelaez 2009_  
[[https://github.com/laure-delisle/lit-review/blob/master/docs/img/region_tree.png|alt=Region tree]]


**3-** Evaluate how good/relevant the extracted regions are

- **Measuring the objectness of image windows** [[url](http://calvin.inf.ed.ac.uk/wp-content/uploads/Publications/alexe12pami.pdf)](_Alexe et al., TPAMI 2012_)  
Distinguish objects with a well-defined boundary in space, such as cows and telephones, from amorphous
background elements, such as grass and road. The measure combines in a Bayesian framework several image cues measuring characteristics of objects:  
. well-defined closed boundary in space [Edge density, Superpixels straddling],  
. different appearance from its surroundings [Color contrast],  
. unique within the image and stands out as salient [Multi-scale saliency].  
[[https://github.com/laure-delisle/lit-review/blob/master/docs/img/objectness.png|alt=Objectness]]

### R-CNN (region) [[url](https://arxiv.org/pdf/1311.2524.pdf])]
> _Girshick et al., UC Berkeley, 2014 CVPR_

### Fast R-CNN [[url](https://arxiv.org/pdf/1504.08083)]
> _Girshick et al., Microsoft Research, 2015 ICCV_

### Faster R-CNN [[url](https://arxiv.org/abs/1506.01497)]
> _Ren et al., 2015 NIPS_

### YOLO (you only look once) [[url](https://arxiv.org/pdf/1506.02640.pdf)]
> _Redmon et al., UW Allen Institute for AI and FAIR, 2016 CVPR_

### R-FCN [[url](https://arxiv.org/pdf/1605.06409.pdf)]
> _Dai et al., Microsoft Research, 2016 NIPS_

### SSD [[url](https://arxiv.org/pdf/1512.02325.pdf)]
> _Liu et al, UNC Chapel Hill Zoox Google and U Michigan, 2016 ECCV_ 

# Feature extraction

### Feature Pyramid Networks for Object Detection [[url](https://arxiv.org/pdf/1612.03144.pdf)]
> _Lin et al., FAIR and Cornell, 2017 CVPR_

**Observation**: Low-resolution are semantically strong features, high-resolution are semantically weak features. Need to combine both.

**Results**: SOTA on COCO using FPN with Faster R-CNN detector. Can be trained end-to-end.

For generic feature extraction. Uses a pyramidal structure with:
- top-down pathway
- lateral skip connections
- prediction at each level 

Prediction: 3×3 convolution is appended on each merged map to generate the final feature map, which is to reduce the aliasing effect of upsampling

[[https://github.com/laure-delisle/lit-review/blob/master/docs/img/fpn_structure.png|alt=FPN structure]]

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

