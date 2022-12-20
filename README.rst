########
ModelNet
########

`<https://modelnet.cs.princeton.edu/>`_

Goal
====

The goal of the Princeton ModelNet project is to provide researchers in
computer vision, computer graphics, robotics and cognitive science, with a
comprehensive clean collection of 3D CAD models for objects. To build the core
of the dataset, we compiled a list of the most common object categories in the
world, using the statistics obtained from the `SUN database
<http://sun.cs.princeton.edu/>`__. Once we established a vocabulary for
objects, we collected 3D CAD models belonging to each object category using
online search engines by querying for each object category term. Then, we hired
human workers on Amazon Mechanical Turk to manually decide whether each CAD
model belongs to the specified cateogries, using our `in-house designed tool
with quality control <http://3dvision.princeton.edu/code.html#TurkCleaner>`__.
To obtain a very clean dataset, we choose 10 popular object categories, and
manually deleted the models that did not belong to these categories.
Furthermore, we manually aligned the orientation of the CAD models for this
10-class subset as well. We provide both the 10-class subset and the full
dataset for download.

Download 10-Class Orientation-aligned Subset
============================================

`ModelNet10.zip
<http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip>`__:
this ZIP file contains CAD models from the 10 categories used to train the deep
network in `our 3D deep learning project
<http://vision.princeton.edu/projects/2014/3DShapeNets/>`__.  Training and
testing split is included in the file. The CAD models are completely cleaned
inhouse, and the orientations of the models (not scale) are manually aligned by
ourselves.

Download 40-Class Subset
========================

`ModelNet40.zip <http://modelnet.cs.princeton.edu/ModelNet40.zip>`__: this ZIP
file contains CAD models from the 40 categories used to train the deep network
in `our 3D deep learning project
<http://vision.princeton.edu/projects/2014/3DShapeNets/>`__.  Training and
testing split is included in the file. The CAD models are completely cleaned
inhouse by ourselves.

[New!] Aligned 40-Class Subset
==============================

Now You can find Aligned 40-Class ModelNet models `Here
<https://github.com/lmb-freiburg/orion>`__. This data is provided by N.
Sedaghat, M. Zolfaghari, E. Amiri and T. Brox authors of Orientation-boosted
Voxel Nets for 3D Object Recognition [8].

The CAD models are in `Object File Format (OFF)
<http://segeval.cs.princeton.edu/public/off_format.html>`__. We also provide
Matlab functions to read and visualize OFF files in our `Princeton Vision
Toolkit (PVT) <http://3dvision.princeton.edu/code.html>`__.

******************************
ModelNet Benchmark Leaderboard
******************************

Please email `Shuran Song <http://shurans.github.io/>`__ to add or update your
results.

In your email please provide following information in this format:

| Algorithm Name, ModelNet40 Classification, ModelNet40 Retrieval, ModelNet10
  Classification, ModelNet10 Retrieval
| Author list, Paper title, Conference.  Link to paper.

Example:

| 3D-DescriptorNet, -, -, -,92.4%,-
| Jianwen Xie, Zilong Zheng, Ruiqi Gao, Wenguan Wang, Song-Chun Zhu, and Ying
  Nian Wu, Learning Descriptor Networks for 3D Shape Synthesis and Analysis. CVPR
  2018, http://...

+-------------+-------------+-------------+-------------+-------------+
| Algorithm   | ModelNet40  | ModelNet40  | ModelNet10  | ModelNet10  |
|             | Cla         | Retrieval   | Cla         | Retrieval   |
|             | ssification | (mAP)       | ssification | (mAP)       |
|             | (Accuracy)  |             | (Accuracy)  |             |
+-------------+-------------+-------------+-------------+-------------+
| RS-CNN[63]  | 93.6%       | -           | -           | -           |
+-------------+-------------+-------------+-------------+-------------+
| L           | 92.1%       | -           | 94.4%       | -           |
| P-3DCNN[62] |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| LDGCNN[61]  | 92.9%       | -           | -           | -           |
+-------------+-------------+-------------+-------------+-------------+
| Primit      | 86.4%       | -           | 92.2%       | -           |
| ive-GAN[60] |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| 3DCapsule   | 92.7%       | -           | 94.7%       | -           |
| [59]        |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| 3D2SeqViews | 93.40%      | 90.76%      | 94.71%      | 92.12%      |
| [58]        |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| Orth        | -           | -           | 88.56%      | 86.85%      |
| ographicNet |             |             |             |             |
| [57]        |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| Ma et al.   | 91.05%      | 84.34%      | 95.29%      | 93.19%      |
| [56]        |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| MLVCNN [55] | 94.16%      | 92.84%      | -           | -           |
+-------------+-------------+-------------+-------------+-------------+
| iMHL [54]   | 97.16%      | -           | -           | -           |
+-------------+-------------+-------------+-------------+-------------+
| HGNN [53]   | 96.6%       | -           | -           | -           |
+-------------+-------------+-------------+-------------+-------------+
| SPNet [52]  | 92.63%      | 85.21%      | 97.25%      | 94.20%      |
+-------------+-------------+-------------+-------------+-------------+
| MHBN [51]   | 94.7        | -           | 95.0        | -           |
+-------------+-------------+-------------+-------------+-------------+
| VIPGAN [50] | 91.98       | 89.23       | 94.05       | 90.69       |
+-------------+-------------+-------------+-------------+-------------+
| Poi         | 92.60       | -           | 95.30       | -           |
| nt2Sequence |             |             |             |             |
| [49]        |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| Tri         | -           | 88.0%       | -           | -           |
| plet-Center |             |             |             |             |
| Loss [48]   |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| PVNet[47]   | 93.2%       | 89.5%       | -           | -           |
+-------------+-------------+-------------+-------------+-------------+
| GVCNN[46]   | 93.1%       | 85.7%       | -           | -           |
+-------------+-------------+-------------+-------------+-------------+
| MLH-MV[45]  | 93.11%      |             | 94.80%      |             |
+-------------+-------------+-------------+-------------+-------------+
| MV          | 95.0%       |             |             |             |
| CNN-New[44] |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| SeqViews2Se | 93.40%      | 89.09%      | 94.82%      | 91.43%      |
| qLabels[43] |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| G3DNet[42]  | 91.13%      |             | 93.1%       |             |
+-------------+-------------+-------------+-------------+-------------+
| VSL [41]    | 84.5%       |             | 91.0%       |             |
+-------------+-------------+-------------+-------------+-------------+
| 3D-C        | 82.73%      | 70.1%       | 93.08%      | 88.44%      |
| apsNets[40] |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| KCNet[39]   | 91.0%       |             | 94.4%       |             |
+-------------+-------------+-------------+-------------+-------------+
| Fol         | 88.4%       |             | 94.4%       |             |
| dingNet[38] |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| binVox      | 85.47%      |             | 92.32%      |             |
| NetPlus[37] |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| D           | 90.3%       |             |             |             |
| eepSets[36] |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| 3D-Descri   |             |             | 92.4%       |             |
| ptorNet[35] |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| SO-Net[34]  | 93.4%       |             | 95.7%       |             |
+-------------+-------------+-------------+-------------+-------------+
| Minto et    | 89.3%       |             | 93.6%       |             |
| al.[33]     |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| Rota        | 97.37%      |             | 98.46%      |             |
| tionNet[32] |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| Lo          |             |             | 94.37       |             |
| nchaNet[31] |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| Achlioptas  | 84.5%       |             | 95.4%       |             |
| et al. [30] |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| P           | 95.56%      | 86.34%      | 96.85%      | 93.28%      |
| ANORAMA-ENN |             |             |             |             |
| [29]        |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| 3D-A-Nets   | 90.5%       | 80.1%       |             |             |
| [28]        |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| Soltani et  | 82.10%      |             |             |             |
| al. [27]    |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| Arvind et   | 86.50%      |             |             |             |
| al. [26]    |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| LonchaNet   |             |             | 94.37%      |             |
| [25]        |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| 3DmFV-Net   | 91.6%       |             | 95.2%       |             |
| [24]        |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| Zanuttigh   | 87.8%       |             | 91.5%       |             |
| and Minto   |             |             |             |             |
| [23]        |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| Wang et al. | 93.8%       |             |             |             |
| [22]        |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| ECC [21]    | 83.2%       |             | 90.0%       |             |
+-------------+-------------+-------------+-------------+-------------+
| PANORAMA-NN | 90.7%       | 83.5%       | 91.1%       | 87.4%       |
| [20]        |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| MVC         | 91.4%       |             |             |             |
| NN-MultiRes |             |             |             |             |
| [19]        |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| FPNN [18]   | 88.4%       |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| P           | 89.2%       |             |             |             |
| ointNet[17] |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| Klokov and  | 91.8%       |             | 94.0%       |             |
| Le          |             |             |             |             |
| mpitsky[16] |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| L           | 88.93%      |             | 93.94%      |             |
| ightNet[15] |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| Xu and      | 81.26%      |             | 88.00%      |             |
| To          |             |             |             |             |
| dorovic[14] |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| Geometry    | 83.9%       | 51.3%       | 88.4%       | 74.9%       |
| Image [13]  |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| Set-        | 90%         |             |             |             |
| convolution |             |             |             |             |
| [11]        |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| PointNet    |             |             | 77.6%       |             |
| [12]        |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| 3D-GAN [10] | 83.3%       |             | 91.0%       |             |
+-------------+-------------+-------------+-------------+-------------+
| VRN         | 95.54%      |             | 97.14%      |             |
| Ensemble    |             |             |             |             |
| [9]         |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| ORION [8]   |             |             | 93.8%       |             |
+-------------+-------------+-------------+-------------+-------------+
| FusionNet   | 90.8%       |             | 93.11%      |             |
| [7]         |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| Pairwise    | 90.7%       |             | 92.8%       |             |
| [6]         |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| MVCNN [3]   | 90.1%       | 79.5%       |             |             |
+-------------+-------------+-------------+-------------+-------------+
| GIFT [5]    | 83.10%      | 81.94%      | 92.35%      | 91.12%      |
+-------------+-------------+-------------+-------------+-------------+
| VoxNet [2]  | 83%         |             | 92%         |             |
+-------------+-------------+-------------+-------------+-------------+
| DeepPano    | 77.63%      | 76.81%      | 85.45%      | 84.18%      |
| [4]         |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| 3DShapeNets | 77%         | 49.2%       | 83.5%       | 68.3%       |
| [1]         |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+

[1] Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang and J. Xiao.  `3D
     ShapeNets: A Deep Representation for Volumetric Shapes
     <http://3dshapenets.cs.princeton.edu>`__. CVPR2015.

[2] D. Maturana and S. Scherer. `VoxNet: A 3D Convolutional Neural Network for
     Real-Time Object Recognition
     <http://danielmaturana.net/extra/voxnet_maturana_scherer_iros15.pdf>`__.
     IROS2015.

[3] H. Su, S. Maji, E. Kalogerakis, E. Learned-Miller. `Multi-view
     Convolutional Neural Networks for 3D Shape Recognition
     <http://people.cs.umass.edu/~kalo/papers/viewbasedcnn/index.html>`__.
     ICCV2015.

[4] B Shi, S Bai, Z Zhou, X Bai. `DeepPano: Deep Panoramic Representation for
     3-D Shape Recognition
     <http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=7273863&tag=1>`__.
     Signal Processing Letters 2015.

[5] Song Bai, Xiang Bai, Zhichao Zhou, Zhaoxiang Zhang, Longin Jan Latecki.
 `GIFT: A Real-time and Scalable 3D Shape Search Engine.
 <https://sites.google.com/site/songbaihust/>`__ CVPR 2016.

[6] Edward Johns, Stefan Leutenegger and Andrew J. Davison. `Pairwise
     Decomposition of Image Sequences for Active Multi-View Recognition
     <https://www.doc.ic.ac.uk/~ejohns/Documents/ejohns-multi-view-recognition-cvpr2016.pdf>`__
     CVPR 2016.

[7] Vishakh Hegde, Reza Zadeh `3D Object Classification Using Multiple Data
     Representations <http://arxiv.org/abs/1607.05695>`__.

[8] Nima Sedaghat, Mohammadreza Zolfaghari, Thomas Brox `Orientation-boosted
    Voxel Nets for 3D Object Recognition. <http://arxiv.org/abs/1604.03351>`__ BMVC

[9] Andrew Brock, Theodore Lim, J.M. Ritchie, Nick Weston `Generative and
     Discriminative Voxel Modeling with Convolutional Neural Networks
     <https://arxiv.org/abs/1608.04236>`__.

[10] Jiajun Wu, Chengkai Zhang, Tianfan Xue, William T. Freeman, Joshua B.
     Tenenbaum. `Learning a Probabilistic Latent Space of Object Shapes via 3D
     Generative-Adversarial Modeling. <https://arxiv.org/pdf/1610.07584.pdf>`__ NIPS
     2016

[11] Siamak Ravanbakhsh, Jeff Schneider, Barnabas Poczos. `Deep Learning with
     sets and point clouds <https://arxiv.org/abs/1611.04500>`__

[12] A. Garcia-Garcia, F. Gomez-Donoso†, J. Garcia-Rodriguez, S.
     Orts-Escolano, M. Cazorla, J. Azorin-Lopez. `PointNet: A 3D Convolutional
     Neural Network for Real-Time Object Class Recognition
     <http://ieeexplore.ieee.org/document/7727386//>`__

[13] Ayan Sinha, Jing Bai, Karthik Ramani. `Deep Learning 3D Shape Surfaces
     Using Geometry Images
     <http://link.springer.com/chapter/10.1007/978-3-319-46466-4_14>`__ ECCV 2016

[14] Xu Xu and Sinisa Todorovic. `Beam Search for Learning a Deep Convolutional
     Neural Network of 3D Shapes <https://arxiv.org/pdf/1612.04774v1.pdf>`__

[15] Shuaifeng Zhi, Yongxiang Liu, Xiang Li, Yulan Guo `Towards real-time 3D
     object recognition: A lightweight volumetric CNN framework using multitask
     learning <https://arxiv.org/pdf/1612.04774v1.pdf>`__ Computers and Graphics
     (Elsevier)

[16] Roman Klokov, Victor Lempitsky `Escape from Cells: Deep Kd-Networks for
     The Recognition of 3D Point Cloud Models <https://arxiv.org/abs/1704.01222>`__

[17] Charles R. Qi, Hao Su, Kaichun Mo, and Leonidas J. Guibas.  `PointNet:
     Deep Learning on Point Sets for 3D Classification and Segmentation.
     <https://arxiv.org/abs/1612.00593>`__ CVPR 2017.

[18] Yangyan Li, Soeren Pirk, Hao Su, Charles R. Qi, and Leonidas J.  Guibas.
     `FPNN: Field Probing Neural Networks for 3D Data.
     <https://arxiv.org/abs/1605.06240>`__ NIPS 2016.

[19] Charles R. Qi, Hao Su, Matthias Niessner, Angela Dai, Mengyuan Yan, and
     Leonidas J. Guibas. `Volumetric and Multi-View CNNs for Object Classification
     on 3D Data. <https://arxiv.org/abs/1604.03265>`__ CVPR 2016.

[20] K. Sfikas, T. Theoharis and I. Pratikakis. `Exploiting the PANORAMA
     Representation for Convolutional Neural Network Classification and Retrieval.
     <https://diglib.eg.org/handle/10.2312/3dor20171045>`__ 3DOR2017.

[21] Martin Simonovsky, Nikos Komodakis `Dynamic Edge-Conditioned Filters in
     Convolutional Neural Networks on Graphs. <https://arxiv.org/abs/1704.02901>`__

[22] Chu Wang, Marcello Pelillo, Kaleem Siddiqi. `Dominant Set Clustering and
     Pooling for Multi-View 3D Object Recognition.
     <http://www.cim.mcgill.ca/~chuwang/files/bmvc2017/0253.pdf>`__ BMVC 2017.

[23] Pietro Zanuttigh and Ludovico Minto `Deep Learning for 3D Shape
     Classification from Multiple Depth Maps
     <https://www2.securecms.com/ICIP2017/Papers/ViewPapers.asp?PaperNum=2413>`__
     ICIP 2017.

[24] Yizhak Ben-Shabat, Michael Lindenbaum, Anath Fischer `3D Point Cloud
     Classification and Segmentation using 3D Modified Fisher Vector Representation
     for Convolutional Neural Networks <https://arxiv.org/abs/1711.08241>`__ arXiv
     2017.

[25] F. Gomez-Donoso, A. Garcia-Garcia, J. Garcia-Rodriguez, S.  Orts-Escolano,
     M. Cazorla `LonchaNet: A sliced-based CNN architecture for real-time 3D object
     recognition <http://ieeexplore.ieee.org/document/7965883/>`__ Neural Networks
     (IJCNN), 2017.

[26] Varun Arvind, Anthony Costa, Marcus Badgeley, Samuel Cho, Eric Oermann
     `Wide and deep volumetric residual networks for volumetric image classification
     <https://arxiv.org/abs/1710.01217>`__ arXiv 2017.

[27] Amir Arsalan Soltani, Haibin Huang, Jiajun Wu, Tejas D. Kulkarni, Joshua
     B. Tenenbaum `Synthesizing 3D Shapes via Modeling Multi-View Depth Maps and
     Silhouettes with Deep Generative Networks
     <https://github.com/Amir-Arsalan/Synthesize3DviaDepthOrSil>`__ CVPR 2017

[28] Mengwei Ren, Liang Niu, Yi Fang `3D-A-Nets: 3D Deep Dense Descriptor for
     Volumetric Shapes with Adversarial Networks
     <https://arxiv.org/abs/1711.10108>`__

[29] K Sfikas, I Pratikakis and T Theoharis, `Ensemble of PANORAMA-based
     Convolutional Neural Networks for 3D Model Classification and Retrieval
     <https://doi.org/10.1016/j.cag.2017.12.001>`__ Computers and Graphics

[30] Panos Achlioptas, Olga Diamanti, Ioannis Mitliagkas, Leonidas Guibas.
     `Learning Representations and Generative Models for 3D Point Clouds
     <https://doi.org/10.1016/j.cag.2017.12.001>`__, arXiv 2017

[31] F. Gomez-Donoso, A. Garcia-Garcia, J. Garcia-Rodriguez, S.  Orts-Escolano,
     M. Cazorla. `LonchaNet: A sliced-based CNN architecture for real-time 3D object
     recognition" <http://ieeexplore.ieee.org/document/7965883/>`__

[32] Asako Kanezaki, Yasuyuki Matsushita and Yoshifumi Nishida.  `RotationNet:
     Joint Object Categorization and Pose Estimation Using Multiviews from
     Unsupervised Viewpoints. CVPR, 2018. <https://arxiv.org/abs/1603.06208>`__

[33] L. Minto ,P. Zanuttigh, G. Pagnutti `Deep Learning for 3D Shape
     Classification Based on Volumetric Density and Surface Approximation Clues,
     International Conference on Computer Vision Theory and Applications (VISAPP),
     2018
     <http://lttm.dei.unipd.it/nuovo/Papers/18_VISAPP_3d_multi_classification.pdf>`__

[34] J. Li, B. M. Chen, G. H. Lee `SO-Net: Self-Organizing Network for Point
     Cloud Analysis. <https://github.com/lijx10/SO-Net>`__ CVPR2018

[35] Jianwen Xie, Zilong Zheng, Ruiqi Gao, Wenguan Wang, Song-Chun Zhu, and
     Ying Nian Wu, `Learning Descriptor Networks for 3D Shape Synthesis and
     Analysis.
     <http://www.stat.ucla.edu/~jxie/3DDescriptorNet/3DDescriptorNet.html>`__ CVPR
     2018

[36] Manzil Zaheer, Satwik Kottur, Siamak Ravanbhakhsh, Barnabás Póczos, Ruslan
     Salakhutdinov1, Alexander J Smola, `Deep Sets.
     <https://papers.nips.cc/paper/6931-deep-sets.pdf>`__ NIPS 2018

[37] Chao Ma, Wei An, Yinjie Lei, Yulan Guo. BV-CNNs: Binary volumetric
     convolutional neural networks for 3D object recognition.  BMVC2017.  Chao Ma,
     Yulan Guo, Yinjie Lei, Wei An. Binary volumetric convolutional neural networks
     for 3D object recognition. IEEE Transactions on Instrumentation & Measurement.

[38] Yaoqing Yang, Chen Feng, Yiru Shen, Dong Tian, `FoldingNet: Point Cloud
     Auto-encoder via Deep Grid Deformation.
     <https://arxiv.org/pdf/1712.07262.pdf>`__ CVPR 2018

[39] Yiru Shen, Chen Feng, Yaoqing Yang and Dong Tian `Mining Point Cloud Local
     Structures by Kernel Correlation and Graph Pooling.
     <https://arxiv.org/pdf/1712.06760.pdf>`__ CVPR 2018

[40] Ryan Lambert, `Capsule Nets for Content Based 3D Model Retrieval
     <https://github.com/Ryanglambert/3d_model_retriever>`__

[41] Shikun Liu, C. Lee Giles, and Alexander G. Ororbia II. `Learning a
     Hierarchical Latent-Variable Model of 3D Shapes
     <%20http://shikun.io/papers/vsl.html>`__ 3DV 2018

[42] Miguel Dominguez,Rohan Dhamdhere,Atir Petkar,Saloni Jain, Shagan Sah,
     Raymond Ptucha, `General-Purpose Deep Point Cloud Feature Extractor.
     <https://ieeexplore.ieee.org/document/8354322/>`__ WACV 2018.

[43] Zhizhong Han, Mingyang Shang, Zhenbao Liu, Chi-Man Vong, Yu-Shen Liu,
     Junwei Han, Matthias Zwicker, C.L. Philip Chen.  `SeqViews2SeqLabels: Learning
     3D Global Features via Aggregating Sequential Views by RNN with Attention.
     <%20http://cgcad.thss.tsinghua.edu.cn/liuyushen/SeqViews2SeqLabels/index.html>`__.
     IEEE Transactions on Image Processing, 2019, 28(2): 658-672.

[44] Jong-Chyi Su, Matheus Gadelha, Rui Wang, and Subhransu Maji. `A Deeper
     Look at 3D Shape Classifiers <https://arxiv.org/abs/1809.02560>`__ . Second
     Workshop on 3D Reconstruction Meets Semantics, ECCV 2018.

[45] Kripasindhu Sarkar, Basavaraj Hampiholi, Kiran Varanasi, Didier Stricker,
     `Learning 3D Shapes as Multi-Layered Height-maps using 2D Convolutional
     Networks
     <http://openaccess.thecvf.com/content_ECCV_2018/html/Kripasindhu_Sarkar_Learning_3D_shapes_ECCV_2018_paper.html>`__
     ECCV 2018.

[46] Yifan Feng, Zizhao Zhang, Xibin Zhao, Rongrong Ji, Yue Gao.  `GVCNN:
     Group-View Convolutional Neural Networks for 3D Shape Recognition
     <http://openaccess.thecvf.com/content_cvpr_2018/papers/Feng_GVCNN_Group-View_Convolutional_CVPR_2018_paper.pdf>`__
     CVPR 2018.

[47] Haoxuan You, Yifan Feng, Rongrong Ji, Yue Gao. `PVNet: A Joint
     Convolutional Network of Point Cloud and Multi-View for 3D Shape Recognition
     <https://arxiv.org/pdf/1808.07659.pdf>`__ ACM MM 2018.

[48] Xinwei He, Yang Zhou, Zhichao Zhou, Song Bai, and Xiang Bai,.
     `Triplet-Center Loss for Multi-View 3D Object Retrieval, CVPR 2018
     <http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1632.pdf>`__

[49] Xinhai Liu, Zhizhong Han, Yu-Shen Liu, Matthias Zwicker.  `Point2Sequence:
     Learning the Shape Representation of 3D Point Clouds with an Attention-based
     Sequence to Sequence Network. AAAI 2019.
     <http://cgcad.thss.tsinghua.edu.cn/liuyushen/Point2Sequence>`__

[50] Zhizhong Han, Mingyang Shang, Yu-Shen Liu, Matthias Zwicker. `View
     Inter-Prediction GAN: Unsupervised Representation Learning for 3D Shapes by
     Learning Global Shape Memories to Support Local View Predictions. AAAI 2019.
     <%20http://cgcad.thss.tsinghua.edu.cn/liuyushen/VIPGAN>`__

[51] Yu, Tan, Jingjing Meng, and Junsong Yuan. `"Multi-view Harmonized Bilinear
     Network for 3D Object Recognition." CVPR 2018.
     <http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0142.pdf>`__

[52] Mohsen Yavartanoo, Eu Young Kim, Kyoung Mu Lee, `SPNet: Deep 3D Object
     Classification and Retrieval using Stereographic Projection, ACCV2018.
     <https://arxiv.org/abs/1811.01571>`__

[53] Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong Ji, Yue Gao, `Hypergraph
     Neural Networks. <http://gaoyue.org/paper/HGNN.pdf>`__ AAAI2019.

[54] Zizhao Zhang, Haojie Lin, Xibin Zhao, Rongrong Ji, and Yue Gao, `Inductive
     Multi-Hypergraph Learning and Its Application on View-Based 3D Object
     Classification. <https://ieeexplore.ieee.org/document/8424480%0A>`__
     Transactions on Image Processing. 2018.

[55] Jianwen Jiang, Di Bao, Ziqiang Chen, Xibin Zhao, Yue Gao `MLVCNN:
     Multi-Loop-View Convolutional Neural Network for 3D Shape Retrieval
     <http://imoonlab.org/paper/mlvcnn.pdf%0A>`__\ AAAI 2019

[56] Chao Ma, Yulan Guo*, Jungang Yang, Wei An. `Learning Multi-view
     Representation with LSTM for 3D Shape Recognition and Retrieval.
     <https://ieeexplore.ieee.org/document/8490588>`__ IEEE Transactions on
     Multimedia, 2018.IEEE Transactions on Multimedia, 2018.

[57] Hamidreza Kasaei `OrthographicNet: A Deep Learning Approach for 3D Object
     Recognition in Open-Ended Domains. <https://arxiv.org/abs/1902.03057>`__ arXiv
     2019.

[58] Zhizhong Han, Honglei Lu, Zhenbao Liu, Chi-Man Vong, Yu-Shen Liu, Matthias
     Zwicker, Junwei Han, C.L. Philip Chen. `3D2SeqViews: Aggregating Sequential
     Views for 3D Global Feature Learning by CNN with Hierarchical Attention
     Aggregation. <https://ieeexplore.ieee.org/document/8666059>`__ IEEE
     Transactions on Image Processing, 2019

[59] A. Cheraghian and L. Petersson, `3DCapsule: Extending the Capsule
     Architecture to Classify 3D Point Clouds,
     <https://ieeexplore.ieee.org/document/8658405>`__ 2019 IEEE Winter Conference
     on Applications of Computer Vision (WACV), 2019, pp.  1194-1202.

[60] Salman H. Khan, Yulan Guo, Munawar Hayat, Nick Barnes, `"Unsupervised
     Primitive Discovery for Improved 3D Generative Modeling"
     <https://salman-h-khan.github.io/papers/CVPR19_2.pdf>`__ CVPR 2019.

[61] Kuangen Zhang, Ming Hao, Jing Wang, Clarence W. de Silva, Chenglong Fu,
     `Linked Dynamic Graph CNN: Learning on Point Cloud via Linking Hierarchical
     Features. <https://arxiv.org/abs/1904.10014>`__ arXiv.

[62] Sudhakar Kumawat and Shanmuganathan Raman, `LP-3DCNN: Unveiling Local
     Phase in 3D Convolutional Neural Networks.
     <https://arxiv.org/pdf/1904.03498.pdf>`__ CVPR 2019.

[63] Yongcheng Liu, Bin Fan, Shiming Xiang, Chunhong Pan.  `Relation-Shape
     Convolutional Neural Network for Point Cloud Analysis.
     <https://arxiv.org/abs/1904.07601>`__ CVPR 2019

Download Full Dataset
=====================

Please email Shuran Song to obtain the Matlab toolbox for downloading.

Citation
========

If you find this dataset useful, please cite the following paper:

| Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang and J. Xiao
| `3D ShapeNets: A Deep Representation for Volumetric Shapes <htt
p://3dvision.princeton.edu/projec ts/2014/3DShapeNets/paper.pdf>`__
| Proceedings of 28th IEEE Conference on Computer Vision and Pattern
Recognition (**CVPR2015**)
| **Oral Presentation** · `3D Deep Learning Project Web page
<http://3dvision.princeton.e du/projects/2014/3DShapeNets/>`__

Copyright
=========

All CAD models are downloaded from the Internet and the original authors hold
the copyright of the CAD models. The label of the data was obtained by us via
Amazon Mechanical Turk service and it is provided freely. This dataset is
provided for the convenience of academic research only.
