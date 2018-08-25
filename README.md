
## A Taxonomy of Current Methods

### 1. "Slow" Neural Methods Based On Online Image Optimization

###  1.1. Parametric "Slow" Neural Methods with Summary Statistics

:white_check_mark: [**Image Style Transfer Using Convolutional Neural Networks**] [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) *(CVPR 2016)*

:white_check_mark: [**Incorporating Long-range Consistency in CNN-based Texture Generation**] [[Paper]](https://arxiv.org/pdf/1606.01286.pdf)  *(ICLR 2017)* 

:sparkle: **Code:**

*   [Theano-based](https://github.com/guillaumebrg/texture_generation)

:white_check_mark: [**Laplacian-Steered Neural Style Transfer**] [[Paper]](https://arxiv.org/pdf/1707.01253.pdf)  *(ACM MM 2017)*

:sparkle: **Code:**

*   [Torch-based & TensorFlow-based](https://github.com/askerlee/lapstyle)

:white_check_mark: [**Demystifying Neural Style Transfer**] [[Paper]](https://arxiv.org/pdf/1701.01036.pdf)  *(Theoretical Explanation)* *(IJCAI 2017)*

:sparkle: **Code:**

*   [MXNet-based](https://github.com/lyttonhao/Neural-Style-MMD)


###  1.2. Non-parametric "Slow" Neural Methods with MRFs

:white_check_mark: [**Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis**] [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Li_Combining_Markov_Random_CVPR_2016_paper.pdf)  *(CVPR 2016)*

:sparkle: **Code:**

*   [Torch-based](https://github.com/chuanli11/CNNMRF)


:white_check_mark: [**Arbitrary Style Transfer with Deep Feature Reshuffle**] [[Paper]](https://arxiv.org/pdf/1805.04103.pdf)  *(CVPR 2018)*

###  2. "Fast" Neural Methods Based On Offline Model Optimization

### 2.1. Per-Style-Per-Model "Fast" Neural Methods


:white_check_mark: [**Perceptual Losses for Real-Time Style Transfer and Super-Resolution**] [[Paper]](https://arxiv.org/pdf/1603.08155.pdf)  *(ECCV 2016)*

:sparkle: **Code:**

*   [Torch-based](https://github.com/jcjohnson/fast-neural-style)
*   [TensorFlow-based](https://github.com/lengstrom/fast-style-transfer)
*   [Chainer-based](https://github.com/yusuketomoto/chainer-fast-neuralstyle)

:sparkle: **Pre-trained Models:**

*   [Torch-models](https://github.com/ProGamerGov/Torch-Models)
*   [Chainer-models](https://github.com/gafr/chainer-fast-neuralstyle-models)


:white_check_mark: [**Texture Networks: Feed-forward Synthesis of Textures and Stylized Images**] [[Paper]](http://www.jmlr.org/proceedings/papers/v48/ulyanov16.pdf)  *(ICML 2016)*

:sparkle: **Code:**

*   [Torch-based](https://github.com/DmitryUlyanov/texture_nets)
*   [TensorFlow-based](https://github.com/tgyg-jegli/tf_texture_net)


:white_check_mark: [**Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks**] [[Paper]](https://arxiv.org/pdf/1604.04382.pdf)  *(ECCV 2016)*

:sparkle: **Code:**

*   [Torch-based](https://github.com/chuanli11/MGANs)




### 2.2. Multiple-Style-Per-Model "Fast" Neural Methods

:white_check_mark: [**A Learned Representation for Artistic Style**] [[Paper]](https://arxiv.org/pdf/1610.07629.pdf)  *(ICLR 2017)*

:sparkle: **Code:**

*   [TensorFlow-based](https://github.com/tensorflow/magenta/tree/master/magenta/models/image_stylization)

:sparkle: **Code:**

*   [PyTorch-based](https://github.com/zhanghang1989/PyTorch-Style-Transfer)
*   [Torch-based](https://github.com/zhanghang1989/MSG-Net)

:white_check_mark: [**Diversified Texture Synthesis With Feed-Forward Networks**] [[Paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Diversified_Texture_Synthesis_CVPR_2017_paper.pdf)  *(CVPR 2017)* 

:sparkle: **Code:**

*   [Torch-based](https://github.com/Yijunmaverick/MultiTextureSynthesis)

:white_check_mark: [**StyleBank: An Explicit Representation for Neural Image Style Transfer**] [[Paper]](https://arxiv.org/pdf/1703.09210.pdf)  *(CVPR 2017)*



### 2.3. Arbitrary-Style-Per-Model "Fast" Neural Methods

:white_check_mark: [**Fast Patch-based Style Transfer of Arbitrary Style**] [[Paper]](https://arxiv.org/pdf/1612.04337.pdf) *(NIPSW 2016)*

:sparkle: **Code:**

*   [Torch-based](https://github.com/rtqichen/style-swap)

:white_check_mark: [**Exploring the Structure of a Real-time, Arbitrary Neural Artistic Stylization Network**] [[Paper]](https://arxiv.org/pdf/1705.06830.pdf)  *(BMVC 2017)*

:sparkle: **Code:**

*   [TensorFlow-based](https://github.com/tensorflow/magenta/tree/master/magenta/models/arbitrary_image_stylization)


:white_check_mark: [**Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization**] [[Paper]](https://arxiv.org/pdf/1703.06868.pdf)  *(ICCV 2017)*

:sparkle: **Code:**

*   [Torch-based](https://github.com/xunhuang1995/AdaIN-style)
*   [TensorFlow-based with Keras](https://github.com/eridgd/AdaIN-TF)
*   [TensorFlow-based without Keras](https://github.com/elleryqueenhomels/arbitrary_style_transfer)

:white_check_mark: [**Universal Style Transfer via Feature Transforms**] [[Paper]](https://arxiv.org/pdf/1705.08086.pdf)  *(NIPS 2017)*

:sparkle: **Code:**

*   [Torch-based](https://github.com/Yijunmaverick/UniversalStyleTransfer)
*   [TensorFlow-based](https://github.com/eridgd/WCT-TF)
*   [PyTorch-based](https://github.com/sunshineatnoon/PytorchWCT)

:white_check_mark: [**Meta Networks for Neural Style Transfer**] [[Paper]](https://arxiv.org/pdf/1709.04111.pdf)  *(CVPR 2018)*

:sparkle: **Code:**

*   [Caffe-based](https://github.com/FalongShen/styletransfer)

:white_check_mark: [**Avatar-Net: Multi-Scale Zero-Shot Style Transfer by Feature Decoration**] [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0137.pdf)  *(CVPR 2018)*

:sparkle: **Code:**

*   [TensorFlow-based](https://github.com/LucasSheng/avatar-net)


## Improvements and Extensions

:white_check_mark: [**Controlling Perceptual Factors in Neural Style Transfer**] [[Paper]](https://arxiv.org/pdf/1611.07865.pdf)  *(CVPR 2017)* 

:sparkle: **Code:**

*   [Torch-based](https://github.com/leongatys/NeuralImageSynthesis)

:white_check_mark: [**Towards Deep Style Transfer: A Content-Aware Perspective**] [[Paper]](http://www.bmva.org/bmvc/2016/papers/paper008/paper008.pdf)  *(BMVC 2016)*

:white_check_mark: [**The Contextual Loss for Image Transformation with Non-Aligned Data**] [[Paper]](https://arxiv.org/pdf/1803.02077)   *(ECCV 2018)* 

:sparkle: **Code:**

*   [TensorFlow-based](https://github.com/roimehrez/contextualLoss)

:white_check_mark: [**Improved Texture Networks: Maximizing Quality and Diversity in Feed-forward Stylization and Texture Synthesis**] [[Paper]](https://arxiv.org/pdf/1701.02096.pdf)  *(CVPR 2017)*

:sparkle: **Code:**

*   [Torch-based](https://github.com/DmitryUlyanov/texture_nets)

:white_check_mark: [**A Style-Aware Content Loss for Real-time HD Style Transfer**] [[Paper]](https://arxiv.org/pdf/1807.10201)  *(ECCV 2018)* 

:white_check_mark: [**Multimodal Transfer: A Hierarchical Deep Convolutional Neural Network for Fast Artistic Style Transfer**] [[Paper]](https://arxiv.org/pdf/1612.01895.pdf)  *(CVPR 2017)* 

:sparkle: **Code:**

*   [TensorFlow-based](https://github.com/fullfanta/multimodal_transfer)

:white_check_mark: [**Stroke Controllable Fast Style Transfer with Adaptive Receptive Fields**] [[Paper]](https://arxiv.org/pdf/1802.07101.pdf)  *(ECCV 2018)* 

:sparkle: **Code:**

*   [TensorFlow-based](https://github.com/LouieYang/stroke-controllable-fast-style-transfer)

:white_check_mark:   [**Separating Style and Content for Generalized Style Transfer**] [[Paper]](https://arxiv.org/pdf/1711.06454.pdf)  *(CVPR 2018)*  

:white_check_mark: [**Visual Attribute Transfer through Deep Image Analogy**] [[Paper]](https://arxiv.org/pdf/1705.01088.pdf)  *(SIGGRAPH 2017)*

:sparkle: **Code:**

*   [Caffe-based](https://github.com/msracver/Deep-Image-Analogy)

*   [Torch-based](https://github.com/luanfujun/deep-painterly-harmonization)

:white_check_mark: [**Learning Selfie-Friendly Abstraction from Artistic Style Images**] [[Paper]]()  *(ACML 2018)*
