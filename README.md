# Few-shot face translation
A GAN based approach for one model to swap them all.

The following figures illustrate our priliminary faceswapping results on random identities drawn from [VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) dataset. We also show failure cases, which demonstrate limitations of our model for genrerating faces with consistent skin tone, eye-glasses, and expression.

![](https://github.com/shaoanlu/faceswap-GAN-swap-them-all/raw/master/images/result_iter40k_00.jpg)
![](https://github.com/shaoanlu/faceswap-GAN-swap-them-all/raw/master/images/result_iter40k_01.jpg)

Our model is still very data sentitive, i.e., when feeding identities that the model had never seem before or are outside training data distribution (such as asian faces that are scarce in VGGFace2), our GAN failed at genereating faces with high fidelity.

![](https://github.com/shaoanlu/faceswap-GAN-swap-them-all/raw/master/images/result2.jpg)

## Architecture
![](https://github.com/shaoanlu/faceswap-GAN-swap-them-all/raw/master/images/few_shot_face_translation.png)

The above image illustrates our generator, which is a encoder-decoder based network, at test phase. Our swap-them-all approach is basically a GAN conditioned on the latent embeddings extracted from a pre-trained face recognition model. [SPADE](https://arxiv.org/abs/1903.07291) and [AdaIN](https://arxiv.org/abs/1905.01723) modules are incorporated in order to inject semantic priors to the networks. 

During training phase, the input face A is heavily blurred and we train the model with resonctruction loss. Other objectives that aimed to improve translation performance while keeping semantic consistency, e.g., perceptual loss on rgb output and cosine similarity loss on laten embeddings, are also introduced.

### Things that didn't work

1. We tried to distort (spline warp, downsample) the input image as in [faceswap-GAN](https://github.com/shaoanlu/faceswap-GAN) instead of masking it. However, the model did not learn proper identity translation but output face that is similar to its input.

## References
1. [Semantic Image Synthesis with Spatially-Adaptive Normalization](https://github.com/shaoanlu/faceswap-GAN)
2. [Few-Shot Unsupervised Image-to-Image Translation](https://arxiv.org/abs/1905.01723)

