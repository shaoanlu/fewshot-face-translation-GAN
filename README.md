# faceswap-GAN-swap-them-all
A GAN based approach for one model to swap them all.

The following figure illustrates our priliminary faceswapping results on random identities drawn from [VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) dataset (from left to right: source, target, result). We also show some failure cases, which demonstrate limitations of our model for genrerating consistent skin tone, eye-glasses, and expression.

![](https://github.com/shaoanlu/faceswap-GAN-swap-them-all/raw/master/images/result.jpg)

Our model is still very sentitive, e.g., when feeding identities that the model had never seem before or are outside training data distribution (such as asian faces that are scarce in VGGFace2), our GAN could not genereate faces with high fidelity.

![](https://github.com/shaoanlu/faceswap-GAN-swap-them-all/raw/master/images/result2.jpg)
