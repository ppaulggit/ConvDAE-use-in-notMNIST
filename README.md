
# ConvDAE-use-in-notMNIST
Win10 Python3.5 Tensorflow-1.1.0-gpu

This is a ConvDAE using in notMNIST dataset to denoise.

The ConvDAE you can see in https://github.com/NELSONZHAO/zhihu/tree/master/denoise_auto_encoder.            
The difference is the loss function, we use the l2_loss, not the sigmoid_cross_entropy.


And the notMNIST dataset you can find in https://github.com/hankcs/udacity-deep-learning.

Here is the result.
The first row is add noisy images, the second row is the denoise images after 
ConvDAE processing, the third row is the original images.
![image](https://github.com/PaulGitt/ConvDAE-use-in-notMNIST/blob/master/Result.png)
