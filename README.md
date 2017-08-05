
# ConvDAE-use-in-notMNIST
Win10 Python3.5 Tensorflow-1.1.0-gpu

This is a ConvDAE using in notMNIST dataset to denoise.

The notMNIST dataset you can find in https://github.com/hankcs/udacity-deep-learning.   
The ConvDAE you can see in https://github.com/NELSONZHAO/zhihu/tree/master/denoise_auto_encoder.            
The differences :
 1) The loss function, we use the l2_loss, not the sigmoid_cross_entropy.
 2) We use 5X5 kernel size with 64 filters in all convolutional layers.
 3) Our structure is:
          conv->pool->conv->pool->resize->conv->resize->conv


Here is the result.
The first row is add noisy images, the second row is the denoise images after 
ConvDAE processing, the third row is the original images.
![image](https://github.com/PaulGitt/ConvDAE-use-in-notMNIST/blob/master/Result.png)
