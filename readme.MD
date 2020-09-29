## maxres ##

動漫插畫放大器

![sample](https://i.imgur.com/3wtqBx7.png)

Artist: 荻pote

### Requirements

- Tensorflow 2.2.0
- Numpy
- Opencv-python
- Flask

### Use ###

##### Upscaling #####

- 2x:  
`py maxres.py 2x img_path`

- 4x:  
`py maxres.py 4x img_path`

- 8x:
`py maxres.py 8x img_path`

##### Jpeg denoise #####

- Very low (Quality: 90 - 100):  
`py maxres.py jpeg-verylow img_path`

- Low (Quality: 75 - 89):  
`py maxres.py jpeg-low img_path`

- Medium (Quality: 55 - 74):  
`py maxres.py jpeg-medium img_path`

- High (Quality: 30 - 54):  
`py maxres.py jpeg-high img_path`

- Very High (Quality: 1 - 29):  
`py maxres.py jpeg-veryhigh img_path`

##### Denoise #####

- Light:  
`py maxres.py denoise-light img_path`

- Medium:  
`py maxres.py denoise-medium img_path`

- Strong:  
`py maxres.py denoise-strong img_path`

### Web UI/API ###

`py webapi.py`

Then go to your host :5000.

### References ###

[1] Lim, B., Son, S., Kim, H., Nah, S., & Mu Lee, K. (2017). Enhanced deep residual networks for single image super-resolution. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops (pp. 136-144).

[2] Dong, C., Loy, C. C., He, K., & Tang, X. (2015). Image super-resolution using deep convolutional networks. IEEE transactions on pattern analysis and machine intelligence, 38(2), 295-307.

[3] Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7132-7141).
