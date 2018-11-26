# PyCCTV

A CCTV camera application with "person" detection and remote monitoring over Wi-Fi.

Read more about it [on my blog](https://varunbpatil.github.io/2018/11/26/PyCCTV.html).


### Usage

1. Download the pretrained Keras Yolo V3 model (trained on the COCO dataset)
   from [here](https://drive.google.com/file/d/1_ZpUKKikmEI5_sZ4Px3z4D2pbBRMnzoK/view?usp=sharing).

2. Run the following command.

	$ py_cctv.py --model <path to yolo.h5> --output <path to output image directory>
