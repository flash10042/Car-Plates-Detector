# Car License Plates Detector

Implementation of car plates detector with opencv and tensorflow.

# Quick overview

This plate detector uses selective search for finding region proposals and MobileNetV3Small for binary classification is it a car plate.

# Run

To use this detector with your custom image run detect.py script as follows:

``` 
python detect.py -i path/to/image.png
```

# Requirements
Tqdm isn't mandatory and you can simply remove it from generate_train_data.py script
```
tensorflow
opencv-contrib-python
numpy
beatifulsoup4
tqdm
```
# Examples

![A car](https://github.com/flash10042/Car-Plates-Detector/blob/main/examples/Car_11.png)

Result withot non-max supression:

![A lot rects](https://github.com/flash10042/Car-Plates-Detector/blob/main/examples/Car_11_no_nms.png)

After NMS:

![Only one rect](https://github.com/flash10042/Car-Plates-Detector/blob/main/examples/Car_11_with_nms.png)

A few more cars:

![Much more cars](https://github.com/flash10042/Car-Plates-Detector/blob/main/examples/Car_223.png)

Result with NMS:

![Three cars four plates](https://github.com/flash10042/Car-Plates-Detector/blob/main/examples/Car_223_with_nms.png)

Here we can see a problem - we have four plates but only three cars. For NMS I was using tensorflow implementation and 
to solve this problem I think the best solution is write own NMS with check is one box fully overlaps another one.

# One more problem

Let's take a look at this image:

![Two cars two plates](https://github.com/flash10042/Car-Plates-Detector/blob/main/examples/Car_71.png)

Here's script output:

![What?](https://github.com/flash10042/Car-Plates-Detector/blob/main/examples/Car_71_no_nms.png)

Yeah, that's quite sad. I think getting more negative training examples may solve this problem. 
If not, the problem is probably with the selective search method.

# TODO

* Train model with more negative examples.
* Create a neural network for finding region proposals instead of opencv selective search.
