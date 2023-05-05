## Classification with Localization

![Example](./Screenshot%202023-05-01%20023159.png)

## Landmark Detection

Someone has to label the data. This is a very time consuming process.

The goal is to predict the position of the landmarks. They have to be consistent across all the images in the dataset.

## Object Detection

#### Example

We start with closely cropped images to train our ConvNet

Training set:
| Image | Label |
| --- | --- |
| car | 1 |
| car | 1 |
| car | 1 |
| not a car | 0 |
| not a car | 0 |

#### Sliding Window Detection

We pick a window size and slide it across the image. We then classify each window as a car or not a car by inputting it into our ConvNet.

Then, we pick a larger window size and repeat the process.

Disadvantages:
- Computationally expensive

#### Convolutional Implementation of Sliding Windows

Turning a fully connected (FC) layer into a convolutional layer.

Let's say we have a 14x14x3 image and a 5x5x3 filter. We can apply the filter to the image to get a 10x10x16 output. Then to a 2x2 max pooling to get a 5x5x16 output. Then, through a FC layer to connect to 400 units. Then, another FC layer. Finally, output y using a softmax (4) to classify: pedestrian, car, motorcycle, background.

![Example](./Screenshot%202023-05-01%20025545.png)

![Example](./Screenshot%202023-05-01%20030219.png)

![Example](./Screenshot%202023-05-01%20030312.png)

There's still one problem. The position of the bounding box is not accurate. We can fix this by using a regression algorithm to predict the position of the bounding box.

#### Intersection Over Union

Evluating object localization 
The intersection over union (IoU) computes the size of the intersection divided by the size of the union of two boxes.
> "Correct" if IoU >= 0.5
More generally, IoU is a measure of the overlap between two bounding boxes. This is also a way to measure how similar two bounding boxes are.

#### Non-max Suppression

Because we run image classification at multiple locations and scales, it's possible that we end up with multiple detections of the same object.

What non-max suppression does is it looks at the probability of each object and keeps the one with the highest probability. Then, it looks at the IoU of the remaining boxes and removes the ones that have a high IoU with the box that was kept.

For multiple classes, we run non-max suppression for each class.

![Example](./Screenshot%202023-05-01%20032149.png)

#### Anchor Boxes

The problem with the previous method is that it only detects one object per grid cell. What if there are multiple objects in one grid cell?

![Example](./Screenshot%202023-05-01%20032524.png)

(grid cell, anchor box) pair

![Example](./Screenshot%202023-05-01%20032808.png)

![Example](./Screenshot%202023-05-01%20033259.png)

_What if there are 2 anchor boxes but 3 objects in the grid cell_? That's one case where this algorithm doesn't work well. Or _what if there are 2 objects but both of them are in the same anchor box_? That's another case where this algorithm doesn't work well.

#### YOLO Algorithm

You only look once (YOLO) is a state-of-the-art, real-time object detection system.

![Example](./Screenshot%202023-05-01%20034008.png)

![Example](./Screenshot%202023-05-01%20034132.png)

![Example](./Screenshot%202023-05-01%20034302.png)

#### Region Proposals

Region proposal algorithms find regions of interest in an image. They're used in object detection algorithms like R-CNNs.

![Example](./Screenshot%202023-05-01%20034647.png)
