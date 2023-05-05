## ResNet

$a^{[l]}$ => $a^{[l+1]}$ => $a^{[l+2]}$

$a^{[l]}$ => linear => ReLU => $a^{[l+1]}$ => linear => ReLU => $a^{[l+2]}$

$z^{[l+1]}$ => activation = $W^{[l+1]}a^{[l]} + b^{[l+1]}$

$a^{[l+1]}$ = ReLU($z^{[l+1]}$)

$z^{[l+2]}$ => activation = $W^{[l+2]}a^{[l+1]} + b^{[l+2]}$

$a^{[l+2]}$ = ReLU($z^{[l+2]}$)

A shortcut/skip is added before the ReLU layer but after the linear layer.

![ResNet](./Screenshot%202023-05-01%20003807.png)

By allowing the activation to go much deeper, this really helps with the vanishing and exploding gradient problem, and allows us to train much deeper neural networks without the performance degrading.

![ResNet vs plain](./Screenshot%202023-05-01%20004212.png)

### Why do ResNets work?

#### 1x1 convolution

![1x1 convolution](./Screenshot%202023-05-01%20012053.png)

### Inception Network

Instead of choosing between 1x1, 3x3, 5x5, etc. convolutions, we can use all of them at the same time.

![Inception Network](./Screenshot%202023-05-01%20012216.png)

![Problem with cost](./Screenshot%202023-05-01%20012411.png)

![Using 1x1 convolution](./Screenshot%202023-05-01%20012738.png)

### Inception Module

Example of an inception module:

![Inception module](./Screenshot%202023-05-01%20013325.png)

### Inception Network

Example of an inception network:

![Inception network](./Screenshot%202023-05-01%20013649.png)