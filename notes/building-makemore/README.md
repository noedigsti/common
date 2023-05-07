Backward Pass:
- Reset gradients to None is more efficient than setting them to zero.

Batch Normalization:
- This is a layer that you can sprinkle into your network to make training faster and more stable.
- Usually between the linear and non-linear layers.
- If you use batch normalization layers, you don't need to use bias in the previous weighted layers. Instead we can use the bias in the batch normalization layer.
- Hint: Nonone likes this layer, it causes a lot of bugs. Avoid it as much as possible.
- Alternatives: Group Normalization, Layer Normalization.