- Freeze all the layers except the last few layers of the model.
- Precompute the output of the last few layers of the model to train the new classifiers. Save to disk. You don't have to recompute those activations every time you train the new classifiers.
- If you have a larger dataset, you can unfreeze more layers and retrain the entire model.

![Transfer Learning](./Screenshot%202023-05-01%20015426.png)

## Data Augmentation

Common:
- Mirroring
- Random Cropping
- Rotation
- Shearing
- Local warping
- Color shifting, PCA color augmentation
- Replication (with slightly different alterations)