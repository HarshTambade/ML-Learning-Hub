# CNN Hands-on Exercises

## Exercise 1: Build a CNN from Scratch
Create a simple CNN with:
- 2 convolutional layers (32 and 64 filters)
- 2 max pooling layers
- 2 fully connected layers
- Use ReLU activation and dropout (0.5)

Train on CIFAR-10 for 5 epochs and report accuracy.

## Exercise 2: Data Augmentation
Implement data augmentation:
- Random rotation (±15 degrees)
- Random horizontal flip
- Random brightness adjustment
- Compare model performance with and without augmentation

## Exercise 3: Transfer Learning
Use a pre-trained ResNet18:
- Freeze all layers except the final FC layer
- Fine-tune on a custom dataset
- Compare accuracy with training from scratch

## Exercise 4: Hyperparameter Tuning
Experiment with:
- Learning rates: [0.001, 0.01, 0.1]
- Batch sizes: [32, 64, 128]
- Optimizers: Adam, SGD with momentum
- Report the best configuration

## Exercise 5: Visualize Feature Maps
- Extract intermediate layer outputs
- Visualize feature maps from conv layers
- Understand what patterns the network learns

## Exercise 6: Model Compression
- Quantize a trained model to INT8
- Measure speedup and accuracy loss
- Compare model sizes before/after

## Exercise 7: Adversarial Examples
- Generate adversarial examples using FGSM
- Test model robustness
- Implement adversarial training
