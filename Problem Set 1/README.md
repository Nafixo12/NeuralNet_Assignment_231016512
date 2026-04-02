# Problem Set 01 – CNN on Medical Images

## Dataset
- Medical image dataset provided
- Target: classification of images into categories

## Preprocessing
- Images resized to 64×64 and normalized
- Train-test split: 80/20

## Model
- Convolutional Neural Network (CNN)
- Layers: Conv2D, MaxPooling, Flatten, Dense
- Optimizer: Adam
- Loss: Categorical crossentropy
- Epochs: 15, Batch size: 64

## Accuracy
0.87

## Confusion Matrix
[[  5 119   0]
 [ 35 1008   0]
 [  0   3    0]]

## Classification Report
              precision    recall  f1-score   support
           1       0.12      0.04      0.06       124
           2       0.89      0.97      0.93      1043
           3       0.00      0.00      0.00         3

    accuracy                           0.87      1170
   macro avg       0.34      0.34      0.33      1170
weighted avg       0.81      0.87      0.83      1170

## Training Plots
CNN_Training_Accuracy_And_Loss.png

## Deliverables
CNN.py  
CNN_Training_Accuracy_And_Loss.png  
README.md
