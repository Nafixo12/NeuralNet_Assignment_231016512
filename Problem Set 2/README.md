# Problem Set 02 – Bank Marketing Classification

## Dataset
- UCI Bank Marketing Dataset (`bank-full.csv`)
- Target: y (term deposit subscription: yes/no)
- Features: 16 attributes

## Preprocessing
- Target converted to binary (yes=1, no=0)
- Categorical features encoded
- Numerical features scaled
- Train-test split: 80/20

## Model
- Feedforward Neural Network (MLP)
- Hidden layers: Dense(32, relu), Dense(16, relu)
- Output: Dense(1, sigmoid)
- Optimizer: Adam
- Loss: Binary crossentropy
- Epochs: 15, Batch size: 64

## Accuracy
0.89

## Confusion Matrix
[[7686  299]
 [ 618  440]]

## Classification Report
              precision    recall  f1-score   support
           0       0.93      0.96      0.95      7985
           1       0.60      0.42      0.49      1058

    accuracy                           0.89      9043
   macro avg       0.77      0.69      0.72      9043
weighted avg       0.88      0.89      0.88      9043

## Training Plots
Banking_Training_Accuracy_And_Loss

## Deliverables
banking_model_using_logistic_regression.py  
plots.png  
README.md
