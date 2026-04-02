import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(64,64),
    batch_size=64,
    class_mode='categorical',
    subset='training'
)
val_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(64,64),
    batch_size=64,
    class_mode='categorical',
    subset='validation'
)

model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128,activation='relu'),
    Dense(train_generator.num_classes,activation='softmax')
])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(train_generator,epochs=15,validation_data=val_generator,verbose=2)

y_true = val_generator.classes
y_pred = model.predict(val_generator)
y_pred_classes = y_pred.argmax(axis=1)

print("Accuracy:", accuracy_score(y_true,y_pred_classes))
print("Confusion Matrix:\n", confusion_matrix(y_true,y_pred_classes))
print("Classification Report:\n", classification_report(y_true,y_pred_classes))

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],label='Train Acc')
plt.plot(history.history['val_accuracy'],label='Val Acc')
plt.title('Accuracy'); plt.xlabel('Epoch'); plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'],label='Train Loss')
plt.plot(history.history['val_loss'],label='Val Loss')
plt.title('Loss'); plt.xlabel('Epoch'); plt.legend()

plt.tight_layout()
plt.show()


