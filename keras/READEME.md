# keras References

| Method                                                                                  | Description                                                                                                  |
| --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| to_categorical(y_train,num_classes)                                                     | Converts an integer to a list of binaries                                                                    |
| Sequential                                                                              | groups a stack of layers into a keras model and provides training and inference features for the model built |
| Dense(512, activation='relu', input_shape=(784,))                                       | input layer for a simple deep-neural network                                                                 |
| Dense(512, activation='relu')                                                           | one of the middle layers                                                                                     |
| Dense(10,activation='softmax')                                                          | output layer, since we have to classify the images among 1-10                                                |
| model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])   | Compile a keras model                                                                                        |
| model.summary()                                                                         | prints a readable summary of a keras model                                                                   |
| history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))      | fit a model on the diven train-test data                                                                     |
| history.history['accuracy']                                                             | Accuracy of the model on training data                                                                       |
| history.history['val_accuracy']                                                         | accuracy of the model on test data                                                                           |
| history.history['loss']                                                                 | loss of the model on training data                                                                           |
| plt.plot(history.history['accuracy'])                                                   | can be simply plotted using plt.plot over the number of iterations                                           |
| Conv2D(32, kernel_size=(5,5), input_shape=(28,28,1), padding='same', activation='relu') | input layer for a 2D-cnn                                                                                     |
| MaxPooling2D()                                                                          | max pooling for 2D spatial data.                                                                             |
| Conv2D(64, kernel_size=(5,5), padding='same', activation='relu')                        | middle layer for a cnn model                                                                                 |
| Flatten()                                                                               | flattens the input                                                                                           |
| cnn.load_weights('weights/cnn-model5.h5')                                               | loads all layer weights to a model (need to read up)                                                         |
| score = cnn.evaluate(X_test,y_test)                                                     | returns loss and metric values                                                                               |
| ImageDataGenerator, flow_from_directory                                                 | used for generating more samples out of the given image dataset                                              |
| img = image.load_img('images/spoon.jpeg', target_size=(224,224))                        | load a given image to be used in a keras model                                                               |
| image.img_to_array(img)                                                                 | Convert the loaded image to a numpy array                                                                    |
| vgg16.VGG16(weights='imagenet'), vgg16.preprocess_input(arr), model.predict(arr)        | trying out the VGG16 model                                                                                   |

```python
# Generating additional samples out of the given set of images
jf_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True
)
train_generator = train_datagen.flow_from_directory('images/sample-train/',target_size=(150,150), save_to_dir='images/sample-confirm/')
j=0
for batch in jf_datagen.flow_from_directory('images/sample-train/', target_size=(150,150), save_to_dir='images/sample-confirm/'):
    j+=1
    if ( j > 10):
        break
```

```python
# sample trial of the inbuilt vgg16 model
model = vgg16.VGG16(weights='imagenet')
img2 = image.load_img('images/fly.jpeg', target_size=(224,224))
arr2 = image.img_to_array(img2)
arr2 = np.expand_dims(arr2, axis=0)
arr2 = vgg16.preprocess_input(arr2)
preds2 = model.predict(arr2)
vgg16.decode_predictions(preds2, top=5)
```
