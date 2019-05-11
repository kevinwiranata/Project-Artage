import struct f
import numpy as np
import pandas as pd
import os, sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from numpy import genfromtxt
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from IPython.display import display
from IPython.display import Image as _Imgdis


onlyfiles = [f for f in os.listdir('resized/') if os.path.isfile(os.path.join('Project Artage/resized/', f))]

print("Working with {0} images".format(len(onlyfiles)))
'''
print("Image examples: ")

for i in range(40, 42):
    print(onlyfiles[i])
    display(_Imgdis(filename="resized/" + onlyfiles[i], width=240, height=320))
'''

artPictures = genfromtxt('all_data_revised2.csv', delimiter=',')
# print(artPictures)
train_files = []
training_date = []
training_img = []

test_files = []
test_date = []
test_img = []
count = 0
for _file in onlyfiles:
    fileNum = int(_file.split(".jpg")[0])
    itemIndex = np.where(artPictures[:,1]==fileNum)
    try:
        if (count <= 5600):
            training_date.append(int(artPictures[int(itemIndex[0])][0]))
        else:
            test_date.append(int(artPictures[int(itemIndex[0])][0]))
    except TypeError:
        pass
    else:
        if (count <= 5600):
            train_files.append(os.path.join('resized/', _file))
        else:
            test_files.append(os.path.join('resized/', _file))
        count += 1

i=0
total_files = train_files + test_files
for img in total_files:
    # Convert to Numpy Array
    img = load_img(img)
    x = img_to_array(img)
    # Normalize
    if (i <= 5600):
        training_img.append(x)
    else:
        test_img.append(x)
    i += 1
    if i % 250 == 0:
        print("%d images to array" % i)

# print(training_img[0].shape)
# print(training_img[0])
print("All images to array!")
# print(len(training_date))
# print(len(train_files))

# print(artPictures.columns.tolist())
# print(artPictures.info())

num_classes = 19
x_train = np.array(training_img)
x_test = np.array(test_img)
print(x_train.shape)
y_train = keras.utils.to_categorical(training_date, num_classes)
y_test = keras.utils.to_categorical(test_date, num_classes)
print(y_train.shape)
# print(y_test.shape)

import re
import matplotlib.pyplot as plt


# pass in a list of tuples(image, label)
def plot_artworks(images_and_labels, predictions=None):
    label_dict = {0: '1050-1099', 1: '1100-1149', 2: '1150-1199',
                  3: '1200-1249', 4: '1250-1299', 5: '1300-1349', 6: '1350-1399',
                  7: '1400-1449', 8: '1450-1499', 9: '1500-1549', 10: '1550-1599',
                  11: '1600-1649', 12: '1650-1699', 13: '1700-1749',
                  14: '1750-1799', 15: '1800-1849',
                  16: '1850-1899', 17: '1900-1949',18: '1950-1999'}
    for i in range(len(images_and_labels)):
        image_data, image_label = images_and_labels[i]
        img = image_data
        title = label_dict[image_label]
        if predictions:
            pred_label = predictions[i]
            title += ' Prediction: {}'.format(label_dict[pred_label])
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111)
        ax.imshow(img, interpolation='bicubic')
        ax.set_title('Category = ' + title, fontsize=15)
        plt.show()


def get_random_images():
    indices = np.random.randint(0, x_train.shape[0] - 1, 10)
    images_and_labels = []
    for idx in indices:
        image, label = x_train[idx], y_train[idx]
        label = [i for i, val in enumerate(label) if val == 1][0]  # un OHE
        images_and_labels.append((image, label))
    return images_and_labels


# to be used later
def get_first_10():
    images_and_labels = []
    for i in range(10):
        image, label = x_train[i], y_test[i]
        label = [i for i, val in enumerate(label) if val == 1][0]  # un OHE
        images_and_labels.append((image, label))
    return images_and_labels


images_and_labels = get_random_images()
plot_artworks(images_and_labels)
# image, label = x_train[0], y_train[0]
# label = [i for i, val in enumerate(label) if val == 1][0]
# images_and_labels = [(image, label)]
# print(label)
# plot_artworks(images_and_labels)


def alexnet_model(img_shape=(200, 200, 3), n_classes=19, l2_reg=0., weights=None):
    alexnet = Sequential()
    # layer 1
    # note: 11x11 changed to 7x7
    alexnet.add(Conv2D(96, (7, 7), input_shape=img_shape, padding='same', kernel_regularizer=l2(l2_reg)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(
        pool_size=(2, 2)))  # stride of 1 would creates overlapping pooling, implicit stride is pool size (2x2)

    # layer 2
    # We should apply Batch Normalization before the Relu
    # also, we should do a 2x2 overlapping max pool at the end
    alexnet.add(Conv2D(256, (5, 5), input_shape=img_shape, padding='same', kernel_regularizer=l2(l2_reg)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # layer 3
    alexnet.add(ZeroPadding2D((1, 1)))
    # Implement layer 3, which has a 3x3 conv, 512 filters, and uses same padding.
    # the batchnorm, activation, and pooling should be the same as in layer 2.
    alexnet.add(Conv2D(512, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # layer 4
    # Implement layer 4 to be the same as layer 3, except we should have 1024 filters and no max pool at the end.
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(1024, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))

    # layer 5
    # Implement layer 5 to be the same as layer 3, except we should have 1024 filters.
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(1024, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # layer 6
    # Implement layer 6, which is the first FC layer.  It should have 3072 neurons.
    # We should have batch norm before the nonlinearity, and dropout with p=0.5
    alexnet.add(Flatten())
    alexnet.add(Dense(3072))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # layer 7
    # Implement layer 7, which is the 2nd FC layer. It should have 4096 neurons.
    # The Batchnorm and dropout should be the same as in layer 6.
    alexnet.add(Dense(4096))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 8
    # Implement layer 8, which is the softmax layer. It should have batchnorm before applying the softmax operation.
    alexnet.add(Dense(n_classes))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('softmax'))

    if weights:
        alexnet.load_weights(weights)
    return alexnet

model = alexnet_model()
print(model.summary())

model = alexnet_model()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
print(x_test.shape, y_test.shape)
model.fit(x_train, y_train,
          batch_size=256,
          epochs=5,
          verbose=1,
          validation_data=(x_test, y_test),
          )
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

predictions = model.predict(x_train[:20])
pred = np.argmax(predictions,axis=1)
print(pred.shape)
images_and_labels = get_first_10()
pred = list(pred)
print(pred)
plot_artworks(images_and_labels, pred)
