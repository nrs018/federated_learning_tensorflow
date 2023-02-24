import numpy as np
import cv2
import os
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sys import argv
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMG_SIZE = 3072
CLASSES = 10
EPOCHES = 1
BATCHSIZE = 64


def load_train(paths, verbose=10000):
    """
    expects images for each class in separate dir,
    e.g all digits in 0 class in the directory named 0
    """
    data = list()
    labels = list()
    # loop over the input images
    for (i, imgpath) in enumerate(paths):
        # load the image and extract the class labels  cv2.IMREAD_GRAYSCALE
        im_gray = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        image = np.array(im_gray).flatten()
        label = imgpath.split(os.path.sep)[-2]
        label = label.split('_')[1]
        # scale the image to [0, 1] and add to list
        data.append(image / 255)
        labels.append(label)
        # show an update every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(paths)))
    # return a tuple of the data and labels
    return data, labels


def load(paths, verbose=10000):
    """
    expects images for each class in seperate dir,
    e.g all digits in 0 class in the directory named 0
    """
    data = list()
    labels = list()
    # loop over the input images
    for (i, imgpath) in enumerate(paths):
        # load the image and extract the class labels
        im_gray = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        image = np.array(im_gray).flatten()
        label = imgpath.split(os.path.sep)[-2]
        # label = label.split('_')[1]
        # scale the image to [0, 1] and add to list
        data.append(image / 255)
        labels.append(label)
        # show an update every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(paths)))
            # time.sleep(0.1)
    # return a tuple of the data and labels
    return data, labels


def create_client(client_id):
    img_path_train = '/home/narisu/src/TFF/cifar10/client/' + client_id
    img_path_test = '/home/narisu/src/TFF/cifar10/test/'
    # get the path list using the path object
    image_paths_train = list(paths.list_images(img_path_train))
    image_paths_test = list(paths.list_images(img_path_test))

    # apply our function
    image_list_train, label_list_train = load_train(image_paths_train, verbose=10000)
    image_list_test, label_list_test = load(image_paths_test, verbose=10000)

    # binarize the labels
    lb = LabelBinarizer()

    x = label_list_train + label_list_test
    x = lb.fit_transform(x)
    label_list_test = lb.fit_transform(label_list_test)
    label_list_train = x[0:len(label_list_train)]

    # split data into training and test set
    X_train = image_list_train
    y_train = label_list_train
    X_test = image_list_test
    y_test = label_list_test
    return X_train, y_train, X_test, y_test

class SimpleMLP:
    @staticmethod
    def build(shape, classes):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(shape,)),
            tf.keras.layers.Dense(512, kernel_initializer='zeros'),
            tf.keras.layers.Dense(64, kernel_initializer='zeros'),
            tf.keras.layers.Dense(classes, kernel_initializer='zeros'),
            tf.keras.layers.Softmax(),
        ])
        #model.load_weights('/home/narisu/src/TFF/Model/globalModel_cifar10.h5')
        return model


lr = 0.01
comms_round = 1
loss = 'categorical_crossentropy'
metrics = ['accuracy']
optimizer = SGD(learning_rate=lr
                #decay=lr / comms_round,
                #momentum=0.9
                )


smlp_SGD = SimpleMLP()
SGD_model = smlp_SGD.build(IMG_SIZE, CLASSES)
SGD_model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

# SGD_model.summary()
X_train1, y_train1, X_test1, y_test1 = create_client('l000_ship')
X_train2, y_train2, X_test2, y_test2 = create_client('l000_cat')

X_train1.extend(X_train2)
print(len(X_train1), ' ', len(X_train2))
x = np.vstack((y_train1, y_train2))
SGD_dataset1 = tf.data.Dataset.from_tensor_slices((X_test2, y_test2)).batch(BATCHSIZE)
img = np.array(X_test1)
img = img.reshape(-1, 3072)
_ = SGD_model.fit(SGD_dataset1, epochs=EPOCHES, verbose=0)
loss2, acc2 = SGD_model.evaluate(img, y_test1)
print(acc2, ' ', loss2)
