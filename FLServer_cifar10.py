import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
import os
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sys import argv
import tensorflow_federated as tff
from datetime import datetime

tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(None)

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

NUM_CLIENTS = 19
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10
SHAPE = 3072
IMG_SIZE = 3072
CLASSES = 10

# Load in the data
cifar10 = tf.keras.datasets.cifar10

# Distribute it to train and test set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

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
        # unbalance data
        label = imgpath.split(os.path.sep)[-1]
        label = label.split('_')[0]
        # non iid data
        # label = imgpath.split(os.path.sep)[-2]
        # label = label.split('_')[1]
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
    img_path_train = '/home/narisu/src/TFF/cifar10/client_unbalance/' + client_id
    # img_path_train = '/home/narisu/src/TFF/cifar10/client_non_iid/' + client_id
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

# Reduce pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0
# flatten the label values
y_train, y_test = y_train.flatten(), y_test.flatten()

X_train2, y_train2, X_test2, y_test2 = create_client('l000')
# X_train3, y_train3, X_test3, y_test3 = create_client('l001_automobile')
# X_train2, y_train2, X_test2, y_test2 = create_client('l000')
# X_train3, y_train3, X_test3, y_test3 = create_client('l001')
#X_train2.extend(X_train3)
tmp1 = list(y_train2)
#tmp2 = list(y_train3)
#tmp1.extend(tmp2)
yy_train = []
for i in range(len(y_test2)):
    for j in range(10):
        if y_test2[i][j] == 1:
            yy_train.append(j)
            break
yy_train = np.asarray(yy_train)
xx = np.array(X_test2)
yy = xx.reshape(-1, 32, 32, 3)

# number of classes
K = len(set(y_train))
# calculate total number of classes
# for output layer
# print("number of classes:", K)

# Build the model using the functional API
# input layer


i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3,3), padding="same", activation="relu")(i)
x = Conv2D(32, (3,3), activation="relu")(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Dropout(0.30)(x)
x = Conv2D(64, (3,3), padding="same", activation="relu")(x)
x = Conv2D(64, (3,3), activation="relu")(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Dropout(0.30)(x)
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(K, activation="softmax")(x)

model = Model(i, x)
# model.load_weights('/home/narisu/src/TFF/globalModel_cifar10.h5')
# model description
model.summary()

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

## First, delete globalModel_cifar10.h5 file
## Second, generate the globalModel_cifar10.h5
## Third, globalModel_cifar10.h5 will be moved to Directory roundX
##        and rename the roundX.h5
## Forth, all of file will be moved to the Directory roundX
## Last, RoundX.h5 file is moved out the Directory roundX and renamed globalModel_cifar10.h5
endFlag = False
for name in os.listdir(r'/home/narisu/src/TFF/Model'):
    if name.endswith('.h5'):
        endFlag = True
client_weights_list = []
global_model = model
if endFlag:
    for file in os.listdir(r'/home/narisu/src/TFF/Model'):
        print(file)
        if os.path.isfile('/home/narisu/src/TFF/Model/' + file):
            aa = model

            aa.load_weights('/home/narisu/src/TFF/Model/' + file)
            # global_model_at_server.set_weights(global_model_at_server.get_weights() + keras_model.get_weights())
            client_weights_list.append(aa.get_weights())

    avg_grad = list()
    # get the average grad across all client gradients
    for grad_list_tuple in zip(*client_weights_list):
        layer_mean = tf.math.reduce_mean(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)

    global_model.set_weights(avg_grad)
    global_model.evaluate(yy, yy_train)
    global_model.save_weights('/home/narisu/src/TFF/Model/globalModel_cifar10.h5')
else:
    nn = 0
    for dd in os.listdir(r'/home/narisu/src/TFF/Model'):
        if os.path.isdir('/home/narisu/src/TFF/Model/' + dd):
            nn = nn + 1
    os.popen("cp /home/narisu/src/TFF/Model/round" + str(nn - 1) +
             "/round" + str(nn - 1) + ".h5 /home/narisu/src/TFF/Model/round" + str(nn - 1) + "/globalModel_cifar10.h5")
