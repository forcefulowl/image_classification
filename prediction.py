from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Activation
from keras import regularizers
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import datetime
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os
import cv2

from keras.applications.imagenet_utils import decode_predictions


sess = tf.InteractiveSession()

tf.global_variables_initializer()

model_path = "C:\\Users\gavin\Desktop\model3.h5"
test_path = "C:\\Users\gavin\Desktop\Test_set_2"
img_path = "C:\\Users\gavin\Desktop\Test_set\img_1.jpg"


img_width, img_height = 224, 224
nb_train_samples = 6900
nb_validation_samples = 3000
batch_size = 64
epochs = 100

model = applications.MobileNet(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))

x = model.output
x = Flatten()(x)
x = Dense(1024)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(7, activation="softmax")(x)

# creating the final model
model_final = Model(input=model.input, output=predictions)

'''
test_image = image.load_img(img_path, target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
index = tf.argmax(result, axis=1)
print(index.eval())

=============================================================


model_final.load_weights(model_path)

predict_datagen = ImageDataGenerator(rescale=1./255)

test_image = image.load_img(img_path, target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model_final.predict(test_image,batch_size=1)
index = tf.argmax(result, axis=1)
print(index.eval())
'''

'''
# predict single image
model_final.load_weights(model_path)

test_image = image.load_img(img_path, target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image = test_image/255
test_image = np.expand_dims(test_image, axis=0)
result = model_final.predict(test_image, batch_size=1)
index = tf.argmax(result, axis=1)
print(index.eval())
'''

'''
# predict set images
model_final.load_weights(model_path)

for i in range(10):
    path = test_path+'\img_'+str(i)+'.jpg'
    test_image = image.load_img(path, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = test_image/255
    test_image = np.expand_dims(test_image, axis=0)
    result = model_final.predict(test_image, batch_size=1)
    index = tf.argmax(result, axis=1)
    print(index.eval())
'''

model_final.load_weights(model_path)


for i in range(1512):
    path = os.path.join(test_path, str(1+i)+'.jpg')
    test_image = image.load_img(path, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = test_image/255
    test_image = np.expand_dims(test_image, axis=0)
    result = model_final.predict(test_image, batch_size=1)
    index = tf.argmax(result, axis=1)
    if index.eval() == 0:
        with open("C:\\Users\gavin\Desktop\Test_result.txt", "a") as f:
            f.write('AKIEC\n')
            f.close()
    elif index.eval() == 1:
        with open("C:\\Users\gavin\Desktop\Test_result.txt", "a") as f:
            f.write('BCC\n')
            f.close()
    elif index.eval() == 2:
        with open("C:\\Users\gavin\Desktop\Test_result.txt", "a") as f:
            f.write('BKL\n')
            f.close()
    elif index.eval() == 3:
        with open("C:\\Users\gavin\Desktop\Test_result.txt", "a") as f:
            f.write('DF\n')
            f.close()
    elif index.eval() == 4:
        with open("C:\\Users\gavin\Desktop\Test_result.txt", "a") as f:
            f.write('MEL\n')
            f.close()
    elif index.eval() == 5:
        with open("C:\\Users\gavin\Desktop\Test_result.txt", "a") as f:
            f.write('NV\n')
            f.close()
    elif index.eval() == 6:
        with open("C:\\Users\gavin\Desktop\Test_result.txt", "a") as f:
            f.write('VASC\n')
            f.close()



