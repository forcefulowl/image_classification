import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import layers
from keras.models import  Model
import datetime
import tensorflow as tf
from keras.backend import manual_variable_initialization

import gconv
import model_separable_dw_resnet50
import model_separable_dw_vgg16
import mobile_net2_revised
import new_revised_resnet50
import new_revised_vgg16
import why_cant_import



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)


start = datetime.datetime.now()



tf.global_variables_initializer()
keras.initializers.glorot_normal(seed=None)

img_width, img_height = 224, 224
train_data_dir = "C:\\Users\gavin\Desktop\Train1_M"
validation_data_dir = "C:\\Users\gavin\Desktop\Test1_M"
data_dir = 'C:\\Users\gavin\Desktop\whole_M'
nb_train_samples = 7000
nb_validation_samples = 3000
batch_size = 8
epochs = 100


model = model_separable_dw_resnet50.MobileNet(weights=None, include_top=False, input_shape=(img_width, img_height, 3),
                               pooling='avg')


# for layer in model.layers[:5]:
#     layer.trainable = False

x = model.output

x = layers.Dense(7, activation='softmax')(x)

# creating the final model
model_final = Model(inputs=model.input, output=x)

model_final.summary()

# compile the model
# model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1, momentum=0.9), metrics=["accuracy"])
model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.adadelta(lr=1, rho=0.95, epsilon=None, decay=0), metrics=['accuracy'])

# Initiate the train and test generators with data Augumentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30,
    validation_split=0.3)


'''
test_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30)
'''


train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    subset='training',
    shuffle=True)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    subset='validation',
    shuffle=True)

# early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)





# Train the model
model_final.fit_generator(
    train_generator,
    steps_per_epoch=7000//batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=3000//batch_size,
    class_weight='auto',
    shuffle=True)
#    callbacks=[TensorBoard(log_dir='mytensorboard')]
#   callbacks=[early_stop],

end = datetime.datetime.now()
print(end-start)
model_final.save_weights('C:\\Users\gavin\Desktop\model_dw_linear_1.h5')
