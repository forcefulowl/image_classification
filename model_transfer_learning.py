from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import layers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Activation
from keras import regularizers
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import datetime
import tensorflow as tf
from keras.backend import manual_variable_initialization


start = datetime.datetime.now()

tf.global_variables_initializer()

img_width, img_height = 224, 224
train_data_dir = "C:\\Users\gavin\Desktop\Train1_M"
validation_data_dir = "C:\\Users\gavin\Desktop\Test1_M"
data_dir = 'C:\\Users\gavin\Desktop\whole_M'
nb_train_samples = 7000
nb_validation_samples = 3000
batch_size = 64
epochs = 100


model = applications.MobileNet(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3),
                               pooling='avg')


# for layer in model.layers[:5]:
#     layer.trainable = False

x = model.output

'''

# x = Flatten()(x)
x = Dropout(1e-3)(x)
x = Dense(1024)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(7, activation="softmax")(x)
'''

shape = (1, 1, 1024)
x = layers.Reshape(shape)(x)
x = layers.Dropout(1e-3)(x)
x = layers.Conv2D(7, (1, 1),
                  padding='same',
                  kernel_initializer='he_uniform')(x)
x = layers.Activation('softmax')(x)
x = layers.Reshape((7,))(x)

# creating the final model
model_final = Model(input=model.input, output=x)

# compile the model
model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.001, momentum=0.9), metrics=["accuracy"])
# model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0), metrics=['accuracy'])

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
    shuffle=True,
    callbacks=[TensorBoard(log_dir='mytensorboard')])
#   callbacks=[early_stop],

end = datetime.datetime.now()
print(end-start)
model_final.save_weights('C:\\Users\gavin\Desktop\model5.h5')
