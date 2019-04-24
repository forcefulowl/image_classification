from keras import layers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import datetime
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model


start = datetime.datetime.now()

tf.global_variables_initializer()

img_width, img_height = 224, 224
train_data_dir = "C:\\Users\gavin\Desktop\Train1_M"
validation_data_dir = "C:\\Users\gavin\Desktop\Test1_M"
data_dir = 'C:\\Users\gavin\Desktop\whole_M'
nb_train_samples = 7000
nb_validation_samples = 3000
batch_size = 64
epochs = 200


def MobileNet(input_shape=None,
              alpha=1.0,
              depth_multiplier=1,
              dropout=1e-3,
              input_tensor=None,
              pooling=None,
              classes=7,
              **kwargs):


    if input_shape is None:
        default_size = 224
    else:
        rows = input_shape[0]
        cols = input_shape[1]

        if rows == cols and rows in [128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape = (default_size, default_size, 3)

    row_axis, col_axis = (0, 1)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    img_input = layers.Input(shape=input_shape)

    x = conv_block(img_input, 32, alpha, strides=(2, 2))
    x = depthwise_conv_block(x, 64, alpha, depth_multiplier, strides=(1, 1))
    x = depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2))
    x = depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(1, 1))
    x = depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2))
    x = depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(1, 1))
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2))
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(1, 1))
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(1, 1))
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(1, 1))
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(1, 1))
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(1, 1))
    x = depthwise_conv_block(x, 1024, alpha, depth_multiplier, strides=(2, 2))
    x = depthwise_conv_block(x, 1024, alpha, depth_multiplier, strides=(1, 1))
    if pooling == 'avg':
        x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D()(x)

    inputs = img_input

    # Create model
    model = Model(inputs, x)

    return model




def conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = -1
    filters = int(filters * alpha)
    x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(inputs)
    x = layers.Conv2D(filters, kernel,
               padding='valid',
               use_bias=False,
               strides=strides,
               kernel_initializer='he_uniform',
               name='conv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return layers.ReLU(6., name='conv1_relu')(x)


def depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                         depth_multiplier=1, strides=(1, 1), block_id=1):
    channel_axis = -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(((0, 1), (0, 1)))(inputs)
    x = layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               kernel_initializer='he_uniform',
                               use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)
    x = layers.ReLU(6.)(x)

    x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)
    return layers.ReLU(6.)(x)


model = MobileNet(input_shape=(img_width, img_height, 3), pooling='avg')

x = model.output

shape = (1, 1, 1024)
x = layers.Reshape(shape)(x)
x = layers.Dropout(1e-3)(x)
x = layers.Conv2D(7, (1, 1),
                  padding='same',
                  kernel_initializer='he_uniform')(x)
x = layers.Activation('softmax')(x)
x = layers.Reshape((7,))(x)


model_final = Model(input=model.input, output=x)
model.summary()

model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.1, momentum=0.9), metrics=["accuracy"])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30,
    validation_split=0.3)

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
model_final.save_weights('C:\\Users\gavin\Desktop\model4.h5')
