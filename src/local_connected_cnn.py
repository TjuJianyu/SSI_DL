# import tensorflow as tf
# class lc_cnnmodel():
#     def __init__(self,width,height):
#
#         self.inputimg = tf.placeholder(shape=[None, width, height], dtype=tf.float32, name='input')
#
#         hidden1 = tf.layers.conv2d(inputimg, filters=16, kernel_size=(3,3),
#                                    padding='same', activation=activation, use_bias=True)
#         hidden1 = tf.layers.max_pooling1d(hidden1, pool_size=pool_size[0], strides=strides * 2, padding='same')
#         hidden1 = tf.layers.dropout(hidden1, rate=dropout)
#
import numpy as np
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[:,:,:,np.newaxis]
x_test = x_test[:,:,:,np.newaxis]

model = tf.keras.models.Sequential([
    tf.keras.layers.LocallyConnected2D(32,(3,3),input_shape=(28,28,1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
