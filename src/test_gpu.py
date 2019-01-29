# import tensorflow as tf
# import numpy as np
# import time

# value = np.random.randn(5000,1000)
# a = tf.constant(value)

# b = a*a

# tic = time.time()
# with tf.Session() as sess:
#     for i in range(100000):
#         sess.run(b)
# toc = time.time()
# t_cost = toc - tic

# print(t_cost)
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print('fit')
model.fit(x_train[:10], y_train[:10], epochs=5)
print('done')
model.evaluate(x_test, y_test)