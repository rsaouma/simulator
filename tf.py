import tensorflow as tf
import pandas as pd
hello = tf.constant('Hello, TensorFlow!')
sess = tf.compat.v1.Session()


x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f =x*x*y+y+2


sess = tf.compat.v1.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
print(sess.run(hello))
sess.close()