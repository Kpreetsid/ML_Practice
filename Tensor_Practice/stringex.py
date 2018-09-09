import tensorflow as tf
import numpy as np

x = tf.constant("My name is Nitish Gaddam")

sess = tf.Session()
y=sess.run(x).decode()
count=0
for i in y:
    count+=1
print(count)
sess.close()