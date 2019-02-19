import tensorflow as tf
#initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([2,35,6,6])
#multiply both vectors
result = tf.multiply(x1, x2)
#init an interactive session and automatically close 
with tf.Session() as sess:
    output = sess.run(result)
    print(output)
