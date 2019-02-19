import tensorflow as tf
#initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([2,35,6,6])
#multiply both vectors
result = tf.multiply(x1, x2)
#init a session
sess = tf.Session()
#result in form of tensor
print(sess.run(result))
#close session
sess.close()
