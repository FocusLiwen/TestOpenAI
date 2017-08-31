import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess = tf.InteractiveSession()

# constant
hello = tf.constant("Hello, TensorFlow!")
print(sess.run(hello))
print(hello)

# node
node1 = tf.constant(3.0, dtype = tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)
print("sess.run(node1, node2): ", sess.run([node1, node2]))

node3 = tf.add(node1, node2)
print(node3)
print("sess.run(node3): ", sess.run(node3))

#placeholder
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = tf.add(a, b)
print(adder_node) # no shape info
print(sess.run(adder_node, {a: 3, b: 4}))
print(sess.run(adder_node, {a: [1,3], b:[2, 4]}))

#variables
W = tf.Variable([.3], dtype = tf.float32)
b = tf.Variable([-.3], dtype = tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b
print("linear_model: ", linear_model)

# initialize variables
init = tf.global_variables_initializer()
sess.run(init)
print(W)
print(b)
print(sess.run([W, b]))
print(sess.run(linear_model, {x: [1,2,3,4]}))

x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# loss function
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:x_train, y:y_train}))

#replace variables
fixW = tf.assign(W, [-1])
fixb = tf.assign(b, [1])
#need to run it to make the change actually happend
sess.run([fixW, fixb])
print(sess.run(loss, {x:x_train, y:y_train}))

#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# get W and b
for i in range(1000):
    sess.run(train, {x:x_train, y:y_train})
print(sess.run([W, b, loss], {x:x_train, y:y_train}))









