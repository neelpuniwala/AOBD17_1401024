from __future__ import print_function
from decimal import *
import tensorflow as tf
import numpy
#rng = numpy.random


# Parameters for learning
learning_rate = 0.001
training_epochs = 500

# Ss
num_Table=9					# desired number to find stuff for. change from 2 to 15
training_samples = 10		# number of training samples (number*1 ..... number*training_samples
testing_samples = 10		# number of testing samples (these are new samples, unknown earlier)


# allocate arrays for samples
train_X = [0.0]*training_samples
train_Y = [0.0]*training_samples
test_X = [0.0]*testing_samples
test_Y = [0.0]*testing_samples


for i in range(training_samples):			# 0 based indexing
     train_X[i] = i+1
     train_Y[i] = num_Table*(i+1)

for i in range(testing_samples):
	test_X[i] = training_samples+i+1
	test_Y[i] = num_Table*(training_samples+i+1)	# 0 based again

train_X = numpy.array(train_X)
train_Y = numpy.array(train_Y)
test_X = numpy.array(test_X)
test_Y = numpy.array(test_Y)

# print if not too large arrays
if training_samples in range(1,29):
	print ("Train values:",train_Y,"\n")
if training_samples in range(1,29):
	print ("Test values:",test_Y,"\n")

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")

# Construct a linear model
pred = tf.add(tf.mul(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*training_samples)

# Gradient descent basedd optimization
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
         for i in range(training_samples):			# batch size is 1
            sess.run(optimizer, feed_dict={X: train_X[i], Y: train_Y[i]})

    print("Finished training the neuron.\n")

	# find the traning cost, from tensor
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})

    print("Calculated W:", sess.run(W), "\nCalculated b:", sess.run(b))
#    getcontext().prec=4		# Precision for decimal point numbers, doesnt work.
    for K in range(testing_samples):
		print("i:",test_X[K],"Output:",Decimal(test_X[K]*W.eval() + b.eval()),"and Desired:",test_Y[K])
		testing_cost = sess.run(
			tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * testing_samples),
			feed_dict={X: test_X, Y: test_Y})  # same function as cost above

# print errors.
print("\nTesting cost=", testing_cost)
print("Training cost=", training_cost)
print("Absolute mean squared loss:", abs(training_cost - testing_cost))
