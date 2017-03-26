from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data	# downloads mnist ZIP

#Here one hot means [1,0,0,0,0,0] like structure
mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

#Hidden Layer 1 
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# we define a total of 10 classes
n_class = 10

#Data to be taken at a time 
batch_size = 100

img_h = 28
img_w = 28
img_pixels = img_h*img_w


# all images are 28*28, which are folded as 784 columns in a vector here.
x = tf.placeholder('float',[None,img_pixels])
y = tf.placeholder('float')

# we define the neural network here, and pass data
def neural_network_model(data):

	#Assign Weights and biases
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([img_pixels,n_nodes_hl1])),
					 'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
					 'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	
	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
					 'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer   = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_class])),
					 'biases':tf.Variable(tf.random_normal([n_class]))}


	# Forward Propogation 				 
	l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])				 
 	# relu works similar to a thresholding function 
 	l1 = tf.nn.relu(l1)			 
			 
	# Same fundamentals as l1, but input is l1
	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])		 
	l2 = tf.nn.relu(l2)

	# Same fundamentals as l1, but input is l2
	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])		 
	l3 = tf.nn.relu(l3)

	# defining the output layer, cascaded from l3
	output = tf.matmul(l3,output_layer['weights'])+output_layer['biases'] 

	return output			# Output layer

# here we define the training function for NN. Input is training data
def train_neural_network(x):

	# Build the Neural Network
	prediction = neural_network_model(x)

	#Defining the cost function
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))

	#Defining the Optimizer (for backwards propogation)
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	#No of times you want to do back run back propogations
	hm_epochs = 10

	with tf.Session() as sess:
#		sess.run(tf.initialize_all_variables())			# initialize all variables
		sess.run(tf.global_variables_initializer())		# initialize all variables

		# begin training the neurons
		for epoch in range (hm_epochs):
			epoch_loss = 0;
			
			pl = int(mnist.train.num_examples/batch_size)

			for _ in range (pl):
				epoch_x,epoch_y = mnist.train.next_batch(batch_size)			# train batch
				_,c = sess.run([optimizer,cost],feed_dict= {x:epoch_x,y:epoch_y})	# Optimize and find cost
				
				epoch_loss += c		# find the loss for the current epoch (accumulate)

			print('Epoch:',epoch,'/',hm_epochs,': loss:',epoch_loss)

		correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))	
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy:',accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))

train_neural_network(x)