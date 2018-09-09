'''
input > weight > hidden layer 1 (activation function) > weights > 
hidden layer 2 (activation function) > weights > output layer

compare output to intended output > using cost/loss function (cross entropy)

optimiztion function (optimizer) > minimize cost (Adamoptimizer.... SGD, AdaGrad)

backpropogation

feed forward + backprop = epoch
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/Users/gaddamnitish/Desktop/TensorFlow", one_hot=True)

n_nodes_hl1 = 500 #no of nodes in hidden layer1
n_nodes_hl2 = 500
n_nodes_hl3 = 500 #total 3 hidden layers

n_classes = 10
batch_size = 100 #batch of 100 images

#height * width
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
	
	# (input_data * weights) + biases

	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])), 
						'biases':tf.Variable(tf.random_normal(n_nodes_hl1))}

	hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 
						'biases':tf.Variable(tf.random_normal(n_nodes_hl2))}

	hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 
						'biases':tf.Variable(tf.random_normal(n_nodes_hl3))}

	output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 
						'biases':tf.Variable(tf.random_normal(n_classes))}

	# (input_data * weights) + biases

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l1 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)


	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
	return output

#model finished

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
	#learning_rate = 0.01
	optimizer = tf.train.Adamoptimizer().minimize(cost)
	#cyles of feedforward + back Prop
	hm_epochs = 10

	with tf.session() as sess:
		sess.run(tf.initilize_all_variables())

		for epoch in hm_epochs:
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				x, y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict={x: x, y: y})
				epoch_loss += c
			print('epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, float))
		print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_neural_network(x)




















