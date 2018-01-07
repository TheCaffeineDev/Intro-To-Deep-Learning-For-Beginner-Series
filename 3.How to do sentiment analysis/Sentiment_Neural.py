# Here, about 1.2 MB of data produced a pickle of about 120MB Of data 100 times
    # training might take a lot of time on bigger data sets
    # saving the model as you train it helps here

import tensorflow as tf
from TF_own_data_model import create_feature_sets_and_labels
import numpy as np

# can load the data from the pickle
    # or you could just write it down
train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
    # 3 layers is probably good enough
        # no. of classes = 2
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
batch_size = 100  # can do batches of 100 images at a time
x = tf.placeholder('float', [None, len(train_x[0])])  # 28X28 = 784 pizels
y = tf.placeholder('float')
def neural_network(data):
    hidden_1_layer = {'weights': tf.Variable(tf.truncated_normal([len(train_x[0]), n_nodes_hl1], stddev=0.1)),
                      'biases': tf.Variable(tf.constant(0.1, shape=[n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2], stddev=0.1)),
                      'biases': tf.Variable(tf.constant(0.1, shape=[n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl2, n_nodes_hl3], stddev=0.1)),
                      'biases': tf.Variable(tf.constant(0.1, shape=[n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl3, n_classes], stddev=0.1)),
                    'biases': tf.Variable(tf.constant(0.1, shape=[n_classes])), }
    layer_1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    # now goes through an activation function - sigmoid function
    layer_1 = tf.nn.relu(layer_1)
    # input for layer 2 = result of activ_func for layer 1
    layer_2 = tf.add(tf.matmul(layer_1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    layer_3 = tf.nn.relu(layer_3)

    output = tf.matmul(layer_3, output_layer['weights']) + output_layer['biases']

    return output

    # now all we have to do is explain to TF, what to do with this model
    # need to specify how we want to run data through that model


def train_neural_network(x):
    prediction = neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)


    n_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # initializes our variables. Session has now begun.

        for epoch in range(n_epochs):
            epoch_loss = 0  # we'll calculate the loss as we go

            i = 0
            while i < len(train_x):
                #we want to take batches(chunks); take a slice, then another size)
                start = i
                end = i+batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i+=batch_size
            print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))


train_neural_network(x)
