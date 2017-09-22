import tensorflow as tf
import numpy as np
import random
import pickle
from collections import deque


class BrainDQN:

    ACTION = 2
    FRAME_PER_ACTION = 1
    # decay rate of past observations
    GAMMA = 0.99
    # timesteps to observe before training
    OBSERVE = 30000000
    # framesover which to anneal epsilon
    explore = 300000
    # final value of epsilon
    FINAL_EPSILON = 0.0001
    # starting value of epsilon
    INITIAL_EPSILON = 0.0001
    # number previous transitions to remeber
    REPLAY_MEMORY = 50000
    # size of minibatch
    BATCH_SIZE = 32

    def __init__(self):
        self.replayMemory = deque()

        self.createQNetwork()

        self.timeStep = 0
        self.epsilon = self.INITIAL_EPSILON

    def createQNetwork(self):
        W_conv1 = self.weight_variable([8, 8, 4, 32])
        b_conv1 = self.bias_variable([32])

        W_conv2 = self.weight_variable([4, 4, 32, 64])
        b_conv2 = self.bias_variable([64])

        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])

        W_fc1 = self.weight_variable([1600, 512])
        b_fc1 = self.bias_variable([512])

        W_fc2 = self.weight_variable([512, self.ACTION])
        b_fc2 = self.bias_variable([self.ACTION])

        # input layer
        self.stateInput = tf.placeholder("float", [None, 80, 80, 4])

        # hidden layer
        h_conv1 = tf.nn.relu(
            self.conv2d(self.stateInput, W_conv1, 4) + b_conv1
        )
        h_pool1 = self.max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(
            self.conv2d(h_pool1, W_conv2, 2) + b_conv2
        )
        h_conv3 = tf.nn.relu(
            self.conv2d(h_conv2, W_conv3, 1) + b_conv3
        )

        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # Q value layer
        self.QValue = tf.matmul(h_fc1, W_fc2) + b_fc2

        self.actionInput = tf.placeholder("float", [None, self.ACTION])
        self.yInput = tf.placeholder("float", [None])

        Q_action = tf.reduce_sum(
            tf.multiply(self.QValue, self.actionInput),
            axis=1
        )

        self.cost = tf.reduce_mean(tf.square(self.yInput - Q_action))

        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

        # saving and loading networks
        self.saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()
        self.session = tf.InteractiveSession()
        self.session.run(init_op)
        checkpoint = tf.train.get_checkpoint_state('saved_networks')
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print('Successfully loaded:' + checkpoint.model_checkpoint_path)
        else:
            print('Could not find checkpoint')

    def trainQNetwork(self):
        minibatch = random.sample(self.replayMemory, self.BATCH_SIZE)
        state_batch, action_batch, reward_batch, nextState_batch = tuple(
            [data[i] for data in minibatch] for i in range(4)
        )

        y_batch = []
        QValue_batch = self.QValue.eval(feed_dict={
            self.stateInput: nextState_batch
        })
        for i in range(0, self.BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(
                    reward_batch[i] + self.GAMMA * np.max(QValue_batch[i])
                )
        self.trainStep.run(feed_dict={
            self.yInput: y_batch,
            self.actionInput: action_batch,
            self.stateInput: state_batch
        })

        # save network every 10000 iteration
        if self.timeStep % 2000 == 0:
            self.saver.save(
                self.session,
                'saved_networks/network-dqn',
                global_step=self.timeStep
            )

    def printStatus(self, other=''):
        print('[%d] now Stroed %d replays, eps = %.8f' %
              (
                  self.timeStep,
                  len(self.replayMemory),
                  self.epsilon,
              ) + ',' + other,
              end='\r')

    def setPerception(self, nextObservation, action, reward, terminal):
        newState = np.append(
            nextObservation,
            self.currentState[:, :, 1:],
            axis=2
        )
        self.replayMemory.append(
            (self.currentState, action, reward, newState, terminal)
        )
        self.printStatus()

        if len(self.replayMemory) > self.REPLAY_MEMORY:
            self.replayMemory.popleft()

        if self.timeStep > self.OBSERVE:
            self.trainQNetwork()

        self.currentState = newState
        self.timeStep += 1

    def getAction(self):
        QValue = self.QValue.eval(
            feed_dict={self.stateInput: [self.currentState]}
        )[0]
        action = np.zeros(self.ACTION)
        action_index = 0
        if self.timeStep % self.FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.ACTION)
                action[action_index] = 1
            else:
                action_index = np.argmax(QValue)
                self.printStatus("QValues is %s" % np.max(QValue))
                action[action_index] = 1
        else:
            action[0] = 1

        if self.epsilon > self.FINAL_EPSILON and self.timeStep > self.OBSERVE:
            self.epsilon -= (
                self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.explore

        return action

    def setInitState(self, observation):
        self.currentState = np.stack(
            # tuple(observation for i in range(4)),
            (observation, observation, observation, observation),
            axis=2
        )

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(
            x, W,
            strides=[1, stride, stride, 1],
            padding="SAME"
        )

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(
            x, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME"
        )
