import os
import pandas as pd
import numpy as np
from pathlib import Path
from module.simulator import Simulator
simulator = Simulator()
root_dir = Path(__file__).resolve().parents[1]

submission_ini = pd.read_csv(root_dir + "/data/sample_submission.csv")
order_ini = pd.read_csv(root_dir + '/data/order.csv')


class Genome():
    def __init__(self, score_ini, input_len, output_len_1, output_len_2,
                 h1=50, h2=50, h3=50):
        # initializing score
        self.score = score_ini

        # number of nodes in hidden layers
        self.hidden_layer1 = h1
        self.hidden_layer2 = h2
        self.hidden_layer3 = h3

        # Generating weights for Event NN
        self.w1 = np.random.randn(input_len, self.hidden_layer1)
        self.w2 = np.random.randn(self.hidden_layer1, self.hidden_layer2)
        self.w3 = np.random.randn(self.hidden_layer2, self.hidden_layer3)
        self.w4 = np.random.randn(self.hidden_layer3, output_len_1)

        # Generating weights for MOL stock NN
        self.w5 = np.random.randn(input_len, self.hidden_layer1)
        self.w6 = np.random.randn(self.hidden_layer1, self.hidden_layer2)
        self.w7 = np.random.randn(self.hidden_layer2, self.hidden_layer3)
        self.w8 = np.random.randn(self.hidden_layer3, output_len_2)

        # Event categories
        self.mask = np.zeros([5], np.bool)  # Checks available events
        self.event_map = {0: 'CHECK_1', 1: 'CHECK_2', 2: 'CHECK_3',
                          3: 'CHECK_4', 4: 'PROCESS'}

        # Status parameters
        self.check_time = 28    # CHECK -1/hr, 28 if process_time >=98
        self.process_ready = False  # False if CHECK is required else True
        self.process_mode = 0   # Represents item in PROCESS; 0 represents STOP
        self.process_time = 0   # PROCESS +1/hr, CHANGE +1/hr, 140 at max

    def update_mask(self):
        self.mask[:] = False
        if self.process_ready is False:
            if self.check_time == 28:
                self.mask[:4] = True  # ambiguity : corresponds to event_map
            if self.check_time < 28:
                self.mask[self.process_mode] = True  # ambiguity : 0 and STOP
        if self.process_ready is True:
            self.mask[4] = True  # ambiguity : corresponds to event_map
            if self.process_time > 98:
                self.mask[:4] = True

    def forward(self, inputs):
        # Event NN
        net = np.matmul(inputs, self.w1)
        net = self.linear(net)
        net = np.matmul(net, self.w2)
        net = self.linear(net)
        net = np.matmul(net, self.w3)
        net = self.sigmoid(net)
        net = np.matmul(net, self.w4)
        net = self.softmax(net)
        net += 1
        net = net * self.mask
        out1 = self.event_map[np.argmax(net)]

        # MOL stock NN
        net = np.matmul(inputs, self.w5)
        net = self.linear(net)
        net = np.matmul(net, self.w6)
        net = self.linear(net)
        net = np.matmul(net, self.w7)
        net = self.sigmoid(net)
        net = np.matmul(net, self.w8)
        net = self.softmax(net)
        out2 = np.argmax(net)
        out2 /= 2
        return out1, out2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def linear(self, x):
        return x

    def create_order(self, order):
        for i in range(30):
            order.loc[91+i, :] = ['0000-00-00', 0, 0, 0, 0]
        return order

    def predict(self, order):
        order = self.create_order(order)
        self.submission = submission_ini
        self.submission.loc[:, 'PRT_1':'PRT_4'] = 0
        for s in range(self.submission.shape[0]):
            self.update_mask()
            inputs = np.array(order.loc[s//24:(s//24+30), 'BLK_1':'BLK_4']).\
                reshape(-1)
            inputs = np.append(inputs, s % 24)
            out1, out2 = self.forward(inputs)

            if out1 == 'CHECK_1':
                if self.process_ready is True:
                    self.process_ready = False
                    self.check_time = 28
                self.check_time -= 1
                self.process_mode = 0
                if self.check_time == 0:
                    self.process_ready = True
                    self.process_time = 0
            elif out1 == 'CHECK_2':
                if self.process_ready is True:
                    self.process_ready = False
                    self.check_time = 28
                self.check_time -= 1
                self.process_mode = 1
                if self.check_time == 0:
                    self.process_ready = True
                    self.process_time = 0
            elif out1 == 'CHECK_3':
                if self.process_ready is True:
                    self.process_ready = False
                    self.check_time = 28
                self.check_time -= 1
                self.process_mode = 2
                if self.check_time == 0:
                    self.process_ready = True
                    self.process_time = 0
            elif out1 == 'CHECK_4':
                if self.process_ready is True:
                    self.process_ready = False
                    self.check_time = 28
                self.check_time -= 1
                self.process_mode = 3
                if self.check_time == 0:
                    self.process_ready = True
                    self.process_time = 0
            elif out1 == 'PROCESS':
                self.process_time += 1
                if self.process_time == 140:
                    self.process_ready = False
                    self.check_time = 28

            self.submission.loc[s, 'Event_A'] = out1
            if self.submission.loc[s, 'Event_A'] == 'PROCESS':
                self.submission.loc[s, 'MOL_A'] = out2
            else:
                self.submission.loc[s, 'MOL_A'] = 0

        # MOL_A = 0 for the first 23 days
        self.submission.loc[:24*23, 'MOL_A'] = 0

        # Line A = Line B
        self.submission.loc[:, 'Event_B'] = self.submission.loc[:, 'Event_A']
        self.submission.loc[:, 'MOL_B'] = self.submission.loc[:, 'MOL_A']

        # Initializing status parameters
        self.check_time = 28
        self.process_ready = False
        self.process_mode = 0
        self.process_time = 0

        return self.submission


def genome_score(genome):
    submission = genome.predict(order_ini)
    genome.submission = submission
    genome.score, _ = simulator.get_score(submission)
    return genome
