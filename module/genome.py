import os
import pandas as pd
import numpy as np
from pathlib import Path
from module.simulator import Simulator
import warnings
warnings.filterwarnings(action='ignore')

simulator = Simulator()
root_dir = Path(__file__).resolve().parents[1].__str__() + "/"

submission_ini = pd.read_csv(root_dir + "data/sample_submission.csv")
submission_ini.loc[:, 'PRT_1':'PRT_4'] = 0

order_ini = pd.read_csv(root_dir + "data/order.csv")


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
        num_events = int(output_len_1 / 2)
        self.a_mask = np.zeros([num_events], np.bool)  # Checks available events
        self.b_mask = np.zeros([num_events], np.bool)  # Checks available events
        events = ['STOP', 'CHECK_1', 'CHECK_2', 'CHECK_3', 'CHECK_4', 'PROCESS']
        change_events = [f"CHANGE_{from_mol}{to_mol}" for from_mol in range(1, 5) for to_mol in range(1, 5) if from_mol != to_mol]
        events.extend(change_events)
        assert (len(events) == num_events), "output_len_1 does not correspond to number of events"
        self.event_map = dict(zip(range(num_events), events))

        # Reading change_time.csv
        self.change_duration = pd.read_csv("data/change_time.csv")
        self.change_duration['event'] = "CHANGE_" + self.change_duration['from'].str[-1] + self.change_duration['to'].str[-1]
        self.change_duration.set_index('event', inplace=True)
        self.change_duration.drop(columns=['from', 'to'], inplace=True)
        
        # Status parameters of line A
        self.a_check_time = 28    # CHECK -1/hr, 28 if process_time >=98
        self.a_process_ready = False  # False if CHECK is required else True
        self.a_process_mode = 0   # Represents ongoing event in event_map; 0 represents STOP
        self.a_process_time = 0   # PROCESS +1/hr, CHANGE +1/hr, 140 at max
        self.a_change_time = 0    # CHANGE +1/hr, cap at change_duration
        self.a_change_gap = 0   # +1 when CHANGE ends, -1/hr afterwards, should be 0 when CHANGE starts
        
        # Status parameters of Line B
        self.b_check_time = 28    # CHECK -1/hr, 28 if process_time >=98
        self.b_process_ready = False  # False if CHECK is required else True
        self.b_process_mode = 0   # Represents ongoing event in event_map; 0 represents STOP
        self.b_process_time = 0   # PROCESS +1/hr, CHANGE +1/hr, 140 at max
        self.b_change_time = 0    # CHANGE +1/hr, cap at change_duration
        self.b_change_gap = 0   # +1 when CHANGE ends, -1/hr afterwards, should be 0 when CHANGE starts
        
    def update_mask(self):
        """Update mask based on status parameters"""
        self.a_mask[:] = False
        if self.a_process_ready is False:
            if self.a_check_time == 28:
                self.a_mask[1:5] = True  # allows any of CHECK events
            elif self.a_check_time < 28:
                self.a_mask[self.a_process_mode] = True  # enforces CHECK event in progress
        else:
            self.a_mask[5] = True  # enforces PROCESS event
            if self.a_process_time > 98:
                self.a_mask[1:5] = True  # allows CHECK event when PROCESS lasted 98 hours
            if self.a_change_gap == 0:
                if self.a_change_time == 0:  # if CHANGE did not start yet
                    # process_mode is not an index of CHANGE event
                    # PROCESS does not update process_mode
                    from_mol = self.event_map[self.a_process_mode][-1]
                    for event_idx, event in self.event_map.items():
                        if event.startswith(f"CHANGE_{from_mol}"):
                            duration = self.change_duration.loc[event, 'time']
                            if self.a_process_time <= (140 - duration):
                                self.a_mask[event_idx] = True
                    
                elif self.a_process_mode > 5:  # if CHANGE already started
                    # process_mode is an index of CHANGE event
                    self.a_mask[:] = False
                    self.a_mask[self.a_process_mode] = True

        
        self.b_mask[:] = False
        if self.b_process_ready is False:
            if self.b_check_time == 28:
                self.b_mask[1:5] = True  # ambiguity : corresponds to event_map
            elif self.b_check_time < 28:
                self.b_mask[self.b_process_mode] = True  # ambiguity : 0 and STOP
        else:
            self.b_mask[5] = True  # ambiguity : corresponds to event_map
            if self.b_process_time > 98:
                self.b_mask[1:5] = True
            if self.b_change_gap == 0:
                if self.b_change_time == 0:  # if CHANGE did not start yet
                    # process_mode is not an index of CHANGE event
                    # PROCESS does not update process_mode
                    from_mol = self.event_map[self.b_process_mode][-1]
                    for event_idx, event in self.event_map.items():
                        if event.startswith(f"CHANGE_{from_mol}"):
                            duration = self.change_duration.loc[event, 'time']
                            if self.b_process_time <= (140 - duration):
                                self.b_mask[event_idx] = True
                    
                elif self.b_process_mode > 5:  # if CHANGE already started
                    # process_mode is an index of CHANGE event
                    self.b_mask[:] = False
                    self.b_mask[self.b_process_mode] = True

    def forward(self, inputs):
        """Feed-forward Event NN and MOL stock NN
        
        Args:
            inputs(numpy.array): BLK demands over a month
                                 shape = (input_len, )
        Returns:
            out1(str): one element from
                       ['CHECK_1', 'CHECK_2', 'CHECK_3', 'CHECK_4', 'PROCESS']
            out2(int): MOL input amount (valid only when event_a == 'PROCESS')
        """
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
        
        net_a, net_b = np.split(net, 2)
        
        net_a = net_a * self.a_mask
        net_b = net_b * self.b_mask
        
        event_a = self.event_map[np.argmax(net_a)]
        event_b = self.event_map[np.argmax(net_b)]

        # MOL stock NN
        net = np.matmul(inputs, self.w5)
        net = self.linear(net)
        net = np.matmul(net, self.w6)
        net = self.linear(net)
        net = np.matmul(net, self.w7)
        net = self.sigmoid(net)
        net = np.matmul(net, self.w8)
        net = self.softmax(net)
        
        net_a, net_b = np.split(net, 2)
        mol_a = np.argmax(net_a) / 2
        mol_b = np.argmax(net_b) / 2
        
        return event_a, mol_a, event_b, mol_b

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def linear(self, x):
        return x

    def create_order(self, order):
        """
        Adds one more month worth of dummy data to order
        
        Args:
            order(pandas.DataFrame): order.csv with shape (91, 5)
        Returns:
            order(pandas.DataFrame): order.csv with shape (121, 5)
        """
        for i in range(30):
            order.loc[91+i, :] = ['0000-00-00', 0, 0, 0, 0]
        return order

    def predict(self, order):
        """
        Generates a schedule based on the order
        
        Args:
            order(pandas.DataFrame): order.csv with shape (91, 5)
        Returns:
            self.submission(pandas.DataFrame): predictions
        Side-effects:
            self.submission(pandas.DataFrame) : from sample_submission.csv to predictions
            self.a_process_time(int) : repeatedly reassigned
            self.a_process_ready(bool) : repeatedly reassigned
            self.a_check_time(int) : repeatedly reassigned
            self.a_process_mode(int) : repeatedly reassigned
        """
        order = self.create_order(order)
        self.submission = submission_ini
        # run a loop row by row
        for s in range(self.submission.shape[0]):
            self.update_mask()
            inputs = np.array(order.loc[s//24:(s//24+30), 'BLK_1':'BLK_4']).\
                reshape(-1)
            inputs = np.append(inputs, s % 24)
            event_a, mol_a, event_b, mol_b = self.forward(inputs)

            if self.a_change_gap > 0:
                    self.a_change_gap -= 1

            if event_a.startswith('CHECK_'):
                if self.a_process_ready is True:
                    self.a_process_ready = False
                    self.a_check_time = 28
                self.a_check_time -= 1
                self.a_process_mode = int(event_a[-1])
                if self.a_check_time == 0:
                    self.a_process_ready = True
                    self.a_process_time = 0
            elif event_a == 'PROCESS':
                self.a_process_time += 1
                if self.a_process_time == 140:
                    self.a_process_ready = False
                    self.a_check_time = 28         
            elif event_a.startswith("CHANGE_"):
                self.a_process_time += 1
                self.a_change_time += 1
                if self.a_process_time == 140:
                    self.a_process_ready = False
                    self.a_check_time = 28
                if self.a_change_time == 1:
                    for idx, event in self.event_map.items():
                        if event == event_a:
                            self.a_process_mode = idx
                elif self.a_change_time == self.change_duration.loc[self.event_map[self.a_process_mode], 'time']:
                    self.a_change_time = 0
                    self.a_change_gap = 1
                    self.a_process_mode = int(event_a[-1])
            
            # Line B
            if self.b_change_gap > 0:
                self.b_change_gap -= 1
            
            if event_b.startswith('CHECK_'):
                if self.b_process_ready is True:
                    self.b_process_ready = False
                    self.b_check_time = 28
                self.b_check_time -= 1
                self.b_process_mode = int(event_b[-1])
                if self.b_check_time == 0:
                    self.b_process_ready = True
                    self.b_process_time = 0
            elif event_b == 'PROCESS':
                self.b_process_time += 1
                if self.b_process_time == 140:
                    self.b_process_ready = False
                    self.b_check_time = 28
            elif event_b.startswith("CHANGE_"):
                self.b_process_time += 1
                self.b_change_time += 1
                if self.b_process_time == 140:
                    self.b_process_ready = False
                    self.b_check_time = 28
                if self.b_change_time == 1:
                    for idx, event in self.event_map.items():
                        if event == event_b:
                            self.b_process_mode = idx
                elif self.b_change_time == self.change_duration.loc[self.event_map[self.b_process_mode], 'time']:
                    self.b_change_time = 0
                    self.b_change_gap = 1
                    self.b_process_mode = int(event_b[-1])

            self.submission.loc[s, 'Event_A'] = event_a
            if self.submission.loc[s, 'Event_A'] == 'PROCESS':
                self.submission.loc[s, 'MOL_A'] = mol_a
            else:
                self.submission.loc[s, 'MOL_A'] = 0
            
            self.submission.loc[s, 'Event_B'] = event_b
            if self.submission.loc[s, 'Event_B'] == 'PROCESS':
                self.submission.loc[s, 'MOL_B'] = mol_b
            else:
                self.submission.loc[s, 'MOL_B'] = 0

        # MOL_A, MOL_B = 0, 0 for the first 23 days
        self.submission.loc[:24*23, 'MOL_A'] = 0
        self.submission.loc[:24*23, 'MOL_B'] = 0

        # Resetting status parameters
        self.a_check_time = 28
        self.a_process_ready = False
        self.a_process_mode = 0
        self.a_process_time = 0

        # Resetting status parameters
        self.b_check_time = 28
        self.b_process_ready = False
        self.b_process_mode = 0
        self.b_process_time = 0
        
        return self.submission


def genome_score(genome):
    """
    Run simulator to obtain genome score
    
    Args:
        genome(obj:Genome)
    Returns:
        genome(obj:Genome)
    Side-effects:
        genome.submission(pandas.DataFrame): from sample_submission.csv to predictions
        genome.score(float): assigned
        genome.process_time(int): repeatedly reassigned
        genome.process_ready(bool): repeatedly reassigned
        genome.check_time(int): repeatedly reassigned
        genome.process_mode(int): repeatedly reassigned
    """
    submission = genome.predict(order_ini)
    genome.submission = submission
    genome.score, _ = simulator.get_score(submission)
    return genome

if __name__ == "__main__":
    genome = Genome(0, 125, 17 * 2, 12 * 2)
    pass