import os
import pandas as pd
import numpy as np
from pathlib import Path
from module.simulator import Simulator

simulator = Simulator()
root_dir = Path(__file__).resolve().parents[1].__str__() + "/"

submission_ini = pd.read_csv(root_dir + "data/sample_submission.csv")
submission_ini.loc[:, submission_ini.columns.difference(['time', 'Event_A', 'Event_B'])] = 0

order_ini = pd.read_csv(root_dir + "data/order.csv")


class Genome():
    
    def __init__(self, score_ini=0, input_len=125, output_len_1=5 * 2, output_len_2=12 * 2,
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
        
        # Creating input schedule template
        schedule_cols = ['time', 'PRT_1', 'PRT_2', 'PRT_3', 'PRT_4', 'Event_A', 'MOL_A', 'Event_B', 'MOL_B']
        schedule_idx = pd.RangeIndex(91 * 24)
        self.schedule = pd.DataFrame(columns = schedule_cols, index = schedule_idx)

        # Event categories
        self.event_map = {0: 'CHECK_1', 1: 'CHECK_2', 2: 'CHECK_3', 3: 'CHECK_4', 4: 'PROCESS'}
        mask = np.zeros([int(output_len_1 / 2)], np.bool)  # Checks available events
        
        # Statues parameters of production lines
        params = {'check_time': 28, 'process_ready': False, 'process_type': 0, 'process_time': 0,
                  'mask': mask}
        self.prod_lines = {line: params for line in ['A', 'B']}

    def update_mask(self):
        for line in self.prod_lines:
            self.prod_lines[line]['mask'][:] = False
            
            if self.prod_lines[line]['process_ready'] is False:
                if self.prod_lines[line]['check_time'] == 28:
                    self.prod_lines[line]['mask'][0:4] = True
                elif self.prod_lines[line]['check_time'] < 28:
                    self.prod_lines[line]['mask'][self.prod_lines[line]['process_type']] = True
            else:
                self.prod_lines[line]['mask'][4] = True
                if self.prod_lines[line]['process_time'] > 98:
                    self.prod_lines[line]['mask'][0:4] = True
    
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
        
        net_a = net_a * self.prod_lines['A']['mask']
        event_a = self.event_map[np.argmax(net_a)]
        
        net_b = net_b * self.prod_lines['B']['mask']
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
        
        print(event_a, mol_a, event_b, mol_b)
        return event_a, mol_a, event_b, mol_b
    
    
    def sigmoid(self, x):
        x = np.array(x, dtype=np.float64)
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
        order = self.create_order(order)
        self.submission = submission_ini
        # run a loop row by row
        for s in range(self.submission.shape[0]):
            self.update_mask()
            inputs = np.array(order.loc[s//24:(s//24+30), 'BLK_1':'BLK_4']).\
                reshape(-1)
            inputs = np.append(inputs, s % 24)
            event_a, mol_a, event_b, mol_b = self.forward(inputs)
            
            events = {'A':event_a, 'B':event_b}
            mols = {'A':mol_a, 'B':mol_b}
            
            for line in self.prod_lines:
                if events[line].startswith("CHECK_"):
                    if self.prod_lines[line]['process_ready'] is True:
                        self.prod_lines[line]['process_ready'] = False
                        self.prod_lines[line]['check_time'] = 28
                    self.prod_lines[line]['check_time'] -= 1
                    self.prod_lines[line]['process_type'] = int(events[line][-1]) - 1
                    if self.prod_lines[line]['check_time'] == 0:
                        self.prod_lines[line]['process_ready'] = True
                        self.prod_lines[line]['process_time'] = 0

                elif events[line].startswith('PROCESS'):
                    print("PROCESS BRANCH traveled")
                    self.prod_lines[line]['process_time'] += 1
                    if self.prod_lines[line]['process_time'] == 140:
                        self.prod_lines[line]['process_ready'] = False
                        self.prod_lines[line]['check_time'] = 28
                
                self.submission.loc[s, f'Event_{line}'] = events[line]
                if self.submission.loc[s, f'Event_{line}'].startswith("PROCESS"):
                    self.submission.loc[s, f'MOL_{line}'] = mols[line]
                    print(f"{mols[line]} MOLs added to {line}")
                else:
                    self.submission.loc[s, f'MOL_{line}'] = 0
                
            self.submission.loc[:24*23, 'MOL_A'] = 0
            self.submission.loc[:24*23, 'MOL_B'] = 0
            
            for line in self.prod_lines:
                self.prod_lines[line]['check_time'] = 28
                self.prod_lines[line]['process_ready'] = False
                self.prod_lines[line]['process_type'] = 0
                self.prod_lines[line]['process_time'] = 0
        
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
    pass