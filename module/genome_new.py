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
    score_ini = 1e8
    input_len = 125
    output_len_1, output_len_2 = 5, 12
    event_map = {0: 'CHECK_1', 1: 'CHECK_2', 2: 'CHECK_3', 3: 'CHECK_4', 4: 'PROCESS'}
    # event_map = {0: 'STOP', 1: 'CHECK_1', 2: 'CHECK_2', 3: 'CHECK_3', 4: 'CHECK_4', 5: 'PROCESS', 6: 'CHANGE'}
    
    def __init__(self, h1=50, h2=50, h3=50):
        # initializing score
        self.score = score_ini

        # initializing mask to check available events
        self.event_mask = np.zeros([5], np.bool)
        
        # Status parameters of production lines
        # check_time : CHECK -1/hr, 28 if process_time >=98
        # process_ready : False if CHECK is required else True
        # process_mode : Represents item in PROCESS; 0 represents STOP
        # process_time : PROCESS +1/hr, CHANGE +1/hr, 140 at max
        status_params = {'check_time':28, 'process_ready':False, 'process_mode':0, 'process_time':0}
        self.line_a, self.line_b = status_params.copy(), status_params.copy()

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
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
    def predict(self):
        pass
