"""
Test code for module/genome.py

Author : Cipher
"""

import pytest
from pytest_mock import mocker
import numpy as np

from module.genome import Genome, genome_score

args = {'score_ini':1e8, 'input_len':125, 'output_len_1':5, 'output_len_2':12, 'h1':50, 'h2':50, 'h3':50}

def test_init():
    genome = Genome(**args)
    assert (genome.score == args['score_ini'])
    assert (genome.hidden_layer1 == args['h1'])
    assert (genome.hidden_layer2 == args['h2'])
    assert (genome.hidden_layer3 == args['h3'])
    
    # Event NN
    assert (genome.w1.shape == (args['input_len'], args['h1']))
    assert (genome.w2.shape == (args['h1'], args['h2']))
    assert (genome.w3.shape == (args['h2'], args['h3']))
    assert (genome.w4.shape == (args['h3'], args['output_len_1']))
    
    # MOL stock NN
    assert (genome.w5.shape == (args['input_len'], args['h1']))
    assert (genome.w6.shape == (args['h1'], args['h2']))
    assert (genome.w7.shape == (args['h2'], args['h3']))
    assert (genome.w8.shape == (args['h3'], args['output_len_2']))
    
    # Event categories
    np.testing.assert_array_equal(genome.mask, np.zeros([5], np.bool), err_msg="genome.mask should be an array with five False's")
    assert (genome.event_map == {0: 'CHECK_1', 1: 'CHECK_2', 2: 'CHECK_3', 3: 'CHECK_4', 4: 'PROCESS'})
    
    # Status parameters
    assert (genome.check_time == 28)
    assert (genome.process_ready == False)
    assert (genome.process_mode == 0)
    assert (genome.process_time == 0)


trivials = {'process_mode':np.random.randint(0, 4 + 1)}
thresholds ={'check_time':28, 'process_time':98}
@pytest.mark.parametrize("process_ready, check_time, process_mode, process_time, expected",
                        [(False, thresholds['check_time'], trivials['process_mode'], thresholds['process_time'], np.array([True, True, True, True, False])),
                         (False, thresholds['check_time'] - 1, 0, thresholds['process_time'], np.array([True, False, False, False, False])),
                         (True, thresholds['check_time'], trivials['process_mode'], thresholds['process_time'], np.array([False, False, False, False, True])),
                         (True, thresholds['check_time'], trivials['process_mode'], thresholds['process_time'] + 1, np.array([True, True, True, True, True]))
                        ])
def test_update_mask(process_ready, check_time, process_mode, process_time, expected):
    genome = Genome(**args)
    genome.process_ready = process_ready
    genome.check_time = check_time
    genome.process_mode = process_mode
    genome.process_time = process_time
    
    genome.update_mask()
    np.testing.assert_array_equal(genome.mask, expected)