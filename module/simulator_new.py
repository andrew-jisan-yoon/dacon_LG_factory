import os
import pandas as pd
import numpy as np
import time
import math
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1].__str__() + "/"

class Simulator:
    def __init__(self):
        """
        Prepares Simulator settings.
        Caution : All time data are str type.
        """
        # Metadata for items
        self.PRT = {'types' : ['PRT_1', 'PRT_2', 'PRT_3', 'PRT_4'],
                    'yield_rate' : 0.985, 'elapsed_hours' : 23 * 24
                   }
        self.MOL = {'types' : ['MOL_1', 'MOL_2', 'MOL_3', 'MOL_4'],
                    'yield_rate' : 0.975, 'elapsed_hours' : 48,
                    'prod_lines' : {'A': ['Event_A', 'MOL_A'],
                                    'B': ['Event_B', 'MOL_B']
                                   }
                   }
        
        cut_yield = pd.read_csv(root_dir + "data/cut_yield.csv")  # Monthly
        cut_yield.rename(columns={'date':'month'}, inplace=True)
        cut_yield['month'] = cut_yield['month'].astype(str).str.replace("2020", "2020-")
        cut_yield.set_index('month', inplace=True)
        
        self.BLK = {'types' : ['BLK_1', 'BLK_2', 'BLK_3', 'BLK_4'],
                    'yield_rate' : cut_yield,
                    'prod_rate' : {'BLK_1':506, 'BLK_2':506, 'BLK_3':400, 'BLK_4':400}
                   }
        # =========================
        
        # Preparing `stock` dataframe
        self.stock = pd.read_csv(root_dir + "data/stock.csv")  # Hourly
        hours = pd.read_csv(root_dir + "data/sample_submission.csv", usecols=['time'])
        hours.rename(columns = {'time':'hour'}, inplace=True)
        self.stock = self.stock.reindex(hours.index)
        self.stock.insert(0, 'hour', hours['hour'])
        
        # Preprocessing `order` dataframe
        self.order = pd.read_csv(root_dir + "data/order.csv")  # Daily
        order_blks = self.order[self.BLK['types']]
        valid_order_idxs = order_blks[order_blks.any(axis='columns')].index
        self.order = self.order.loc[valid_order_idxs]
        self.order['time'] = self.order['time'] + " 18:00:00"
        self.order.rename(columns={'time':'shipping_time'}, inplace=True)
        
        # Preparing `max_count` dataframe
        self.max_count = pd.read_csv(root_dir + "data/max_count.csv")  # Daily
        self.max_count.set_index('date', inplace=True)
        
        # Preparing `change_time` dataframe
        self.change_time = pd.read_csv(root_dir + "data/change_time.csv")  # Time-independent
    
    def get_score(self, submission):
        """
        Evaluates the given 'submission'
        This function does not edit instance attributes.
        
        Args:
            submission(pandas.DataFrame) : input schedule
        Returns:
            score(float)
        """
        schedule = submission.copy(deep=True)
        schedule.rename(columns={'time':'hour'}, inplace=True)
        
        end_idx = schedule.index[-1]
        
        stock = self.stock.copy(deep=True)
        order = self.order.copy(deep=True)

        assert schedule.index.identical(stock.index), "submitted schedule does not have expected number of rows"
        
        # Records delta PRT on `stock` dataframe
        for schedule_idx, row in schedule[self.PRT['types']].iterrows():
            if schedule_idx + self.PRT['elapsed_hours'] > end_idx:
                break

            for col_label, item_input in row.iteritems():
                if item_input > 0:
                    rate = self.PRT['yield_rate']
                    completed_amount = np.sum(np.random.choice(2, item_input, p=[1-rate, rate]))
                    stock.loc[schedule_idx + self.PRT['elapsed_hours'], col_label] = completed_amount
        
        # Records delta MOL on `stock` dataframe
        mol_prods = [prod_attr for prod_status in self.MOL['prod_lines'].values() for prod_attr in prod_status]
        line_items = dict.fromkeys(self.MOL['prod_lines'].keys())
        for schedule_idx, row in schedule[mol_prods].iterrows():
            if schedule_idx + self.MOL['elapsed_hours'] > end_idx:
                break
            
            for line in self.MOL['prod_lines']:
                if row[f'Event_{line}'].startswith(r"CHECK") or row[f'Event_{line}'].startswith(r"CHANGE"):
                    line_items[line] = row[f'Event_{line}'][-1]
                
                elif (row[f'Event_{line}'] == 'PROCESS') and (row[f'MOL_{line}'] > 0):
                    # Consume the specified amount of corresponding PRT
                    if stock.loc[schedule_idx, f'PRT_{line_items[line]}'] is None:
                        stock.loc[schedule_idx, f'PRT_{line_items[line]}'] = 0 - row[f'MOL_{line}']
                    else:
                        stock.loc[schedule_idx, f'PRT_{line_items[line]}'] -= row[f'MOL_{line}']
                    
                    rate = self.MOL['yield_rate']
                    completed_amount = row[f'MOL_{line}'] * rate
                    stock.loc[schedule_idx + self.MOL['elapsed_hours'], f'MOL_{line_items[line]}'] = completed_amount
        
        # Calculate PRT and MOL stocks cumulatively
        for item_type, item_stock in stock[self.PRT['types'] + self.MOL['types']].iteritems():
            delta_idxs = item_stock[item_stock.notna()].index
            for idx_pos in range(1, len(delta_idxs)):
                curr_idx, prev_idx = delta_idxs[idx_pos], delta_idxs[idx_pos - 1]
                stock.loc[curr_idx, item_type] += stock.loc[prev_idx, item_type]
        stock[self.PRT['types'] + self.MOL['types']] = stock[self.PRT['types'] + self.MOL['types']].fillna(method='ffill')

        """
        To do:
            1. BLKs should be generated when orders are made.
        """
        # return score
        return schedule, stock

if __name__ == "__main__":
    sim = Simulator()
    sample = pd.read_csv(root_dir + "Dacon_baseline.csv")
    
    schedule, stock = sim.get_score(sample)