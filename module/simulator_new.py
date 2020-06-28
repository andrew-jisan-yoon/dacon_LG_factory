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
        cut_yield = cut_yield / 100
        
        self.BLK = {'types' : ['BLK_1', 'BLK_2', 'BLK_3', 'BLK_4'],
                    'yield_rate' : cut_yield,
                    'prod_ratio' : {'BLK_1':506, 'BLK_2':506, 'BLK_3':400, 'BLK_4':400}
                   }
        # =========================
        
        # Preparing `stock` and `delta` dataframes
        self.stock = pd.read_csv(root_dir + "data/stock.csv")  # Hourly
        hours = pd.read_csv(root_dir + "data/sample_submission.csv", usecols=['time'])
        hours.rename(columns = {'time':'hour'}, inplace=True)
        self.stock = self.stock.reindex(hours.index)
        self.stock.insert(0, 'hour', hours['hour'])
        
        self.delta = self.stock.copy(deep=True)
        self.delta.loc[0, self.delta.columns.difference(['hour'], sort=False)] = None
        self.delta.fillna(0, inplace=True)
        
        # Preprocessing `order` dataframe
        self.order = pd.read_csv(root_dir + "data/order.csv")  # Daily
        order_blks = self.order[self.BLK['types']]
        valid_order_idxs = order_blks[order_blks.any(axis='columns')].index
        self.order = self.order.loc[valid_order_idxs]
        self.order['time'] = self.order['time'] + " 18:00:00"
        self.order.rename(columns={'time':'shipping_time'}, inplace=True)

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
        delta = self.delta.copy(deep=True)
        order = self.order.copy(deep=True)

        assert schedule.index.identical(stock.index), "submitted schedule does not have expected number of rows"
        
        # Records delta PRT on `delta` dataframe
        for schedule_idx, row in schedule[self.PRT['types']].iterrows():
            if schedule_idx + self.PRT['elapsed_hours'] > end_idx:
                break

            for col_label, item_input in row.iteritems():
                if item_input > 0:
                    rate = self.PRT['yield_rate']
                    completed_amount = np.sum(np.random.choice(2, item_input, p=[1-rate, rate]))
                    delta.loc[schedule_idx + self.PRT['elapsed_hours'], col_label] += completed_amount
        
        # Records delta MOL on `delta` dataframe
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
                    delta.loc[schedule_idx, f'PRT_{line_items[line]}'] -= row[f'MOL_{line}']
                    
                    rate = self.MOL['yield_rate']
                    completed_amount = row[f'MOL_{line}'] * rate
                    delta.loc[schedule_idx + self.MOL['elapsed_hours'], f'MOL_{line_items[line]}'] += completed_amount
        
        # Ship BLKs upon order
        stock_items = stock.columns.difference(['hour'], sort=False)
        order_items = order.columns.difference(['shipping_time'], sort=False)
        blk_diffs = []
        for stock_idx, stock_info in stock.loc[1:].iterrows():
            stock.loc[stock_idx, stock_items] = stock.loc[stock_idx - 1, stock_items] + delta.loc[stock_idx, stock_items]
            query_order = order['shipping_time'].str.contains(stock_info['hour'])
            if any(query_order):
                order_amounts = order[query_order == True].iloc[0].loc[order_items]
                for blk_type, order_amount in order_amounts.iteritems():
                    if order_amount > 0:
                        mol_input = stock.loc[stock_idx, f'MOL_{blk_type[-1]}']
                        ratio = self.BLK['prod_ratio'][blk_type]
                        rate = self.BLK['yield_rate'].loc[stock_info['hour'][:7], blk_type]

                        # Consume corresponding MOL
                        stock.loc[stock_idx, f'MOL_{blk_type[-1]}'] = 0

                        # Adding new BLK to BLK stock and subtracting ordered amount
                        blk_gen = int(mol_input * ratio * rate)
                        stock.loc[stock_idx, blk_type] += blk_gen
                        stock.loc[stock_idx, blk_type] -= order_amount

                        # Recording the remaining BLK stock to `blk_diffs`
                        blk_diffs.append(stock.loc[stock_idx, blk_type])

        # Calculating the score
        """
        To do:
            1. Implement the evaluation metrdic used by the leaderboard
        """
        def f_xa(x, a):
            return 1 - (x/a) if x < a else 0
        
        N = order.iloc[:, 1:].sum().sum()
        p = sum([abs(blk_diff) for blk_diff in blk_diffs if blk_diff < 0])
        q = sum([blk_diff for blk_diff in blk_diffs if blk_diff > 0])
        
        score = 50 * f_xa(p, 10 * N) + 20 * f_xa(q, 10 * N) + 30
        
        # score = sum(map(abs, blk_diffs))
        # return score
        return schedule, delta, stock, blk_diffs, score

if __name__ == "__main__":
    sim = Simulator()
    sample = pd.read_csv(root_dir + "Dacon_baseline.csv")
    
    schedule, delta, stock, blk_diffs, score = sim.get_score(sample)
    
    from module.simulator import Simulator as Old_Simulator
    old_sim = Old_Simulator()
    
    old_score, old_out, old_blk_diffs = old_sim.get_score(sample)