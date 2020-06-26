import os
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1].__str__() + "/"

class Simulator:
    def __init__(self):
        self.sample_submission = pd.read_csv(root_dir + "data/sample_submission.csv")
        self.max_count = pd.read_csv(root_dir + "data/max_count.csv")
        self.stock = pd.read_csv(root_dir + "data/stock.csv")
        order = pd.read_csv(root_dir + "data/order.csv", index_col=0)
        order.index = pd.to_datetime(order.index)
        self.order = order

    def get_state(self, data):
        if 'CHECK' in data:
            return int(data[-1])
        elif 'CHANGE' in data:
            return int(data[-1])
        else:
            return np.nan

    def cal_schedule_part_1(self, df):
        """
        Records how many of PRTS would be generated at what time, according to df.
        """
        columns = ['PRT_1', 'PRT_2', 'PRT_3', 'PRT_4']
        df_set = df[columns]
        df_out = df_set * 0  # ambiguity : why make them all zero?
        # to make use of index column -> better to save it as template

        p = 0.985  # Yield rate of PRTs
        dt = pd.Timedelta(days=23)  # PRT processing time
        end_time = df_out.index[-1]

        for time in df_out.index:
            out_time = time + dt
            if end_time < out_time:
                break
            else:
                for column in columns:
                    set_num = df_set.loc[time, column]
                    if set_num > 0:
                        out_num =\
                            np.sum(np.random.choice(2, set_num, p=[1-p, p]))
                        df_out.loc[out_time, column] = out_num

        df_out['MOL_1'] = 0.0
        df_out['MOL_2'] = 0.0
        df_out['MOL_3'] = 0.0
        df_out['MOL_4'] = 0.0
        df_out['BLK_1'] = 0.0
        df_out['BLK_2'] = 0.0
        df_out['BLK_3'] = 0.0
        df_out['BLK_4'] = 0.0
        return df_out

    def cal_schedule_part_2(self, df, line='A'):
        if line == 'A':
            columns = ['Event_A', 'MOL_A']
        elif line == 'B':
            columns = ['Event_B', 'MOL_B']
        else:
            columns = ['Event_A', 'MOL_A']

        schedule = df[columns].copy()

        # state represents item number being processed
        schedule['state'] = 0
        schedule['state'] =\
            schedule[columns[0]].apply(lambda x: self.get_state(x))
        schedule['state'] = schedule['state'].fillna(method='ffill')
        schedule['state'] = schedule['state'].fillna(0)

        schedule_process = schedule.loc[schedule[columns[0]] == 'PROCESS']
        df_out = schedule.drop(schedule.columns, axis=1)
        df_out['PRT_1'] = 0.0
        df_out['PRT_2'] = 0.0
        df_out['PRT_3'] = 0.0
        df_out['PRT_4'] = 0.0
        df_out['MOL_1'] = 0.0
        df_out['MOL_2'] = 0.0
        df_out['MOL_3'] = 0.0
        df_out['MOL_4'] = 0.0

        p = 0.975  # Yield rate of MOL
        times = schedule_process.index
        for i, time in enumerate(times):
            value = schedule.loc[time, columns[1]]  # MOL input
            state = int(schedule.loc[time, 'state'])  # MOL input type
            df_out.loc[time, 'PRT_'+str(state)] = -value  # reduce PRT
            if i+48 < len(times):  # 48 hours = MOL process time
                out_time = times[i+48]
                # MOL output 48 hours later
                df_out.loc[out_time, 'MOL_'+str(state)] = value*p

        df_out['BLK_1'] = 0.0
        df_out['BLK_2'] = 0.0
        df_out['BLK_3'] = 0.0
        df_out['BLK_4'] = 0.0
        return df_out

    def cal_stock(self, df, df_order):
        df_stock = df * 0

        blk2mol = {}
        blk2mol['BLK_1'] = 'MOL_1'
        blk2mol['BLK_2'] = 'MOL_2'
        blk2mol['BLK_3'] = 'MOL_3'
        blk2mol['BLK_4'] = 'MOL_4'

        # daily production rate
        cut = {}
        cut['BLK_1'] = 506
        cut['BLK_2'] = 506
        cut['BLK_3'] = 400
        cut['BLK_4'] = 400

        # cut yield
        p = {}
        p['BLK_1'] = 0.851
        p['BLK_2'] = 0.901
        blk_diffs = []
        for i, time in enumerate(df.index):
            month = time.month
            if month == 4:
                p['BLK_3'] = 0.710
                p['BLK_4'] = 0.700
            elif month == 5:
                p['BLK_3'] = 0.742
                p['BLK_4'] = 0.732
            elif month == 6:
                p['BLK_3'] = 0.759
                p['BLK_4'] = 0.749
            else:
                p['BLK_3'] = 0.0
                p['BLK_4'] = 0.0

            if i == 0:
                df_stock.iloc[i] = df.iloc[i]
            else:
                # accumulate stocks
                df_stock.iloc[i] = df_stock.iloc[i-1] + df.iloc[i]
                for column in df_order.columns:  # column : BLK 1~4
                    val = df_order.loc[time, column]  # orders by hours
                    if val > 0:  # if there is an order
                        mol_col = blk2mol[column]
                        # MOL produced at time
                        mol_num = df_stock.loc[time, mol_col]
                        df_stock.loc[time, mol_col] = 0  # needless line

                        # blk generated at time
                        blk_gen = int(mol_num*p[column]*cut[column])
                        blk_stock = df_stock.loc[time, column] + blk_gen
                        # stock - order
                        blk_diff = blk_stock - val

                        df_stock.loc[time, column] = blk_diff
                        blk_diffs.append(blk_diff)
        return df_stock, blk_diffs

    def subprocess(self, df):
        """
        Converts 'time' column into datetime and set it as index
        """
        out = df.copy()
        column = 'time'

        out.index = pd.to_datetime(out[column])
        out = out.drop([column], axis=1)
        out.index.name = column
        return out

    def add_stock(self, df, df_stock):
        df_out = df.copy()
        for column in df_out.columns:
            df_out.iloc[0][column] =\
                df_out.iloc[0][column] + df_stock.iloc[0][column]
        return df_out

    def order_rescale(self, df, df_order):
        df_rescale = df.drop(df.columns, axis=1)
        dt = pd.Timedelta(hours=18)  # represents 6 pm (deployment time)
        for column in ['BLK_1', 'BLK_2', 'BLK_3', 'BLK_4']:
            for time in df_order.index:
                df_rescale.loc[time+dt, column] = df_order.loc[time, column]
        df_rescale = df_rescale.fillna(0)
        return df_rescale

    def cal_score(self, blk_diffs):
        """
        Parameters :
            blk_diffs : list of int
        """
        # Block Order Difference
        blk_diff_m = 0
        blk_diff_p = 0
        for item in blk_diffs:
            if item < 0:
                blk_diff_m += abs(item)
            elif item > 0:
                blk_diff_p += abs(item)
        score = blk_diff_m + blk_diff_p
        return score

    def get_score(self, df):
        df = self.subprocess(df)
        out_1 = self.cal_schedule_part_1(df)
        out_2 = self.cal_schedule_part_2(df, line='A')
        out_3 = self.cal_schedule_part_2(df, line='B')
        out = out_1 + out_2 + out_3
        out = self.add_stock(out, self.stock)
        order = self.order_rescale(out, self.order)  # order now considers deployment
        out, blk_diffs = self.cal_stock(out, order)  # out = cumulative stocks
        score = self.cal_score(blk_diffs)
        return score, out
