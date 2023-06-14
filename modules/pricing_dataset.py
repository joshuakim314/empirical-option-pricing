import numpy as np
import pandas as pd
import datetime
from collections import OrderedDict
from pathlib import Path
import pickle
import torch
import torch.utils.data as data

from modules.tcn import TemporalConvNet
from modules.yield_curve import YieldCurve


STANDARD_VOL_SURFACE_DAYS = [10, 14, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730, 1825]
STANDARD_YC_DAYS = [14, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730, 912, 1095, 1460, 1825, 2555, 3285]


class PricingDataset(data.Dataset):
    def __init__(
        self, option_price_dict, security_price_dict, security_vol_dict, yc_dict, 
        date_range=None, lookback_days=500, sample_size=100, random_seed=2023
    ):
        super().__init__()
        self.option_price_dict = option_price_dict
        self.security_price_dict = security_price_dict
        self.security_vol_dict = security_vol_dict
        self.yc_dict = yc_dict
        
        self.date_range = date_range
        self.lookback_days = lookback_days
        self.sample_size = sample_size
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.current_device = torch.cuda.current_device()
        
        if date_range is None:
            # use the entire dataset
            self.dates = sorted(option_price_dict.keys())
        elif len(date_range) == 2:
            # date_range = (min_date, max_date)
            self.dates = [date for date in sorted(option_price_dict.keys()) if self.date_range[0] <= date <= self.date_range[1]]
        else:
            # assume `date_range` is the list of dates we want to use
            self.dates = date_range
        
        self.populate_ts()
        self.populate()
    
    def transform(self, array_or_tensor, dtype=torch.float32):
        if torch.is_tensor(array_or_tensor):
            tensor = array_or_tensor.type(dtype)
        else:
            tensor = torch.as_tensor(array_or_tensor).type(dtype)
        if self.cuda_available:
            return tensor.to(self.current_device)
        else:
            return tensor
    
    def populate_ts(self):
        # combine the following time-series:
        # 1. security_price_dict: adjusted close price, log-return
        # 2. security_vol_dict: historical volatility for each number of days in `STANDARD_VOL_SURFACE_DAYS`
        # 3. yc_dict: zero-coupon yield curve for each number of days in `STANDARD_YC_DAYS`
        self.ts_dict = dict()
        for date in self.dates:
            date_idx = list(self.option_price_dict.keys()).index(date)
            ts = []
            for lb_date in list(self.option_price_dict.keys())[date_idx-self.lookback_days:date_idx]:
                price_ret = np.array([self.security_price_dict[lb_date]["close"], self.security_price_dict[lb_date]["return"]])
                vols = np.array(list(self.security_vol_dict[lb_date].values()))
                yc = YieldCurve(lb_date, self.yc_dict[lb_date], STANDARD_YC_DAYS)
                ts.append(np.concatenate([price_ret, vols, yc()]))
            self.ts_dict[date] = np.stack(ts)
    
    def populate(self):
        self.sample_list = []
        for date in self.dates:
            options = np.random.choice(self.option_price_dict[date], self.sample_size, replace=False)
            self.sample_list.extend([(date, option) for option in options])
    
    def __getitem__(self, index):
        date, option = self.sample_list[index]
        option_data = np.array([option[feature] for feature in ['ex_days', 'strike_price', 'cp_flag', 'volume', 'open_interest']])
        return (self.transform(option_data), self.transform(self.ts_dict[date].T)), self.transform(option['impl_volatility'])
    
    def __len__(self):
        return len(self.sample_list)


if __name__ == '__main__':
    DATA_FOLDER = Path("/Users/joshuakim/Desktop/Graduate/Research/Empirical Option Pricing/empirical-option-pricing/data")
    
    # 'volume', 'open_interest', 'cfadj', 'am_settlement', 'contract_size', 'ss_flag', 'ticker', 'div_convention', 'exercise_style'
    option_df = pd.read_csv(DATA_FOLDER / "option_price_SPX.csv", encoding='utf-8')
    option_df["date"] = option_df["date"].apply(lambda date: datetime.date.fromisoformat(date))
    option_df["exdate"] = option_df["exdate"].apply(lambda date: datetime.date.fromisoformat(date))
    option_df["cp_flag"] = option_df["cp_flag"].apply(lambda cp: 0 if cp == 'C' else 1 if cp == 'P' else 2)
    option_df["exercise_style"] = option_df["exercise_style"].apply(lambda es: 0 if es == 'E' else 1 if es == 'A' else 2)
    option_df["ex_days"] = (option_df["exdate"] - option_df["date"]).apply(lambda ed: ed.days)
    option_df = option_df[option_df.impl_volatility.notnull()]
    option_dict = OrderedDict()
    for date, group in option_df.groupby("date"):
        if datetime.date(2004, 1, 1) <= date <= datetime.date(2020, 12, 31):
            option_dict[date] = group[['ex_days', 'strike_price', 'cp_flag', 'volume', 'open_interest', 'impl_volatility']].to_dict('records')
    
    # other features to consider: 'volume', 'cfadj', 'shrout', 'cfret'
    # we did not include it for SPX as these fields each have 1 unique value
    security_price_df = pd.read_csv(DATA_FOLDER / "hist_price_SPX.csv", encoding='utf-8')
    security_price_df["date"] = security_price_df["date"].apply(lambda date: datetime.date.fromisoformat(date))
    security_price_df.sort_values(by=["date"], ascending=True, inplace=True, ignore_index=True)
    security_price_dict = security_price_df[['date', 'close', 'return']].set_index('date').T.to_dict(into=OrderedDict)  # no need of adjustment factors for SPX
    
    security_vol_df = pd.read_csv(DATA_FOLDER / "hist_vol_SPX.csv", encoding='utf-8')
    security_vol_df["date"] = security_vol_df["date"].apply(lambda date: datetime.date.fromisoformat(date))
    security_vol_df.sort_values(by=["date", "days"], ascending=True, inplace=True, ignore_index=True)
    security_vol_dict = OrderedDict()
    for date, group in security_vol_df.groupby("date"):
        security_vol_dict[date] = OrderedDict(zip(group.days, group.volatility))
    
    # TODO: some misalignments in date between yield curves and other datasets until 2012
    zcyc_df = pd.read_csv(DATA_FOLDER / "zero_coupon_yc.csv", encoding='utf-8')
    zcyc_df["date"] = zcyc_df["date"].apply(lambda date: datetime.date.fromisoformat(date))
    zcyc_df.sort_values(by=["date", "days"], ascending=True, inplace=True, ignore_index=True)
    zcyc_dict = OrderedDict()
    for date, group in zcyc_df.groupby("date"):
        if datetime.date(2001, 7, 5) <= date <= datetime.date(2021, 12, 7):
            zcyc_dict[date] = OrderedDict(zip(group.days, group.rate))
    
    dataset = PricingDataset(option_dict, security_price_dict, security_vol_dict, zcyc_dict, (datetime.date(2015, 1, 1), datetime.date(2020, 12, 31)))
    breakpoint()
