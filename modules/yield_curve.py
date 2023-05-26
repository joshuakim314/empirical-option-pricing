import pandas as pd
import numpy as np
import datetime
from collections import OrderedDict
from bisect import bisect
from pathlib import Path


# 2 weeks, 1 month, 2 months, 3 months, 4 months, 5 months, 6 months, 9 months, 
# 1 year, 1.5 years, 2 years, 2.5 years, 3 years, 4 years, 5 years, 7 years, 9 years
STANDARD_YC_DAYS = [
    14, 30, 60, 91, 122, 152, 182, 273, 365, 
    547, 730, 912, 1095, 1460, 1825, 2555, 3285
]

class YieldCurve():
    def __init__(self, date, input_yc, standard_yc_days=STANDARD_YC_DAYS) -> None:
        self.date = date
        self.input_yc = input_yc  # {n_days: rate}
        self.standard_yc_days = standard_yc_days
        self.standard_yc = self.standardize_yc()
    
    def __call__(self):
        return self.yc_to_numpy(self.standard_yc)
    
    def infer(self, n_days):
        # extrapolation
        if n_days < min(self.input_yc.keys()) or n_days > max(self.input_yc.keys()):
            raise NotImplementedError(f"Extrapolation not implemented.")
        
        # existing data
        elif n_days in self.input_yc.keys():
            return self.input_yc[n_days]
        
        # interpolation
        else:
            input_yc_days = sorted(self.input_yc.keys())
            index = bisect(input_yc_days, n_days)
            l, r = input_yc_days[index-1], input_yc_days[index]
            alpha = (r - n_days) / (r - l)
            return alpha*self.input_yc[l] + (1-alpha)*self.input_yc[r]
    
    def standardize_yc(self):
        standard_yc = OrderedDict()
        for n_days in self.standard_yc_days:
            standard_yc[n_days] = self.infer(n_days)
        return standard_yc
    
    @staticmethod
    def yc_to_numpy(yc_dict):
        # this method assumes `yc_dict` is properly ordered
        return np.array(list(yc_dict.values()))


class YieldSurface():
    def __init__(self, input_ys, standard_yc_days=STANDARD_YC_DAYS) -> None:
        self.input_ys = input_ys  # {date: {n_days: rate}}
        self.standard_yc_days = standard_yc_days
        self.standard_ys = self.standardize_ys()
    
    def __call__(self):
        return self.ys_to_numpy(self.standard_ys)
    
    def standardize_ys(self):
        standard_ys = OrderedDict()
        for date in sorted(self.input_ys.keys()):
            standard_yc = YieldCurve(date, self.input_ys[date], self.standard_yc_days)
            standard_ys[date] = standard_yc
        return standard_ys
    
    @staticmethod
    def ys_to_numpy(ys_dict):
        # this method assumes `ys_dict` is properly ordered
        return np.stack([yc() for yc in ys_dict.values()])


if __name__ == '__main__':
    # notes:
    # 1. data before '2001-07-05' or after '2021-12-07' can be weird
    # 2. the maximum of shortest maturities is 10 days
    #       except the data between '2001-07-05' and '2004-03-26' 
    #       and data points on '2007-06-07' and '2007-06-08'
    #       which may have 11 days as shortest maturities
    # 3. the minimum of longest maturities is 3547 days
    
    DATA_FOLDER = Path("/Users/joshuakim/Desktop/Graduate/Research/Empirical Option Pricing/empirical-option-pricing/data")
    zcyc_df = pd.read_csv(DATA_FOLDER / "zero_coupon_yc.csv", encoding='utf-8')
    zcyc_dict = dict()
    for date, group in zcyc_df.groupby("date"):
        if datetime.date(2001, 7, 5) <= datetime.date.fromisoformat(date) <= datetime.date(2021, 12, 7):
            zcyc_dict[datetime.date.fromisoformat(date)] = dict(zip(group.days, group.rate))
    
    ys = YieldSurface(zcyc_dict, STANDARD_YC_DAYS)
    print(ys())
    breakpoint()
