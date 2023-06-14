import numpy as np
import pandas as pd
import datetime
from collections import OrderedDict
from pathlib import Path
import pickle
import dill
import itertools

import torch
import torch.utils.data as data

from modules.tcn import TemporalConvNet
from modules.yield_curve import YieldCurve
from modules.pricing_dataset import PricingDataset
from modules.pricing_model import PricingModel
from modules.pricing_model_config import PricingModelConfig
from modules.train_model import ModelTrainer


def train_pricing_model(config, train_set, eval_set, param_search_grid, random_seed=2023):
    np.random.seed(random_seed)
    available_devices = max(torch.cuda.device_count(), 1)
    print('available devices: ' + str(available_devices))
    if available_devices >= 1:
        torch.cuda.empty_cache()  # clear cache in case prior experiment did not finish properly
    
    train_loader = data.DataLoader(train_set, batch_size=config.batch_size*available_devices, drop_last=True, shuffle=True)
    eval_loader = data.DataLoader(eval_set, batch_size=config.batch_size*available_devices, drop_last=True, shuffle=True)
    score_loader = data.DataLoader(eval_set, batch_size=config.batch_size*available_devices, drop_last=False, shuffle=False)
    print(f"train_loader size: {len(train_loader)}")
    print(f"train_loader size: {len(eval_loader)}")
    
    param_search_list = []
    for key in param_search_grid:
        param_search_list.append([(key, value) for value in param_search_grid[key]])
    param_product = list(itertools.product(*param_search_list))
    np.random.shuffle(param_product)
    
    model_trainers = []
    res_dfs = []
    # loop through combinations of model params in random order
    for i, combination in enumerate(param_product):
        rs = i + random_seed
        torch.manual_seed(rs)
        for param_tuple in combination:
            config.__dict__[param_tuple[0]] = param_tuple[1]
            config.random_seed = rs
        print('Experiment combination: ' + str(combination))
        model = PricingModel(config)
        model_trainer = ModelTrainer(model, train_loader, eval_loader)
        print('Experiment id: ' + model_trainer.experiment_id)
        # clear GPU memory if using CUDA
        if available_devices >= 1:
            torch.cuda.empty_cache()
        model_trainer.train()
        res_df = model_trainer.score_model(score_loader)
        
        model_trainers.append(model_trainer)
        res_dfs.append(res_df)

    return model_trainer, res_df
    

if __name__ == '__main__':
    STANDARD_VOL_SURFACE_DAYS = [10, 14, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730, 1825]
    STANDARD_YC_DAYS = [14, 30, 60, 91, 122, 152, 182, 273, 365, 547, 730, 912, 1095, 1460, 1825, 2555, 3285]
    DATA_FOLDER = Path("/Users/joshuakim/Desktop/Graduate/Research/Empirical Option Pricing/empirical-option-pricing/data")

    cached = True
    if not cached:
        # 'volume', 'open_interest', 'cfadj', 'am_settlement', 'contract_size', 'ss_flag', 'ticker', 'div_convention', 'exercise_style'
        option_df = pd.read_csv(DATA_FOLDER / "option_price_SPX.csv", encoding='utf-8')  # mixed dtypes: ['expiry_indicator', 'root', 'suffix']
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
        with (DATA_FOLDER / "option_dict.pkl").open("wb") as f:
            pickle.dump(option_dict, f, pickle.HIGHEST_PROTOCOL)
        
        # other features to consider: 'volume', 'cfadj', 'shrout', 'cfret'
        # we did not include it for SPX as these fields each have 1 unique value
        security_price_df = pd.read_csv(DATA_FOLDER / "hist_price_SPX.csv", encoding='utf-8')
        security_price_df["date"] = security_price_df["date"].apply(lambda date: datetime.date.fromisoformat(date))
        security_price_df.sort_values(by=["date"], ascending=True, inplace=True, ignore_index=True)
        security_price_dict = security_price_df[['date', 'close', 'return']].set_index('date').T.to_dict(into=OrderedDict)  # no need of adjustment factors for SPX
        with (DATA_FOLDER / "security_price_dict.pkl").open("wb") as f:
            pickle.dump(security_price_dict, f, pickle.HIGHEST_PROTOCOL)
        
        security_vol_df = pd.read_csv(DATA_FOLDER / "hist_vol_SPX.csv", encoding='utf-8')
        security_vol_df["date"] = security_vol_df["date"].apply(lambda date: datetime.date.fromisoformat(date))
        security_vol_df.sort_values(by=["date", "days"], ascending=True, inplace=True, ignore_index=True)
        security_vol_dict = OrderedDict()
        for date, group in security_vol_df.groupby("date"):
            security_vol_dict[date] = OrderedDict(zip(group.days, group.volatility))
        with (DATA_FOLDER / "security_vol_dict.pkl").open("wb") as f:
            pickle.dump(security_vol_dict, f, pickle.HIGHEST_PROTOCOL)
        
        zcyc_df = pd.read_csv(DATA_FOLDER / "zero_coupon_yc.csv", encoding='utf-8')
        zcyc_df["date"] = zcyc_df["date"].apply(lambda date: datetime.date.fromisoformat(date))
        zcyc_df.sort_values(by=["date", "days"], ascending=True, inplace=True, ignore_index=True)
        zcyc_dict = OrderedDict()
        for date, group in zcyc_df.groupby("date"):
            if datetime.date(2001, 7, 5) <= date <= datetime.date(2021, 12, 7):
                zcyc_dict[date] = OrderedDict(zip(group.days, group.rate))
        with (DATA_FOLDER / "zcyc_dict.pkl").open("wb") as f:
            pickle.dump(zcyc_dict, f, pickle.HIGHEST_PROTOCOL)
    
    with (DATA_FOLDER / "option_dict.pkl").open("rb") as f:
        option_dict = pickle.load(f)
    with (DATA_FOLDER / "security_price_dict.pkl").open("rb") as f:
        security_price_dict = pickle.load(f)
    with (DATA_FOLDER / "security_vol_dict.pkl").open("rb") as f:
        security_vol_dict = pickle.load(f)
    with (DATA_FOLDER / "zcyc_dict.pkl").open("rb") as f:
        zcyc_dict = pickle.load(f)
    
    # train_set = PricingDataset(option_dict, security_price_dict, security_vol_dict, zcyc_dict, (datetime.date(2015, 1, 1), datetime.date(2018, 12, 31)))
    train_set = PricingDataset(option_dict, security_price_dict, security_vol_dict, zcyc_dict, (datetime.date(2015, 1, 1), datetime.date(2015, 1, 15)))
    # eval_set = PricingDataset(option_dict, security_price_dict, security_vol_dict, zcyc_dict, (datetime.date(2019, 1, 1), datetime.date(2020, 12, 31)))
    eval_set = PricingDataset(option_dict, security_price_dict, security_vol_dict, zcyc_dict, (datetime.date(2019, 1, 1), datetime.date(2019, 1, 15)))
    
    # with open(DATA_FOLDER / 'train_set.dill', 'wb') as handle:
    #     dill.dump(train_set, handle)        
    # with open(DATA_FOLDER / 'eval_set.dill', 'wb') as handle:
    #     dill.dump(eval_set, handle)
    # with open('data/train_set.dill', 'rb') as handle:
    #     train_set = dill.load(handle)
    # with open('data/eval_set.dill', 'rb') as handle:
    #     eval_set = dill.load(handle)
    
    # breakpoint()

    config = PricingModelConfig(
        model_type="TCN",
        tcn_input_size=32,
        seq_len=500,
        tabular_data_size=5,
        n_targets=1,
        tcn_num_channels=[200, 200, 200],
        tcn_kernel_size=3,
        tcn_dropout=0.1,
        linear_sizes=[256, 64],
        linear_dropout=0.2,
        batch_size=32,
        sample_size=100,
        optimizer=torch.optim.RAdam,
        loss_function=torch.nn.L1Loss(),
        learning_rate=1e-6,
        l2_lambda=0.0,
        max_epochs=10,  # 10
        accumulation_steps=20,
        evaluate_every_n_steps=40,  # 40
        consecutive_losses_to_stop=3
    )

    param_search_grid = {
        'tcn_num_channels': [[200, 200, 200]],
        'linear_sizes': [[256, 64]]
    }
    model_trainers = train_pricing_model(config, train_set, eval_set, param_search_grid)
    breakpoint()