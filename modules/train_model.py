import uuid
import time
import datetime
import os
from pathlib import Path
import json
import logging
import dill
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.utils.data as data


FOLDER = Path("/Users/joshuakim/Desktop/Graduate/Research/Empirical Option Pricing/empirical-option-pricing")
DATA_FOLDER = Path(FOLDER / "data")
LOG_FOLDER = Path(FOLDER / "mlruns")


class ModelTrainer():
    def __init__(self, model, train_loader, test_loader):
        self.experiment_id = uuid.uuid4().hex
        self.model = model
        self.config = model.config
        self.model_filename = self.config.model_type
        self.log_folder = LOG_FOLDER / f"{self.experiment_id}"
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
        with open(self.log_folder / "config.dill", 'wb') as handle:
            dill.dump(self.config, handle)
        self.log_file = self.log_folder / "train.log"
        self.log_setup()
        
        self.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.model_device)
        if torch.cuda.is_available():
            self.data_parallel = True if torch.cuda.device_count() > 1 else False
            self.device_count = max(torch.cuda.device_count(), 1)
            if self.device_count > 1:
                print("Using " + str(torch.cuda.device_count()) + " GPUs")
                self.model = nn.DataParallel(model)
                self.model = self.model.cuda()
            else:
                self.model = model.cuda()
        else:
            self.data_parallel = False
            self.device_count = 1

        self.optimizer = self.config.optimizer(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.l2_lambda
        )
        self.max_epochs = self.config.max_epochs
        self.init_counters()
    
    def log_setup(self):
        self.logger = self.create_logger(self.log_file)

    def init_counters(self):
        self.finished_training = False
        self.steps = 0
        self.recent_steps = 0
        self.test_steps = 0
        self.consecutive_losses_increasing = 0
        self.finished_training = False
        self.detailed_losses = {i: [] for i in range(self.config.n_targets)}
        self.train_losses = []
        self.test_losses = []

    def train(self):
        for epoch in range(0, self.max_epochs):
            self.running_loss = 0.0
            self.recent_loss = 0.0
            self.epoch_steps = 0
            self.epoch = epoch
            
            for inputs, labels in self.train_loader:
                self.steps += 1
                self.recent_steps += 1
                self.epoch_steps += 1

                # display info on the very first step
                if self.epoch == 0 and self.steps == 1:
                    print(f"Batch input shape is: {[input.shape for input in inputs]}")
                    if self.data_parallel:
                        print("Parallel devices being used: " + str(self.model.device_ids))
                
                model_output = self.model.forward(inputs)
                if len(model_output.shape) == 3:
                    model_output = torch.squeeze(model_output[:, -1, :], 1)

                # display info on the very first step
                if self.epoch == 0 and self.steps == 1:
                    print("Batch output shape is: ", model_output.size())
                
                print(labels.view(self.config.n_targets, -1))
                print(model_output.view(self.config.n_targets, -1))
                loss = self.config.loss_function(model_output.view(-1, self.config.n_targets), labels.view(-1, self.config.n_targets))
                self.logger.debug(f'n_steps_train: {self.steps}')
                self.logger.debug(f'train_batch_loss: {loss.item()}')
                
                loss_4_accum = loss / self.config.accumulation_steps
                loss_4_accum.backward()
                self.running_loss += loss.item()
                self.recent_loss += loss.item()

                if self.steps % self.config.accumulation_steps == 0:
                    if self.config.accumulation_steps > 1:
                        self.logger.info(f"step #{self.steps}, loss = {loss}")
                    elif self.steps % 10 == 0:
                        self.logger.info(f"step #{self.steps}, loss = {loss}")
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.steps % (self.config.evaluate_every_n_steps // self.device_count) == 0:
                    self.evaluate_model()
                    # print(self.test_losses)

                if self.consecutive_losses_increasing == self.config.consecutive_losses_to_stop:
                    self.finished_training = True
                    self.logger.info(f'best_loss: {min(self.test_losses)}')
                    dt = datetime.datetime.fromtimestamp(time.time())
                    self.logger.info(f'run_date: {dt.year*1000+dt.month*100+dt.day}')
                    self.logger.info(f'run_end_datetime: {time.time()}')
                    break
            self.epoch_steps = 0
            if self.finished_training:
                break

    def evaluate_model(self):
        test_loss = 0
        start_time = time.time()
        self.model.eval()
        with torch.no_grad():
            self.test_steps_running = 0
            detailed_losses_temp = {
                i: 0.0 for i in range(self.config.n_targets)}
            combined_features = []
            combined_targets = []
            combined_outputs = []
            
            for inputs, labels in self.test_loader:
                self.test_steps += 1
                self.test_steps_running += 1
                model_output = self.model.forward(inputs)
                if len(model_output.shape) == 3:
                    model_output = torch.squeeze(model_output[:, -1, :], 1)
                if self.test_steps_running == 1 and len(self.test_losses) == 0:
                    print("(Eval) Batch output shape is: ", model_output.size())
                model_output = model_output.view(-1, self.config.n_targets)
                labels = labels.view(-1, self.config.n_targets)
                for i, (output, label) in enumerate(zip(torch.transpose(model_output, 0, 1), torch.transpose(labels, 0, 1))):
                    detailed_losses_temp[i] += self.config.loss_function(output, label).item() / len(self.test_loader)
                batch_loss = self.config.loss_function(model_output, labels)
                test_loss += batch_loss.item()
                combined_features.append(inputs)
                combined_targets.append(labels)
                combined_outputs.append(model_output)

            self.test_losses.append(test_loss / len(self.test_loader))
            # record losses for each target
            for key in self.detailed_losses:
                self.detailed_losses[key].append(detailed_losses_temp[key])
            
            self.logger.info(f'n_evals: {len(self.test_losses)}')
            self.logger.info(f'test_loss_at_eval: {test_loss / len(self.test_loader)}')
            self.logger.info(f'train_running_loss_at_eval: {self.running_loss / self.epoch_steps}')
            self.logger.info(f'train_recent_loss_at_eval: {self.recent_loss / self.recent_steps}')

            if self.test_losses[-1] > min(self.test_losses):
                self.consecutive_losses_increasing += 1
            else:
                self.consecutive_losses_increasing = 0

            # DataParallel messes with the state dict - cleaner to save out of module
            if self.data_parallel:
                torch.save(self.model.module.state_dict(), self.log_folder / "model.pth")
            else:
                torch.save(self.model.state_dict(), self.log_folder / "model.pth")
            
            self.logger.info(f'n_epoch: {self.epoch}')
            self.logger.info(f'train_loss_epoch: {self.running_loss / self.epoch_steps}')
            self.logger.info(f'best_eval_loss_epoch: {min(self.test_losses)}')
            self.logger.info(f'consecutive_losses_epoch: {self.consecutive_losses_increasing}')
            self.logger.info(f"Time to eval--- {time.time() - start_time} seconds ---")
            self.logger.info(f"Epoch {self.epoch :.0f}.. ")
            self.logger.info(f"Step: {self.steps :.0f}.. ")
            self.logger.info(f"Train loss (running): {self.running_loss / self.epoch_steps :.6f}.. ")
            self.logger.info(f"Train loss (recent): {self.recent_loss / self.recent_steps :.6f}.. ")
            self.logger.info(f"Test loss: {test_loss / len(self.test_loader) :.6f}.. ")
            self.logger.info(f"Consecutive losses: {self.consecutive_losses_increasing}")
            
            self.train_losses.append(
                self.running_loss / len(self.train_loader))
            self.recent_steps = 0
            self.recent_loss = 0
            # Set model back to train
            self.model.train()

    def score_model(self, score_data_loader):
        self.model.eval()
        option_dataset = np.zeros((0, self.config.tabular_data_size))
        res = np.zeros((0, self.config.n_targets * 2))
        with torch.no_grad():
            for inputs, labels in score_data_loader:
                option_dataset = np.concatenate((option_dataset, inputs[0].numpy()), axis=0)
                model_output = self.model.forward(inputs)
                if len(model_output.shape) == 3:
                    model_output = torch.squeeze(model_output[:, -1, :], 1)
                model_output = model_output.view(-1, self.config.n_targets)
                labels = labels.view(-1, self.config.n_targets)
                # concatenate targets and labels
                cat_res = torch.cat([model_output, labels], dim=1).cpu().detach().numpy()
                # concat results batch to np dataframe
                res = np.concatenate((res, cat_res), axis=0)
                # and append security IDs and datetimes for reference, then save the whole DF down
        res_col_names = ['target_' + str(n) + '_pred' for n in range(0, self.config.n_targets)] + \
            ['target_' + str(n) + '_actual' for n in range(0, self.config.n_targets)]
        res_df = pd.concat(
            [
                pd.DataFrame(option_dataset, columns=['ex_days', 'strike_price', 'cp_flag', 'volume', 'open_interest']),
                pd.DataFrame(res, columns=res_col_names)
            ], 
            axis=1
        )
        res_df.to_csv(self.log_folder / "scored_results.csv", index=False)
        
        mse_avg = 0.0
        mae_avg = 0.0
        for i in range(0, self.config.n_targets):
            mse = ((res[:, i] - res[:, i+self.config.n_targets])**2).mean()
            mse_avg += mse / self.config.n_targets
            mae = abs(res[:, i] - res[:, i+self.config.n_targets]).mean()
            mae_avg += mae / self.config.n_targets
            self.logger.info(f'mse_target_{i}: {mse}')
            self.logger.info(f'mae_target_{i}: {mae}')

        self.logger.info(f'mse_avg: {mse_avg}')
        self.logger.info(f'mae_avg: {mae_avg}')
        
        return res_df

    def cleanup_model(self):
        self.model.destroy()
    
    @staticmethod
    def create_logger(
        log_file, 
        console_logging_format="%(levelname)s: %(asctime)s: %(message)s", 
        file_logging_format="%(levelname)s: %(asctime)s: %(message)s",
        console_logging_level=logging.INFO, 
        file_logging_level=logging.DEBUG,
    ):
        """[Create a log file to record the experiment's logs]
        
        Arguments:
            path {string} -- path to the directory
            file {string} -- file name
        
        Returns:
            [obj] -- [logger that record logs]
        """

        # check if the file exist
        if not os.path.isfile(log_file):
            open(log_file, "w+").close()

        # configure logger
        logging.basicConfig(level=console_logging_level, format=console_logging_format)
        logger = logging.getLogger()
        
        # create a file handler for output file
        handler = logging.FileHandler(log_file)

        # set the logging level for log file
        handler.setLevel(file_logging_level)
        
        # create a logging format
        formatter = logging.Formatter(file_logging_format)
        handler.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(handler)

        return logger
