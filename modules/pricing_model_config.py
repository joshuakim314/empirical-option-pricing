class PricingModelConfig():
    def __init__(
        self,
        model_type,
        tcn_input_size,
        seq_len,
        tabular_data_size,
        n_targets,
        tcn_num_channels,
        tcn_kernel_size,
        tcn_dropout,
        linear_sizes,
        linear_dropout,
        batch_size,
        sample_size,
        optimizer,
        loss_function,
        learning_rate,
        l2_lambda,
        max_epochs,
        accumulation_steps,
        evaluate_every_n_steps,
        consecutive_losses_to_stop
    ):
        self.model_type = model_type
        self.tcn_input_size = tcn_input_size
        self.seq_len = seq_len
        self.tabular_data_size = tabular_data_size
        self.n_targets = n_targets
        self.tcn_num_channels = tcn_num_channels
        self.tcn_kernel_size = tcn_kernel_size
        self.tcn_dropout = tcn_dropout
        self.linear_sizes = linear_sizes
        self.linear_dropout = linear_dropout
        
        # training parameters
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.max_epochs = max_epochs
        self.accumulation_steps = accumulation_steps
        self.evaluate_every_n_steps = evaluate_every_n_steps
        self.consecutive_losses_to_stop = consecutive_losses_to_stop