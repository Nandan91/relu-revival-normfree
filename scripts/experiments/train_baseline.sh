# Baseline with standard gelu activation
python run_clm.py num_token_mult=6 model.n_layer=12 model.norm_type=ln model.activation_function=gelu


# Baseline with standard relu activation
python run_clm.py num_token_mult=6 model.n_layer=12 model.norm_type=ln model.activation_function=relu


