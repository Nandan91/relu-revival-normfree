# Normalization-free with standard gelu activation
python run_clm.py num_token_mult=6 model.n_layer=12 model.norm_type=free model.activation_function=gelu


# Normalization-free with standard gelu activation
python run_clm.py num_token_mult=6 model.n_layer=12 model.norm_type=free model.activation_function=relu


# Normalization-free with leaky relu and fixed negative slope
python run_clm.py num_token_mult=6 model.n_layer=12 model.norm_type=free model.activation_function=leaky_relu model.lrelu_neg_slope=1e-1


# Normalization-free with layerwise learnable negative slope 
python run_clm.py num_token_mult=6 model.n_layer=12 model.norm_type=free model.activation_function=learnable_lrelu model.learnable_lrelu_mode=per_layer

# Normalization-free with a global learnable negative slope 
python run_clm.py num_token_mult=6 model.n_layer=12 model.norm_type=free model.activation_function=learnable_lrelu model.learnable_lrelu_mode=global
