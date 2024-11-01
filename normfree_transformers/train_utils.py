import torch
from transformers import Trainer
from transformers.trainer_pt_utils import get_parameter_names
import wandb
import os
from transformers import TrainerCallback
from .model_utils import myGPT2Attention, myGPT2MLP, MyConv1D


class EntropyLoggingCallback(TrainerCallback):
	# Callback to log entropy values per layer during training
	def __init__(self, output_dir, num_layers):
		os.makedirs(output_dir, exist_ok=True)
		self.entropy_files = [open(f"{output_dir}/layer_{i}_entropy.txt", "a") for i in range(num_layers)]
	
	def log_entropy(self, step, layer_index, entropy_value):
		self.entropy_files[layer_index].write(f"Step {step} - Layer {layer_index}: {entropy_value}\n")
		self.entropy_files[layer_index].flush()

	def on_train_end(self, args, state, control, **kwargs):
		for file in self.entropy_files:
			file.close()

class NaNCountLoggingCallback(TrainerCallback):
	# Callback to log NaN count per layer during training
	def __init__(self, output_dir, num_layers):
		os.makedirs(output_dir, exist_ok=True)
		self.nan_count_files = [open(f"{output_dir}/layer_{i}_nan_count.txt", "a") for i in range(num_layers)]
	   
	def log_nan_count(self, step, layer_index, avg_nan_count):
		self.nan_count_files[layer_index].write(f"Step {step} - Layer {layer_index}: {avg_nan_count}\n")
		self.nan_count_files[layer_index].flush()

	def on_train_end(self, args, state, control, **kwargs):
		for file in self.nan_count_files:
			file.close()

class SlopeLoggingCallback(TrainerCallback):
	# Callback to log activation function slopes (global or per-layer)
	def __init__(self, mode, output_dir, num_layers):
		os.makedirs(output_dir, exist_ok=True)
		self.mode = mode
		self.logged_steps = set()
		self.num_layers = num_layers
		
		# Initialize files based on logging mode
		if self.mode == 'global':
			self.slope_file = open(f"{output_dir}/global_slope.txt", "a")
		elif self.mode == 'per_layer':
			self.slope_files = [open(f"{output_dir}/layer_{i}_slope.txt", "a") for i in range(num_layers)]        
		else:
			raise ValueError("Unsupported mode for SlopeLoggingCallback")

	def log_slope(self, model, step):

		if step in self.logged_steps:
			return
		self.logged_steps.add(step)

		if self.mode == 'global':
			for block in model.transformer.h:
				if hasattr(block.mlp.act, 'slopes'):
					self.slope_file.write(f"Step {step}: {block.mlp.act.slopes.item()}\n")
					self.slope_file.flush()
					break  

		elif self.mode == 'per_layer':
			for i, block in enumerate(model.transformer.h):
				if hasattr(block.mlp.act, 'slopes'):
					self.slope_files[i].write(f"Step {step}: {block.mlp.act.slopes[i].item()}\n")
					self.slope_files[i].flush()

	def on_train_end(self, args, state, control, **kwargs):
		if self.mode == 'global':
			self.slope_file.close()
		elif self.mode == 'per_layer':
			for file in self.slope_files:
				file.close()

			
class MyTrainer(Trainer):
	def create_optimizer(self):
		"""
		Identical to standard HF AdamW optimizer, but with no WD for gain parameters.
		"""
		opt_model = self.model
		self.logged_steps = set()

		if self.optimizer is None:
			decay_parameters = get_parameter_names(
				opt_model, [torch.nn.LayerNorm]
			)
			decay_parameters = [name for name in decay_parameters if "bias" not in name]

			gain_parameters = [name for name in decay_parameters if "gain" in name]

			optimizer_grouped_parameters = [
				{
					"params": [
						p
						for n, p in opt_model.named_parameters()
						if (
							n in decay_parameters
							and n not in gain_parameters
							and p.requires_grad
						)
					],
					"weight_decay": self.args.weight_decay,
				},
				{
					"params": [
						p
						for n, p in opt_model.named_parameters()
						if (n in gain_parameters and p.requires_grad)
					],
					"weight_decay": 0.0,
				},
				{
					"params": [
						p
						for n, p in opt_model.named_parameters()
						if (n not in decay_parameters and p.requires_grad)
					],
					"weight_decay": 0.0,
				},
			]

			optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
				self.args
			)
			self.optimizer = optimizer_cls(
				optimizer_grouped_parameters, **optimizer_kwargs
			)

		return self.optimizer

	def compute_loss(self, model, inputs, return_outputs=False):
		"""
		Identical to HF transformers compute_loss, but with extra logging.
		"""

		outputs = model(**inputs)
		
		if self.args.past_index >= 0:
			self._past = outputs[self.args.past_index]

		if isinstance(outputs, dict) and "loss" not in outputs:
			raise ValueError(
				"The model did not return a loss from the inputs, only the following keys: "
				f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
			)
		
		loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

		# Log metrics every 100 steps and at step 0, avoiding duplicate logging
		if (self.state.global_step % 100 == 0 or self.state.global_step == 0) and self.state.global_step not in self.logged_steps:

			# Logging NaN counts for each layer
			for i, attn_mat in enumerate(outputs["attentions"]):
				nan_counts = []
				# Check each attention head separately
				for head_index in range(attn_mat.shape[1]):
					flattened_attn_mat = attn_mat[:, head_index, :, :].reshape(-1, attn_mat.shape[-1])
					nan_count = torch.isnan(flattened_attn_mat).sum().item()
					nan_counts.append(nan_count)
				# Log average NaN count across all heads for this layer
				avg_nan_count = sum(nan_counts) / len(nan_counts)
				self.nan_count_logging_callback.log_nan_count(self.state.global_step, i, avg_nan_count)

			# Logging Shannon's entropy for attention mechanism in each layer
			for i, attn_mat in enumerate(outputs["attentions"]):
				# Compute attention entropy: -sum(p * log(p))
				ent = -torch.nansum(attn_mat * torch.log(attn_mat), dim=-1)
				avg_ent = ent.mean().item()
				self.entropy_logging_callback.log_entropy(self.state.global_step, i, avg_ent)

			# Log negative slopes for learnable leaky relu activation functions
			for i, block in enumerate(self.model.transformer.h):
				if hasattr(block.mlp.act, 'slopes'):
					if self.neg_slope_logging_callback.mode == 'global':
						# For global mode, log once and break
						self.neg_slope_logging_callback.log_slope(self.model, self.state.global_step)
						break  
					elif self.neg_slope_logging_callback.mode == 'per_layer':
						# For per-layer mode, log for each layer
						self.neg_slope_logging_callback.log_slope(self.model, self.state.global_step)
			
			# Mark this step as logged to avoid duplicate logging			
			self.logged_steps.add(self.state.global_step)

		with torch.no_grad():
			if self.state.global_step % 100 == 0 and "wandb" in self.args.report_to:
				to_report = {}
				# Logging Shannon's entropy for each attention heads to create heatmap of trained network  
				if self.args.report_attn_entropy:
					for i, attn_mat in enumerate(outputs["attentions"]):
						ent = -torch.nansum(attn_mat * torch.log(attn_mat), dim=-1)
						to_report[f"attn.{i}.entropy"] = ent.mean()

				
				wandb.log(to_report, step=self.state.global_step, commit=False)

		return (loss, outputs) if return_outputs else loss
