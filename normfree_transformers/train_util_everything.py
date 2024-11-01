import torch
import torch.nn as nn
import random
import os
import logging
from transformers import Trainer
from transformers import TrainerCallback
from transformers.trainer_pt_utils import get_parameter_names
import wandb

# from ent_baseline_gelu import baseline_entropy
from ent_baseline_relu import baseline_entropy
# from ent_baseline_no_acts import baseline_entropy
# from ent_no_ln_gelu import baseline_entropy
# from ent_no_ln_relu import baseline_entropy

from .model_utils import myGPT2Attention, myGPT2MLP, MyConv1D, RMSNorm, CustomLeakyReLU
# from .model_utils_baseline import myGPT2AttentionB, myGPT2MLPB, MyConv1DB,  


class EntropyLoggingCallback(TrainerCallback):
	def __init__(self, output_dir, num_layers):
		os.makedirs(output_dir, exist_ok=True)
		self.entropy_files = [open(f"{output_dir}/layer_{i}_entropy.txt", "a") for i in range(num_layers)]
		

	def log_entropy(self, step, layer_index, entropy_value):
		self.entropy_files[layer_index].write(f"Step {step} - Layer {layer_index}: {entropy_value}\n")
		self.entropy_files[layer_index].flush()

	def on_train_end(self, args, state, control, **kwargs):
		for file in self.entropy_files:
			file.close()

class RankLoggingCallback(TrainerCallback):
	def __init__(self, output_dir, num_layers):
		os.makedirs(output_dir, exist_ok=True)
		self.rank_files = [open(f"{output_dir}/layer_{i}_rank.txt", "a") for i in range(num_layers)]
		

	def log_rank(self, step, layer_index, avg_rank_value):
		self.rank_files[layer_index].write(f"Step {step} - Layer {layer_index}: {avg_rank_value}\n")
		self.rank_files[layer_index].flush()

	def on_train_end(self, args, state, control, **kwargs):
		for file in self.rank_files:
			file.close()

# class TemperatureLoggingCallback(TrainerCallback):
#     def __init__(self, output_dir, num_layers):
#         os.makedirs(output_dir, exist_ok=True)
#         self.temperature_files = [open(f"{output_dir}/layer_{i}_temperature.txt", "a") for i in range(num_layers)]
#         #self.temperature_accumulator = temperature_accumulator

#     def log_temperature(self, step, layer_index, avg_temp_value):
#         self.temperature_files[layer_index].write(f"Step {step} - Layer {layer_index}: {avg_temp_value}\n")
#         self.temperature_files[layer_index].flush()

#     def on_train_end(self, args, state, control, **kwargs):
#         for file in self.temperature_files:
#             file.close()

class NaNCountLoggingCallback(TrainerCallback):
	def __init__(self, output_dir, num_layers):
		os.makedirs(output_dir, exist_ok=True)
		self.nan_count_files = [open(f"{output_dir}/layer_{i}_nan_count.txt", "a") for i in range(num_layers)]
		# self.nan_count_accumulator = nan_count_accumulator

	def log_nan_count(self, step, layer_index, avg_nan_count):
		self.nan_count_files[layer_index].write(f"Step {step} - Layer {layer_index}: {avg_nan_count}\n")
		self.nan_count_files[layer_index].flush()

	def on_train_end(self, args, state, control, **kwargs):
		for file in self.nan_count_files:
			file.close()

class LossLoggingCallback(TrainerCallback):
	def __init__(self, output_dir):
		os.makedirs(output_dir, exist_ok=True)
		self.train_ce_loss_file = open(f"{output_dir}/train_ce_loss.txt", "a")
		self.train_total_loss_file = open(f"{output_dir}/train_total_loss.txt", "a")
		self.eval_ce_loss_file = open(f"{output_dir}/eval_ce_loss.txt", "a")
		self.eval_total_loss_file = open(f"{output_dir}/eval_total_loss.txt", "a")

	def log_loss(self, step, ce_loss, total_loss, phase='train'):
		if phase == 'train':
			self.train_ce_loss_file.write(f"Step {step} -  {ce_loss}\n")
			self.train_total_loss_file.write(f"Step {step} -  {total_loss}\n")
		elif phase == 'eval':
			self.eval_ce_loss_file.write(f"Step {step} - {ce_loss}\n")
			self.eval_total_loss_file.write(f"Step {step} -  {total_loss}\n")
		self.train_ce_loss_file.flush()
		self.train_total_loss_file.flush()
		self.eval_ce_loss_file.flush()
		self.eval_total_loss_file.flush()

	def on_train_end(self, args, state, control, **kwargs):
		self.train_ce_loss_file.close()
		self.train_total_loss_file.close()
		self.eval_ce_loss_file.close()
		self.eval_total_loss_file.close()


class SlopeLoggingCallback(TrainerCallback):
	def __init__(self, output_dir, num_layers, mode='global'):
		os.makedirs(output_dir, exist_ok=True)
		self.mode = mode
		self.logged_steps = set()
		self.num_layers = num_layers
		
		if self.mode == 'global':
			self.slope_file = open(f"{output_dir}/global_slope.txt", "a")
		elif self.mode == 'per_layer':
			self.slope_files = [open(f"{output_dir}/layer_{i}_slope.txt", "a") for i in range(num_layers)]
		elif self.mode == 'per_feature_per_layer':
			self.slope_files_mean = [open(f"{output_dir}/layer_{i}_slope_mean.txt", "a") for i in range(num_layers)]
			self.slope_files_std = [open(f"{output_dir}/layer_{i}_slope_std.txt", "a") for i in range(num_layers)]
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

		if self.mode == 'per_layer':
			for i, block in enumerate(model.transformer.h):
				if hasattr(block.mlp.act, 'slopes'):
					self.slope_files[i].write(f"Step {step}: {block.mlp.act.slopes[i].item()}\n")
					self.slope_files[i].flush()

		elif self.mode == 'per_feature_per_layer':
			for i, block in enumerate(model.transformer.h):
				if hasattr(block.mlp.act, 'slopes'):
					mean_slope = block.mlp.act.slopes[i].mean().item()
					std_slope = block.mlp.act.slopes[i].std().item()
					self.slope_files_mean[i].write(f"Step {step}: {mean_slope}\n")
					self.slope_files_std[i].write(f"Step {step}: {std_slope}\n")
					self.slope_files_mean[i].flush()
					self.slope_files_std[i].flush()

	def on_train_end(self, args, state, control, **kwargs):
		if self.mode == 'global':
			self.slope_file.close()
		elif self.mode in ['per_layer', 'per_feature_per_layer']:
			for file in self.slope_files:
				file.close()
			if self.mode == 'per_feature_per_layer':
				for file in self.slope_files_std:
					file.close()
   
class MyTrainer(Trainer):

	def __init__(self, *args, **kwargs):

		super().__init__(*args, **kwargs)
		self.baseline_entropy = baseline_entropy
		self.logged_steps = set()
				
	def create_optimizer(self):
		"""
		Identical to standard HF AdamW optimizer, but with no WD for gain parameters.
		"""
		opt_model = self.model

		if self.optimizer is None:
			decay_parameters = get_parameter_names(
				opt_model, [torch.nn.LayerNorm, RMSNorm]
			)
			decay_parameters = [name for name in decay_parameters if "bias" not in name]

			gain_parameters = [name for name in decay_parameters if "gain" in name]

			temperature_parameters = [name for name in decay_parameters if "temperature" in name]
			entropy_reg_parameters = [name for name in decay_parameters if "entropy" in name]
			reg_threshold_parameters = [name for name in decay_parameters if "threshold" in name]

			optimizer_grouped_parameters = [
				{
					"params": [
						p
						for n, p in opt_model.named_parameters()
						if (
							n in decay_parameters
							and n not in gain_parameters
							and n not in temperature_parameters
							and n not in entropy_reg_parameters
							and n not in reg_threshold_parameters
							and p.requires_grad
						)
					],
					"weight_decay": self.args.weight_decay,
				},
				{
					"params": [p for n, p in opt_model.named_parameters() if (n in gain_parameters and p.requires_grad) ],
					"weight_decay": 0.0,
				},
				{
					"params": [ p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad) ],
					"weight_decay": 0.0,
				},
				
				{   "params": [p for n, p in opt_model.named_parameters() if n in temperature_parameters and p.requires_grad],
            	    "weight_decay": 0.0, 
        		},

        		{  "params": [p for n, p in opt_model.named_parameters() if n in entropy_reg_parameters and p.requires_grad],
            	   "weight_decay": 0.0, 
            	},

        		{  "params": [p for n, p in opt_model.named_parameters() if n in reg_threshold_parameters and p.requires_grad],
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
		num_layers = model.config.n_layer
		num_heads = model.config.n_head

						
		outputs = model(**inputs)
		
		if self.args.past_index >= 0:
			self._past = outputs[self.args.past_index]

		if isinstance(outputs, dict) and "loss" not in outputs:
			raise ValueError(
				"The model did not return a loss from the inputs, only the following keys: "
				f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
			)
		# We don't use .loss here since the model may return tuples instead of ModelOutput.
		ce_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
		device = outputs["loss"].device
		
		total_steps_per_epoch = len(self.get_train_dataloader())
						
		max_entropy = torch.log(torch.tensor(128.0))

		tolerance_margin = 0.30 * max_entropy 

		entropy_reg_loss = 0

		# for layer_idx, (block, attn_mat) in enumerate(zip(model.transformer.h, outputs["attentions"])):

		# 	ent_val = -torch.nansum(attn_mat * torch.log(attn_mat + 1e-9), dim=-1)

		# 	for head_idx in range(block.attn.num_heads):
		# 		head_entropy = ent_val[:, head_idx, :]
		# 		baseline_head_entropy = self.baseline_entropy[str(layer_idx)][str(head_idx)]

		# 		deviation = torch.abs(head_entropy - baseline_head_entropy)
		# 		squared_deviation_penalty = torch.square(deviation)
		# 		penalty = torch.where(deviation > 0, squared_deviation_penalty, torch.zeros_like(deviation))
		# 		entropy_reg_loss += penalty.sum()

		# entropy_reg_loss /= (num_layers * num_heads)
		# total_loss = ce_loss + 5e-5 * entropy_reg_loss

		# if self.state.global_step > 0.10 * total_steps_per_epoch:
			
		# 	total_loss = ce_loss + 5e-6 * entropy_reg_loss
		# 	# print ("Loss:", ce_loss.item())
		# 	# print ("EntLoss:", entropy_reg_loss.item())
		# 	# print ("TotalLoss:", total_loss.item())
		# else:
		# 	# print ("Hello Hi!")
		# 	total_loss = ce_loss


		# penalized_layers = [1]

		# for layer_idx, (block, attn_mat) in enumerate(zip(model.transformer.h, outputs["attentions"])):
		# 	ent_val = -torch.nansum(attn_mat * torch.log(attn_mat + 1e-9), dim=-1)

		# 	if layer_idx in penalized_layers:  # Only apply penalty for specific layers
		# 		for head_idx in range(block.attn.num_heads):
		# 			head_entropy = ent_val[:, head_idx, :]
		# 			baseline_head_entropy = self.baseline_entropy[str(layer_idx)][str(head_idx)]

		# 			deviation = torch.abs(head_entropy - baseline_head_entropy)
		# 			squared_deviation_penalty = torch.square(deviation)
		# 			penalty = torch.where(deviation > 0, squared_deviation_penalty, torch.zeros_like(deviation))
		# 			entropy_reg_loss += penalty.sum()

		# entropy_reg_loss /= (len(penalized_layers) * num_heads)
		# total_loss = ce_loss + 5e-4 * entropy_reg_loss  

		# if 0.10 * total_steps_per_epoch <= self.state.global_step < 0.80 * total_steps_per_epoch:

		# 	total_loss = loss + 1e-6 * entropy_reg_loss
		# else:
		# 	total_loss = ce_loss
		
		# if self.state.global_step < 0.05 * total_steps_per_epoch:
		# if 0.30 * total_steps_per_epoch <= self.state.global_step < 0.80 * total_steps_per_epoch:
			
		
		# for layer_idx, (block, attn_mat) in enumerate(zip(model.transformer.h, outputs.attentions)):
		# 	ent_val = -torch.nansum(attn_mat * torch.log(attn_mat + 1e-9), dim=-1).to(device)
		# 	layer_entropy_reg_loss = 0
					
		# 	for head_idx in range(block.attn.num_heads):
		# 		head_entropy = ent_val[:, head_idx, :]
		# 		threshold = block.attn.reg_threshold_weights[head_idx] * max_entropy
		# 		deviation = head_entropy - threshold
				

		# 		# Penalize deviations (square the deviation)
		# 		penalty = torch.square(torch.where(torch.abs(deviation) > tolerance_margin, deviation,torch.zeros_like(deviation)))
		# 		layer_entropy_reg_loss += penalty.sum()

		# 	layer_entropy_reg_loss /= num_heads
		# 	entropy_reg_loss += layer_entropy_reg_loss

		# entropy_reg_loss /= num_layers

		# total_loss = ce_loss + 1e-5 * entropy_reg_loss

		# if self.state.global_step > 0.05 * total_steps_per_epoch:
		# 	total_loss = ce_loss + 1e-5 * entropy_reg_loss
		# else:
		# 	total_loss = ce_loss
		
		# if (self.state.global_step % 100 == 0 or self.state.global_step == 0) and self.state.global_step not in self.logged_steps:
		# 	phase = 'train' if self.model.training else 'eval'
		# 	self.loss_logging_callback.log_loss(self.state.global_step, ce_loss.item(), total_loss.item(), phase)
			
		# 	self.logged_steps.add(self.state.global_step)

		if (self.state.global_step % 100 == 0 or self.state.global_step == 0) and self.state.global_step not in self.logged_steps:
			
			# Rank Logging
			# for i, attn_mat in enumerate(outputs["attentions"]):
			# 	layer_ranks = []
			# 	for head_index in range(attn_mat.shape[1]):
			# 		flattened_attn_mat = attn_mat[:, head_index, :, :].reshape(-1, attn_mat.shape[-1])
			# 		flattened_attn_mat = torch.nan_to_num(flattened_attn_mat, nan=1e-5)
			# 		_, s, _ = torch.linalg.svd(flattened_attn_mat, full_matrices=False)
			# 		spectral_gaps = torch.diff(s, n=1)
			# 		largest_gap, effective_rank_index = torch.max(spectral_gaps, dim=0)
			# 		effective_rank = effective_rank_index.item() + 1
			# 		layer_ranks.append(effective_rank)
			# 	avg_rank = sum(layer_ranks) / len(layer_ranks)
			# 	self.rank_logging_callback.log_rank(self.state.global_step, i, avg_rank)

			# NaN Count Logging
			# for i, attn_mat in enumerate(outputs["attentions"]):
			# 	nan_counts = []
			# 	for head_index in range(attn_mat.shape[1]):
			# 		flattened_attn_mat = attn_mat[:, head_index, :, :].reshape(-1, attn_mat.shape[-1])
			# 		nan_count = torch.isnan(flattened_attn_mat).sum().item()
			# 		nan_counts.append(nan_count)
			# 	avg_nan_count = sum(nan_counts) / len(nan_counts)
			# 	self.nan_count_logging_callback.log_nan_count(self.state.global_step, i, avg_nan_count)

			# Entropy Logging
			# for i, attn_mat in enumerate(outputs["attentions"]):
			# 	ent = -torch.nansum(attn_mat * torch.log(attn_mat), dim=-1)
			# 	avg_ent = ent.mean().item()
			# 	self.entropy_logging_callback.log_entropy(self.state.global_step, i, avg_ent)

			# Neg slope logging
			# for i, block in enumerate(self.model.transformer.h):
			# 	if hasattr(block.mlp.act, 'slopes'):
			# 		if self.neg_slope_logging_callback.mode == 'global':
			# 			self.neg_slope_logging_callback.log_slope(self.model, self.state.global_step)
			# 			break  
			# 		elif self.neg_slope_logging_callback.mode in ['per_layer', 'per_feature_per_layer']:
			# 			self.neg_slope_logging_callback.log_slope(self.model, self.state.global_step)

			# Loss logging
			# phase = 'train' if self.model.training else 'eval'
			# self.loss_logging_callback.log_loss(self.state.global_step, ce_loss.item(), total_loss.item(), phase)
			
			self.logged_steps.add(self.state.global_step)
			

		if self.state.global_step % 200 == 0 and "wandb" in self.args.report_to:
			if self.args.report_gains:
				to_report = {}
				for i, block in enumerate(model.transformer.h):
					if type(block.mlp) is myGPT2MLP:
						to_report[
							f"{i}.mlp_block_resid_gain"
						] = block.mlp_block_resid_gain.data.norm()
					# if type(block.attn.v_attn) is MyConv1D:
					#     to_report[
					#         f"attn.{i}.value_skip_gain"
					#     ] = block.attn.v_attn.skip_gain.data
					#     to_report[
					#         f"attn.{i}.value_resid_gain"
					#     ] = block.attn.v_attn.resid_gain.data
					# if type(block.attn.c_proj) is MyConv1D:
					#     to_report[
					#         f"attn.{i}.proj_skip_gain"
					#     ] = block.attn.c_proj.skip_gain.data
					#     to_report[
					#         f"attn.{i}.proj_resid_gain"
					#     ] = block.attn.c_proj.resid_gain.data

					if hasattr(block, 'attn_block_resid_gain'):
						to_report[f"{i}.attn_block_resid_gain"] = block.attn_block_resid_gain.data

					if type(block.attn) is myGPT2Attention:
						to_report[
							f"attn.{i}.attn_mat_skip_gain_mean"
						] = block.attn.attn_mat_skip_gain.data.mean()
						to_report[
							f"attn.{i}.attn_mat_resid_gain_mean"
						] = block.attn.attn_mat_resid_gain.data.mean()
						to_report[
							f"attn.{i}.centre_attn_gain_mean"
						] = block.attn.centre_attn_gain.data.mean()
						to_report[
							f"attn.{i}.attn_mat_skip_gain_std"
						] = block.attn.attn_mat_skip_gain.data.std()
						to_report[
							f"attn.{i}.attn_mat_resid_gain_std"
						] = block.attn.attn_mat_resid_gain.data.std()
						to_report[
							f"attn.{i}.centre_attn_gain_std"
						] = block.attn.centre_attn_gain.data.std()
				wandb.log(to_report)

			# if self.args.report_neg_slope:
			# 	to_report = {}
			# 	for i, block in enumerate(model.transformer.h):
			# 		if type(block.mlp) is myGPT2MLP:
			# 			if isinstance(block.mlp.act, CustomLeakyReLU):
			# 				mode = block.mlp.act.mode
			# 				slopes = block.mlp.act.slopes.data
				
			# 				if mode == 'global':
			# 					to_report[f"global_slope"] = slopes.item()
			# 				elif mode == 'per_layer':
			# 					to_report[f"{i}_layer_slope"] = slopes[i].item()
			# 				elif mode == 'per_feature_per_layer':
			# 					to_report[f"{i}_layer_slope_mean"] = slopes[i].mean().item()
			# 					to_report[f"{i}_layer_slope_std"] = slopes[i].std().item()
			# wandb.log(to_report)    
   
			
			# if self.args.report_neg_slope:
			#   to_report = {}
			#   for i, block in enumerate(model.transformer.h):
			#       if type(block.mlp) is myGPT2MLP:
			#           if isinstance(block.mlp.act, CustomLeakyReLU):
			#               mode = block.mlp.act.mode
							
			#               # Log Positive Slopes if present
			#               if block.mlp.act.pos_slopes is not None:
			#                   pos_slopes = block.mlp.act.pos_slopes.data
			#                   if mode == 'global':
			#                       to_report[f"global_pos_slope"] = pos_slopes.item()
			#                   elif mode == 'per_layer':
			#                       to_report[f"{i}_layer_pos_slope"] = pos_slopes[block.mlp.layer_idx].item()
			#                   elif mode == 'per_feature_per_layer':
			#                       to_report[f"{i}_layer_pos_slope_mean"] = pos_slopes[block.mlp.layer_idx].mean().item()
			#                       to_report[f"{i}_layer_pos_slope_std"] = pos_slopes[block.mlp.layer_idx].std().item()
							
			#               # Log Negative Slopes if present
			#               if block.mlp.act.neg_slopes is not None:
			#                   neg_slopes = block.mlp.act.neg_slopes.data
			#                   if mode == 'global':
			#                       to_report[f"global_neg_slope"] = neg_slopes.item()
			#                   elif mode == 'per_layer':
			#                       to_report[f"{i}_layer_neg_slope"] = neg_slopes[block.mlp.layer_idx].item()
			#                   elif mode == 'per_feature_per_layer':
			#                       to_report[f"{i}_layer_neg_slope_mean"] = neg_slopes[block.mlp.layer_idx].mean().item()
			#                       to_report[f"{i}_layer_neg_slope_std"] = neg_slopes[block.mlp.layer_idx].std().item()
							
			#               # Log Shifts if present
			#               if block.mlp.act.shifts is not None:
			#                   shifts = block.mlp.act.shifts.data
			#                   if mode == 'global':
			#                       to_report[f"global_shift"] = shifts.item()
			#                   elif mode == 'per_layer':
			#                       to_report[f"{i}_layer_shift"] = shifts[block.mlp.layer_idx].item()
			#                   elif mode == 'per_feature_per_layer':
			#                       to_report[f"{i}_layer_shift_mean"] = shifts[block.mlp.layer_idx].mean().item()
			#                       to_report[f"{i}_layer_shift_std"] = shifts[block.mlp.layer_idx].std().item()

			# wandb.log(to_report)


			# if self.args.report_ent_reg_weights:
			# 	to_report = {}
			# 	for layer_idx, block in enumerate(model.transformer.h):
			# 		attn = block.attn
			# 		for head_idx in range(attn.num_heads):
			# 			weight = attn.entropy_reg_weights[head_idx].item()
			# 			to_report[f"attn.{layer_idx}.head.{head_idx}.ent_reg_weights"] = weight

			# wandb.log(to_report)


			# if self.args.report_reg_threshold_weights:
			# 	to_report = {}
			# 	for layer_idx, block in enumerate(model.transformer.h):
			# 		attn = block.attn
			# 		for head_idx in range(attn.num_heads):
			# 			weight = attn.reg_threshold_weights[head_idx].item()
			# 			to_report[f"attn.{layer_idx}.head.{head_idx}.reg_th_weights"] = weight

			# wandb.log(to_report)

			
			if self.args.report_attn_entropy:
				to_report = {}
				for i, attn_mat in enumerate(outputs["attentions"]):
					ent = -torch.nansum(attn_mat * torch.log(attn_mat), dim=-1)
					
					for head_index in range(ent.shape[1]):  # assuming ent shape is [batch_size, num_heads, sequence_length]
						head_entropy = ent[:, head_index, :].mean()  # mean over the sequence_length
						to_report[f"attn.{i}.head.{head_index}.entropy"] = head_entropy

					# to_report[f"attn.{i}.entropy"] = ent.mean()
					
				wandb.log(to_report)

			# if self.args.report_softmax_temperature:
			#   to_report = {}
			#   for i, block in enumerate(model.transformer.h):
			#       if type(block.attn) is myGPT2Attention:
			#           for j in range(block.attn.num_heads):
			#               to_report[f"attn.{i}.head.{j}.temp"] = block.attn.temp_factors[i, j].item()
						
			#   wandb.log(to_report)


			# if self.args.report_nan_counts:
			# 	to_report = {}
			# 	for i, attn_mat in enumerate(outputs["attentions"]):
			# 		for head_index in range(ent.shape[1]):  
			# 			flattened_attn_mat = attn_mat[:, head_index, :, :].reshape(-1, attn_mat.shape[-1])
			# 			nan_count = torch.isnan(flattened_attn_mat).sum()
			# 			to_report[f"attn.{i}.head.{head_index}.NaNs"] = nan_count.item()
					
			# 	wandb.log(to_report)

			
			# if self.args.report_attention_rank:
			#   to_report = {}
			#   for i, attn_mat in enumerate(outputs["attentions"]):
			# 	  for head_index in range(ent.shape[1]):  
			# 		  flattened_attn_mat = attn_mat[:, head_index, :, :].reshape(-1, attn_mat.shape[-1])
			# 		  flattened_attn_mat = torch.nan_to_num(flattened_attn_mat, nan=1e-5)
			# 		  _, s, _ = torch.linalg.svd(flattened_attn_mat, full_matrices=False)

			# 		  # relative_threshold = 0.01 * torch.max(s)  #  1% of the largest singular value
			# 		  # rank = torch.sum(s > relative_threshold)
			# 		  # to_report[f"attn.{i}.head.{head_index}.rank"] = rank

			# 		  spectral_gaps = torch.diff(s, n=1)

			# 		  # Find the largest spectral gap and its index (effective rank)
			# 		  largest_gap, effective_rank_index = torch.max(spectral_gaps, dim=0)
			# 		  effective_rank = effective_rank_index + 1  # Adding 1 because index is zero-based
			# 		  to_report[f"attn.{i}.head.{head_index}.rank"] = effective_rank
									
			#   wandb.log(to_report)
				

		return (ce_loss, outputs) if return_outputs else ce_loss

