import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from typing import Optional, Tuple, Union
from transformers.activations import ACT2FN
import math


def convertGPT2model(gpt2_model, new_cfg):
    """
    Convert the decoder blocks in a gpt2 model to customisable myGPT2BLock.
    """
    new_blocks = []
    for i, _ in enumerate(gpt2_model.transformer.h):
        new_block = myGPT2Block(new_cfg, layer_idx=i)       
        new_blocks.append(new_block)
    gpt2_model.transformer.h = nn.ModuleList(new_blocks)
    return gpt2_model


class myGPT2Block(nn.Module):
    """
    A customisable GPT2Block that implements baseline (Pre-LN) and normalization-free configurations. 
    """
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.layer_idx = layer_idx

        if config.norm_type == "ln":
            self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        elif config.norm_type == "free":
            self.ln_1 = nn.Identity()
            self.ln_2 = nn.Identity()
        else:
            raise NotImplementedError

        self.attn = myGPT2Attention(config, layer_idx=layer_idx)
        self.mlp = myGPT2MLP(inner_dim, config, layer_idx=layer_idx)
     

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],
    ]:
        
        skip_branch = hidden_states
        hidden_states = self.ln_1(hidden_states)
       
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        
        hidden_states =  attn_output
        hidden_states += skip_branch
        
        skip_branch = hidden_states
        hidden_states = self.ln_2(hidden_states)
        
        feed_forward_hidden_states = self.mlp(hidden_states)           
        hidden_states = feed_forward_hidden_states
        hidden_states += skip_branch

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs


class myGPT2Attention(nn.Module):
    """
    Attn sub-block.
    """
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        assert is_cross_attention == False
        max_positions = config.max_position_embeddings

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.layer_idx = layer_idx

        self.qk_attn = MyConv1D(2 * self.embed_dim, self.embed_dim)
                
        self.v_attn = MyConv1D(self.embed_dim, self.embed_dim, bias=False)
        self.c_proj = MyConv1D(self.embed_dim, self.embed_dim, bias=False)
        
        self.split_size = self.embed_dim
        query_weight, key_weight = self.qk_attn.weight.data.split(self.split_size, dim=1)
                
                
        uniform_causal_attn_mat = torch.ones(
            (max_positions, max_positions), dtype=torch.float32
        ) / torch.arange(1, max_positions + 1).view(-1, 1)
        self.register_buffer(
            "uniform_causal_attn_mat",
            torch.tril(
                uniform_causal_attn_mat,
            ).view(1, 1, max_positions, max_positions),
            persistent=False,
        )
        
        self.register_buffer(
            "diag",
            torch.eye(max_positions).view(1, 1, max_positions, max_positions),
            persistent=False,
        )
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.bool)
            ).view(1, 1, max_positions, max_positions),
            persistent=False,
        )

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [],
                value.size(-1) ** 0.5,
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )

                
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[
            :, :, key_length - query_length : key_length, :key_length
        ]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(
            attn_weights.device
        )
        attn_weights = torch.where(
            causal_mask, attn_weights.to(attn_weights.dtype), mask_value
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        new_attn_weights =  attn_weights.type(value.dtype)
        attn_output = torch.matmul(new_attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        assert encoder_hidden_states is None
        (query, key) = self.qk_attn(hidden_states).split(self.split_size, dim=2)
        value = self.v_attn(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(
            query, key, value, attention_mask, head_mask
        )

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        proj_output = self.c_proj(attn_output)
        
        outputs = (proj_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  

class myGPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config, layer_idx=None):
        """
        Initializes the myGPT2MLP module with customizable activation functions.
        
        Parameters:
        - intermediate_size: (int) Dimension of the intermediate layer.                            
        - layer_idx: (int, optional) Index of the current layer, used for per-layer mode. Defaults to None.
        """
        super().__init__()
        
        embed_dim = config.hidden_size  
        self.layer_idx = layer_idx  
        self.mode = config.learnable_lrelu_mode  # Set activation mode from config
        
        self.c_fc = MyConv1D(intermediate_size, embed_dim, bias=False)
        self.c_proj = MyConv1D(embed_dim, intermediate_size, bias=False)

        # Select activation function based on configuration
        if config.activation_function == "leaky_relu":
            self.act = LeakyReLU(negative_slope=config.lrelu_neg_slope)   
        elif config.activation_function == "learnable_lrelu":           
            self.act = LearnableLeakyReLU(config=config, 
            initial_slope=config.lrelu_neg_slope, 
            layer_idx=layer_idx)  # Custom LearnableLeakyReLU with learnable slope(s)
        else:          
            self.act = ACT2FN[config.activation_function]   
        
    def forward(self, hidden_states: Optional[torch.FloatTensor]) -> torch.FloatTensor:
       
        hidden_states = self.c_fc(hidden_states)  
        hidden_states = self.act(hidden_states)   
        hidden_states = self.c_proj(hidden_states)  
        
        return hidden_states

class MyConv1D(nn.Module):
    """
    (Linear) 1D-convolutional layer that can be reparameterised into skip (see Eq. 6 of paper).

    Args:
        nf (int): The number of output features.
        nx (int): The number of input features.
        bias (bool): Whether or not to use bias parameters.
    """
    def __init__(self, nf, nx, bias=True):
        super().__init__()
        self.nx = nx
        self.nf = nf

        if bias:
            self.bias = nn.Parameter(torch.zeros(nf))
        else:
            self.bias = nn.Parameter(torch.zeros(nf), requires_grad=False)

        self.weight = nn.Parameter(torch.zeros(nx, nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x

    def extra_repr(self):
        return f"in_dim={self.nx}, out_dim={self.nf}"  

class LeakyReLU(nn.Module):
    # LeakyReLU nonlinearity.
    __constants__ = ["inplace", "negative_slope"]
    inplace: bool
    negative_slope: float

    def __init__(self, negative_slope: float = 1e-2, inplace: bool = False) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.where(input >= 0.0, input, input * self.negative_slope)

    def extra_repr(self) -> str:
        inplace_str = ", inplace=True" if self.inplace else ""
        return "negative_slope={}{}".format(self.negative_slope, inplace_str)


class LearnableLeakyReLU(nn.Module):
    def __init__(self, config, initial_slope=0.01, layer_idx=None):
        """
        Initializes the Learnable Leaky ReLU activation function with learnable negative slope.

        Parameters:
        - initial_slope: (float, optional) Initial slope value for the negative part of Leaky ReLU. Defaults to 0.01.
        - layer_idx: (int, optional) Specifies the current layer index for per-layer mode. Defaults to None.
             
        """
        super().__init__()
        
        self.layer_idx = layer_idx       
        self.mode = config.learnable_lrelu_mode
        self.n_layers = config.n_layer

        # Initialize the slope parameter based on the mode
        if self.mode == 'global':            
            self.slopes = nn.Parameter(torch.tensor([initial_slope], dtype=torch.float32)) # Single learnable slope shared across all layers
        elif self.mode == 'per_layer':
            self.slopes = nn.Parameter(torch.full((self.n_layers,), initial_slope, dtype=torch.float32))  # Individual learnable slope for each layer
        else:
            raise ValueError("Invalid mode for LearnableLeakyReLU: must be 'global' or 'per_layer'")

    def forward(self, input):       
        if self.mode == 'global':
            slopes = self.slopes # Use a single slope for the entire model
        elif self.mode == 'per_layer':
            if self.layer_idx is None:
                raise ValueError("layer_idx must be specified in 'per_layer' mode")
            slopes = self.slopes[self.layer_idx].unsqueeze(0).unsqueeze(0).unsqueeze(-1) # Retrieve the slope for the current layer index
        else:
            raise ValueError("Unsupported mode")
        
        return torch.where(input >= 0, input, input * slopes)

    def extra_repr(self):
        return f"mode={self.mode}"  

