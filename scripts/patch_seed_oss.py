import vllm.model_executor.models.seed_oss as seed_oss_module
from vllm.model_executor.layers.linear import RowParallelLinear

_original_init = seed_oss_module.SeedOSSForCausalLM.__init__


def _patched_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)

    # Get configuration parameters
    hidden_size = self.config.hidden_size
    total_num_heads = self.config.num_attention_heads
    head_dim = getattr(self.config, 'head_dim', hidden_size // total_num_heads)
    quant_config = kwargs.get('quant_config', None)

    attention_out_bias = getattr(self.config, 'attention_out_bias', False)

    for layer_idx, layer in enumerate(self.model.layers):
        layer.self_attn.o_proj = RowParallelLinear(
            total_num_heads * head_dim,
            hidden_size,
            bias=attention_out_bias,  # Read from config
            quant_config=quant_config,
            prefix=f"model.layers.{layer_idx}.self_attn.o_proj",
        )

seed_oss_module.SeedOSSForCausalLM.__init__ = _patched_init