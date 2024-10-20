import json
import argparse
import os
from tabulate import tabulate

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

class CausalLM:
  def __init__(self, config, param_bpe, kv_bpe, seq_len, batch):
    self.vocab_size = config['vocab_size']
    self.hidden_size = config['hidden_size']
    self.intermediate_size = config['intermediate_size']
    self.num_attention_heads = config['num_attention_heads']
    self.num_hidden_layers = config['num_hidden_layers']
    self.num_key_value_heads = config['num_key_value_heads']
    if config.get("head_dim") is not None:
      self.head_dim = config["head_dim"]
    else:
      self.head_dim = self.hidden_size // self.num_attention_heads

    self.param_bpe = param_bpe
    self.kv_bpe = kv_bpe
    self.seq_len = seq_len
    self.batch = batch
    
  def embedding_layer_size(self):
    return self.vocab_size * self.hidden_size * self.param_bpe

  def embedding_layer_nums(self):
    # decoder only
    return 1

  def attention_layer_size(self):
    qkv_size = self.hidden_size * self.head_dim * (self.num_attention_heads + self.num_key_value_heads * 2)
    output_size = self.hidden_size * self.hidden_size
    return (qkv_size + output_size) * self.param_bpe
  def attention_layer_nums(self):
    return self.num_hidden_layers

  def intermediate_layer_size(self):
    gate_proj = self.hidden_size * self.intermediate_size
    up_proj = self.hidden_size * self.intermediate_size
    down_proj = self.intermediate_size * self.hidden_size
    return (gate_proj + up_proj + down_proj) * self.param_bpe
  def intermediate_layer_nums(self):
    return self.num_hidden_layers

  def kv_cache_layer_token_size(self):
    return 2 * self.num_key_value_heads * self.head_dim * self.kv_bpe
  def kv_cache_layer_nums(self):
    return self.num_hidden_layers

  def report(self):
    print(f"Config: {config_file}")
    print(f"param_bpe={self.param_bpe}, kv_cache_bpe={self.kv_bpe}, seq_len={self.seq_len}")

    total_parameters_size = (
      self.embedding_layer_size() * self.embedding_layer_nums() + 
      self.attention_layer_size() * self.attention_layer_nums() + 
      self.intermediate_layer_size() * self.intermediate_layer_nums())

    sizes = [
      ["Layer", "Bytes Per Layer", "Layers", "Total Bytes"],
      ["Embedding Layer", self.embedding_layer_size(), self.embedding_layer_nums(), self.embedding_layer_size() * self.embedding_layer_nums()],
      ["Attention Layer", self.attention_layer_size(), self.attention_layer_nums(), self.attention_layer_size() * self.attention_layer_nums()],
      ["Intermediate Layer", self.intermediate_layer_size(), self.intermediate_layer_nums(), self.intermediate_layer_size() * self.intermediate_layer_nums()],
      ["Total Parameters", "", "", total_parameters_size],
    ]
    print(tabulate(sizes, headers='firstrow', tablefmt='fancy_grid',stralign='center', numalign='right'))

    kv_cache_sizes = [
      ["Batch", "SeqLength", "Bytes Per Layer Per Token", "Layers", "Total KV Cache Bytes"],
      [self.batch, self.seq_len, self.kv_cache_layer_token_size(), self.kv_cache_layer_nums(), self.batch * self.seq_len * self.kv_cache_layer_token_size() * self.kv_cache_layer_nums()],
    ]
    print(tabulate(kv_cache_sizes, headers='firstrow', tablefmt='fancy_grid',stralign='center', numalign='right'))



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Calculate model sizes from a config file.")
  parser.add_argument('config_file', type=str, help='Path to the JSON configuration file')
  parser.add_argument('--parameter_bpe', type=int, default=1, help='Bytes Per Element of the model parameter (default: 1 for fp8)')
  parser.add_argument('--kvcache_bpe', type=int, default=2, help='Byte Per Element of the kv-cache (default: 2 for fp16)')
  parser.add_argument('--seq_len', type=int, default=-1, help='Sequence length (default: max_position_embeddings in config)')
  parser.add_argument('--batch', type=int, default=1, help='Batch Size (default: 1)')

  args = parser.parse_args()
  
  if not os.path.isfile(args.config_file):
    parser.print_help()
    exit(1)

  config_file = args.config_file
  with open(config_file, 'r') as file:
    config = json.load(file)
    if args.seq_len == -1:
      args.seq_len = config['max_position_embeddings']

    for arch in config['architectures']:
      if arch.find('CausalLM') != -1:
        model = CausalLM(config, args.parameter_bpe, args.kvcache_bpe, args.seq_len, args.batch)
        model.report()
        break

