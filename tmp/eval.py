import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test1(model_name):
  # 1. 加载 tokenizer 和模型
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
  model.eval()

  # 2. 将模型移动到适当的设备（如果有 GPU，使用 GPU）
  device = torch.device("cuda" if torch.cuda.is_available() else "meta")
  model.to(device)

  # 3. 准备输入
  input_text = "how are you"
  input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

  # 4. 生成输出
  with torch.profiler.profile(
      activities=[
          torch.profiler.ProfilerActivity.CPU,
          torch.profiler.ProfilerActivity.CUDA],
      record_shapes=True,
      with_stack=True,
      profile_memory=True
  ) as prof:
      output = model.generate(input_ids, max_length=10)

  # 打印各个算子的输入输出形状
  print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=1000))

  prof.export_chrome_trace("prof_data.json")


  # 5. 解码输出
  output_text = tokenizer.decode(output[0], skip_special_tokens=True)

  # 6. 打印生成的文本
  print(output_text)

def test2(model_name):
  import torch
  from transformers import AutoModelForCausalLM, AutoTokenizer

  # 1. 加载 tokenizer 和模型
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
  model.eval()

  print(model)

  # 2. 将模型移动到适当的设备（如果有 GPU，使用 GPU）
  device = torch.device("cuda" if torch.cuda.is_available() else "mps")
  model.to(device)

  # 3. 定义一个钩子函数来捕获输出
  def hook_fn(module, input, output):
      # if isinstance(output, tuple):
      #     print(f"Layer: {module.__class__.__name__}, Output Shape: {output[0].shape}")
      # elif isinstance(output, torch.Tensor):
      #     print(f"Layer: {module.__class__.__name__}, Output Shape: {output.shape}")
      # elif hasattr(output, 'logits'):
      #     print(f"Layer: {module.__class__.__name__}: logits shape: {output.logits.shape}")
      #     if output.past_key_values is not None:
      #       # 打印 past_key_values 中每个元素的形状
      #       for i, past in enumerate(output.past_key_values):
      #         if past is not None:
      #           print(f"  Output Shape (past_key_values[{i}]): {[kv.shape for kv in past]}")
      pass

  # 4. 注册钩子函数到每一层
  hooks = []
  for layer in model.modules():
      print(layer)
      hooks.append(layer.register_forward_hook(hook_fn))

  # 5. 准备输入
  input_text = "how are you"
  input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
  

  # 6. 生成输出并打印每一层输出形状
  with torch.no_grad():
      output = model.generate(input_ids, max_length=input_ids.shape[1]+2)

  # 7. 解码输出
  output_text = tokenizer.decode(output[0], skip_special_tokens=True)

  # 8. 打印生成的文本
  print("Generated Text:", output_text)

  # 9. 移除钩子
  for hook in hooks:
      hook.remove()

if __name__ == "__main__":
    model_name = "/Users/jiamingqiao/WorkSpace/huggingface/models/chatglm2-6b"
    test1(model_name)
