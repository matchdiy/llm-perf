from transformers import AutoConfig, AutoModel
from torchsummary import summary
from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        # print(name)
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


# 指定你要分析的模型名称
#model_name = "/Users/jiamingqiao/WorkSpace/huggingface/models/chatglm2-6b"
model_name = "/Users/jiamingqiao/WorkSpace/models/Meta-Llama-3-70B"
# 1. 加载模型的配置文件，不加载预训练的权重
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
print(config)
# 2. 使用配置文件来构建模型架构，不加载权重
model = AutoModel.from_config(config, trust_remote_code=True)
model.eval()

# 3. 打印模型的架构
print(model)

# 可选：如果你想更详细地查看模型的层结构和参数信息
try:
    summary(model, input_size=(1, 512))  # 输入大小可以根据你实际的模型输入调整
except:
    print("torchsummary 仅适用于支持的模型结构")


count_parameters(model)
