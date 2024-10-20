import os
import platform
import signal
from transformers import AutoTokenizer, AutoModel
import readline
import time

class LLMArgs:
  def __init__(self, tokenizer, model_name):
    self.tokenizer = tokenizer
    self.mode_name = model_name

args = LLMArgs("/Users/jiamingqiao/WorkSpace/huggingface/models/chatglm2-6b", 
               "/Users/jiamingqiao/WorkSpace/huggingface/models/chatglm2-6b")

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
model = AutoModel.from_pretrained(args.mode_name, trust_remote_code=True).half().to('mps')

model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main():
    history = []
    global stop_stream
    print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        count = 0
        duration = 0.0
        time_start = time.perf_counter()
        for response, history in model.stream_chat(tokenizer, query, history=history):
            time_stop = time.perf_counter()
            duration += time_stop - time_start
            time_start = time_stop
            if stop_stream:
                stop_stream = False
                break
            else:
                count += 1
                if count % 8 == 0:
                    #os.system(clear_command)
                    print(build_prompt(history), flush=True)
                    signal.signal(signal.SIGINT, signal_handler)
        #os.system(clear_command)
        print(build_prompt(history), flush=True)
        if count > 0:
          print('count={0}, time={1}s: {2} tps'.format(count, duration, count/duration))

def test_once():
    query = "how are you"
    history = []
    count = 0
    time_start = time.perf_counter()
    for response, history in model.stream_chat(tokenizer, query, history=history):
      count += 1
    time_stop = time.perf_counter()
    print(history)

if __name__ == "__main__":
    main()

