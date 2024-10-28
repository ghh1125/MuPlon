import os
import json
from langchain_community.llms import Ollama
from tqdm import tqdm
import torch
import re

# 定义模型列表
# model_names = [
#     "gemma2:27b",
#     "gemma2:latest",
#     "phi:latest",
#     "glm4:latest",
#     "llama3:latest",
#     "llama2:latest",
#     "mistral:latest",
#     "qwen2:72b",
#     "qwen:7b",
#     "qwen:72b"
# ]
# model_names = [
#     "llama3:latest",
#     "llama2:latest",
#     "mistral:latest",
#     "qwen:7b"
# ]

model_names = [
    # "EntropyYue/chatglm3:6b",
    # "gemma2:latest",
    # "phi:latest",
    # "llama3:latest",
    # "llama2:latest",
    # "mistral:latest",
    # "qwen:7b",
    "gemma2:27b"
]

# json_files = [
#     "data/fever/fever_dev-2class.json",
#     "data/politihop/original/politihop_valid_even.json"
# ]

# 读取所有JSON文件的路径
json_files = [
    # "data/climate-fever/two/climate-fever.json"
    "data/cladder/cladder_sol/test-generate-anti.json",
    "data/cladder/cladder_sol/test-generate-det.json",
    "data/cladder/cladder_sol/test-generate-easy.json",
    "data/cladder/cladder_sol/test-generate-hard.json",
    "data/cladder/cladder_sol/test-generate-non.json",
    "data/cladder/cladder_sol/cladder-v1-questions.json",
    "data/cladder/llamagenerareddata/test-generate-anti.json",
    "data/cladder/llamagenerareddata/test-generate-det.json",
    "data/cladder/llamagenerareddata/test-generate-easy.json",
    "data/cladder/llamagenerareddata/test-generate-hard.json",
    "data/cladder/llamagenerareddata/test-generate-non.json",
    "data/cladder/llamagenerareddata/cladder-v1-questions.json"
]
# json_files = [
#     "data/fever/fever_dev-2class.json",
#     "data/fever/fever_dev-2class-mh.json",
#     "data/fever/fever2-2class.json",
#     "data/fever/fever2-2class-mh.json",
#     "data/politihop/original/politihop_test_adv-2class.json",
#     "data/politihop/symmetric/Symmetric_PolitiHop_ShareEvi.json",
#     "data/politihop/symmetric/Symmetric_PolitiHop.json",
#     "data/politihop/original/politihop_valid_all.json",
#     "data/politihop/original/politihop_test_all.json",
#     "data/politihop/original/politihop_valid_adv.json",
#     "data/politihop/original/politihop_test_adv.json",
#     "data/politihop/original/politihop_valid_even.json",
#     "data/politihop/original/politihop_test_even.json",
#     "data/politihop/original/Hard_PolitiHop_by_gpt-3.5.json",
#     "data/politihop/original/Hard_PolitiHop_by_gpt-4.0.json",
#     "data/cladder/cladder_sol/test-generate-anti.json",
#     "data/cladder/cladder_sol/test-generate-det.json",
#     "data/cladder/cladder_sol/test-generate-easy.json",
#     "data/cladder/cladder_sol/test-generate-hard.json",
#     "data/cladder/cladder_sol/test-generate-non.json",
#     "data/cladder/cladder_sol/cladder-v1-questions.json",
#     "data/cladder/llamagenerareddata/test-generate-anti.json",
#     "data/cladder/llamagenerareddata/test-generate-det.json",
#     "data/cladder/llamagenerareddata/test-generate-easy.json",
#     "data/cladder/llamagenerareddata/test-generate-hard.json",
#     "data/cladder/llamagenerareddata/test-generate-non.json",
#     "data/cladder/llamagenerareddata/cladder-v1-questions.json"
# ]

# 创建结果文件夹
# 创建结果文件夹和模型的子目录
base_output_dir = "result_LLM_two_classifier"
# model_output_dir = os.path.join(base_output_dir, model_name)
# if not os.path.exists(model_output_dir):
#     os.makedirs(model_output_dir)

case_prompt = """
I say :I am going to give you a series of natural language inference tasks. Based on the evidence I provide (note: the evidence may contain errors), please determine if the claim should be labeled as “SUPPORT” or “REFUTE”.Please answer with only one of these options.Don't answer anything else!”

LLM responds: "Okay."
"""

def generate_prompt(claim, evidences, evi_labels):
    """
    生成发送给LLM的提示词，根据claim和evidence进行推理
    """
    prompt = f"Claim: {claim}\n\nEvidences:\n"
    for idx, evidence in enumerate(evidences):
        prompt += f"Evidence {idx + 1}: {evidence} \n"

    prompt += "\nBased on the evidences above(note: the evidence may contain errors,you need to determine which ones are correct), does the claim 'REFUTE' or 'SUPPORT'?,Please answer with only one of these options.Don't answer anything else!"

    prompt = case_prompt + prompt + "You just need to answer:'REFUTE' or 'SUPPORT',Please answer with only one of these options.Don't answer anything else!"
    return prompt

def extract_valid_option(result, valid_options):
    result = result.upper()

    # if "NOT ENOUGH INFO" in result:
    #     return "NOT ENOUGH INFO"

    for option in valid_options:
        if option in result:
            return option

    return "REFUTES"


def process_json_file(file_path, llm):
    i = 0
    results = []
    valid_options = {"REFUTE", "SUPPORT", "REFUTES", "SUPPORTS", "NOT ENOUGH INFO"}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="Processing lines"):
                i += 1
                # if i == 5:
                #     break
                try:
                    item = json.loads(line)
                    claim = item["claim"]
                    evidences = item["evidences"]
                    evi_labels = item["evi_labels"]
                    original_label = item["label"]

                    prompt = generate_prompt(claim, evidences, evi_labels)
                    result = llm.invoke(prompt).strip()
                    # print("最初的回答：",result)
                    cc = result.upper()
                    result = extract_valid_option(result, valid_options)
                    # print("处理后的回答：",result)
                    if result == "REFUTE" or result == "SUPPORT":
                        result = result + "S"
                    results.append({
                        "original_label": original_label,
                        "model_output": result,
                        "result": cc
                    })
                except json.JSONDecodeError as e:
                    print(f"JSON Decode Error: {e} for line: {line}")
    except Exception as e:
        print(f"An error occurred: {e} in file {file_path}")

    return results

def free_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA memory has been freed.")

def process_all_files(json_files, model_names):
    for model_name in model_names:
        free_cuda_memory()
        # 初始化模型
        print(f"Initializing model: {model_name}")
        llm = Ollama(model=model_name)

        # 创建模型的结果文件夹
        model_output_dir = os.path.join("result_LLM_two_classifier", model_name)
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)

        for file_path in json_files:
            print(f"Processing file: {file_path} with model: {model_name}")
            results = process_json_file(file_path, llm)

            # 获取父目录名并规范化（去除路径中的非法符号）
            parent_dir = os.path.basename(os.path.dirname(file_path))
            parent_dir_sanitized = re.sub(r'[\/:*?"<>|]', '_', parent_dir)

            # 规范化模型名称（去除非法字符）
            model_name_sanitized = re.sub(r'[\/:*?"<>|]', '_', model_name)

            # 确定保存文件名，将父目录名、模型名称添加到文件名中，防止文件名冲突
            base_filename = os.path.basename(file_path).replace(".json",
                                                                f"_{parent_dir_sanitized}_{model_name_sanitized}_results.json")
            output_file_path = os.path.join(model_output_dir, base_filename)

            # 将结果保存到一个新的JSON文件，只包含原始label和推理结果
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

            print(f"Results saved to {output_file_path}")


# 调用主函数处理所有文件和模型
process_all_files(json_files, model_names)
