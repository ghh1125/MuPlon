import json
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torch.nn as nn
from pytorch_pretrained_bert.tokenization import BertTokenizer
import random
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings
import logging
warnings.filterwarnings("ignore", category=UserWarning)  # 忽略用户警告
logging.getLogger('pgmpy').setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning, message="overflowing tokens are not returned")



from pytorch_pretrained_bert.tokenization import BertTokenizer

class InputExample(object):
    def __init__(self, unique_id, text_a, text_b, label, index, is_claim, is_evidence, cos=None):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.index = index
        self.is_claim = is_claim
        self.is_evidence = is_evidence
        self.cos = cos  # Add cosine similarity



class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids, label, index, is_claim, is_evidence, cos):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.label = label
        self.index = index
        self.is_claim = is_claim
        self.is_evidence = is_evidence
        self.cos = cos  # Add cosine similarity


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
                label=example.label,
                index=example.index,
                is_claim=example.is_claim,
                is_evidence = example.is_evidence,
                cos = example.cos))
    return features


def convert_examples_to_features_LLM(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    # 确保 tokenizer 有一个 padding token
    if tokenizer.pad_token is None:
        # 使用 eos_token 作为 pad_token，或者定义一个新的 pad_token
        tokenizer.pad_token = tokenizer.eos_token

    features = []
    for (ex_index, example) in enumerate(examples):
        # Tokenize the text_a and text_b using LLaMA tokenizer
        if example.text_b:
            encoded = tokenizer(
                example.text_a,
                example.text_b,
                max_length=seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            # 如果存在第二段文本，设置 input_type_ids：0表示 text_a，1表示 text_b
            input_type_ids = [0] * seq_length
            input_type_ids[len(encoded['input_ids'][0]) // 2:] = [1] * (
                        len(encoded['input_ids'][0]) - len(encoded['input_ids'][0]) // 2)

        else:
            encoded = tokenizer(
                example.text_a,
                max_length=seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            # 如果没有第二段文本，input_type_ids 全部为 0
            input_type_ids = [0] * seq_length

        # Extract input_ids and attention_mask from the encoded inputs
        input_ids = encoded["input_ids"].squeeze().tolist()
        input_mask = encoded["attention_mask"].squeeze().tolist()

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokenizer.convert_ids_to_tokens(input_ids),
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,  # 确保 input_type_ids 被正确设置
                label=example.label,
                index=example.index,
                is_claim=example.is_claim,
                is_evidence=example.is_evidence,
                cos=example.cos
            )
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


# def read_examples(input_file, max_evi_num):
#     """Read a list of `InputExample`s from an input file."""
#     examples = []
#     unique_id = 0
#     all_sent_labels = [] # 有几个句子,包括claim
#     all_evi_labels = [] # 句子中哪几个是证据，包括claim
#     with open(input_file, "r", encoding='utf-8') as reader:
#         while True:
#             line = reader.readline()
#             if not line:
#                 break
#             instance = json.loads(line.strip())
#             index = instance["id"]
#             claim = instance['claim']
#             claim = process_sent(claim)
#             label_map = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
#             label = label_map[instance['label']]
#             examples.append(InputExample(unique_id=unique_id, text_a=claim, text_b=None, label=label, index=index, is_claim=True, is_evidence=False))
#             unique_id += 1

#             sent_label = [1]
#             for evidence in instance['evidence'][0:max_evi_num]:
#                 # examples.append(InputExample(unique_id=unique_id, text_a=process_sent(evidence[2]), text_b=None, label=label, index=index, is_claim=False, is_evidence=True))
#                 # unique_id += 1
#                 examples.append(InputExample(unique_id=unique_id, text_a=process_sent(evidence[2]), text_b=claim, label=label, index=index, is_claim=True, is_evidence=True))
#                 unique_id += 1
#                 sent_label.append(1)
#             for i in range(max_evi_num-len(instance['evidence'][0:max_evi_num])):
#                 examples.append(InputExample(unique_id=unique_id, text_a="[PAD]", text_b=None, label=label, index=index, is_claim=True, is_evidence=True))
#                 unique_id += 1
#                 sent_label.append(0)
#             evi_label = instance['evi_labels'][0:max_evi_num]
#             evi_label = evi_label + [0] * (max_evi_num - len(evi_label))
#             evi_label = [1] + evi_label
#             all_sent_labels.append(sent_label)
#             all_evi_labels.append(evi_label)
            
#     return examples, all_sent_labels, all_evi_labels

def compute_similarity(text1, text2):
    """
    使用某种方法计算两个文本之间的相似度，比如余弦相似度。这里使用一个简单的实现作为示例。
    """
    # 将文本向量化，例如使用TF-IDF或BERT嵌入
    # For simplicity, here we assume text1_vec and text2_vec are obtained by some embedding method
    text1_vec = np.random.rand(1, 768)  # 需要替换成实际的向量化操作
    text2_vec = np.random.rand(1, 768)  # 需要替换成实际的向量化操作

    similarity = cosine_similarity(text1_vec, text2_vec)[0][0]  # 计算余弦相似度
    return similarity


from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.estimators import BayesianEstimator
from pgmpy.factors.discrete import TabularCPD
import pandas as pd



def print_cpd_states(model):
    for cpd in model.get_cpds():
        print(f"Variable: {cpd.variable}")
        print(f"Values: {cpd.values}")
        print(f"State Names: {cpd.state_names}")

# 定义贝叶斯网络结构
def build_bayes_network(num_evidences):
    # 创建贝叶斯网络的边结构
    edges = [('claim', f'evidence{i}') for i in range(num_evidences)]
    model = BayesianNetwork(edges)
    return model


# 使用数据进行参数学习
def train_bayes_network(model, data, num_evidences):
    # 定义 Claim 的条件概率分布（先验概率）
    cpd_claim = TabularCPD(variable='claim', variable_card=2, values=[[0.5], [0.5]])  # 先验概率假设均匀分布

    # 创建每个 Evidence 的条件概率分布 (CPD)
    cpds = [cpd_claim]
    for i in range(num_evidences):
        cpd_evidence = TabularCPD(
            variable=f'evidence{i}',
            variable_card=2,
            values=[[0.8, 0.1], [0.2, 0.9]],  # 初始值
            evidence=['claim'],
            evidence_card=[2]
        )
        cpds.append(cpd_evidence)

    # 将所有的 CPD 添加到模型中
    model.add_cpds(*cpds)
    model.check_model()

    # print_cpd_states(model)

    # 使用数据进行参数学习 (最大似然估计)
    mle = MaximumLikelihoodEstimator(model, data)

    # for node in model.nodes():
    #     cpd = mle.estimate_cpd(node)
    #     model.add_cpds(cpd)

    model.check_model()
    # print("OK")
    # print_cpd_states(model)
    return model


# 计算后验概率(后门准则)
def compute_posterior_probabilities_with_control(model, evidence_data, control_variables):
    inference = VariableElimination(model)


    try:
        query_result = inference.query(variables=['claim'], evidence={**evidence_data, **control_variables})
    except KeyError as e:
        print(f"KeyError: {e}")
        print("Possible Issues with State Values or Evidence")
        raise

    return query_result


# 计算样本权重
def compute_weights_for_samples(evidence_data, control_variable_probs):
    weights = {}
    for key, value in evidence_data.items():
        if key in control_variable_probs:
            prob = control_variable_probs[key]
            if prob > 0:
                weights[key] = 1 / prob
            else:
                weights[key] = 0
    return weights

def adjust_sample_weights(data, weights):
    adjusted_data = []
    for sample in data:
        weight = weights.get(sample['evidence_key'], 1)  # 使用默认权重1
        adjusted_sample = sample.copy()
        adjusted_sample['weight'] = weight
        adjusted_data.append(adjusted_sample)
    return adjusted_data

# 将 evidence_data 转换为 DataFrame
def convert_to_dataframe(evidence_data, num_evidences):
    # 初始化数据字典
    data = {
        'claim': [],
        **{f'evidence{i}': [] for i in range(num_evidences)}
    }

    # 填充数据
    for example in evidence_data:
        data['claim'].append(int(example.is_claim))  # Claim 的值
        for i in range(num_evidences):
            if f'evidence{i}' not in data:
                data[f'evidence{i}'] = []
            data[f'evidence{i}'].append(int(example.cos > 0.5))  # 根据cos值将evidence处理为0或1

    return pd.DataFrame(data)


def read_examples(input_file, max_evi_num):
    """Read a list of `InputExample`s from an input file."""
    label_map = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
    examples = []
    unique_id = 0
    all_sent_labels = [] # 有几个句子,包括claim
    all_evi_labels = [] # 句子中哪几个是证据，包括claim
    i = 0 # 记录example的位置
    logging.info(f'Reading examples from {input_file}...')
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            evidence_data = []
            instance = json.loads(line.strip())
            index = instance["index"]
            claim = instance['claim']
            label = label_map[instance['label']]


            # 记录当前实例的开始位置
            start_index = len(examples)

            examples.append(InputExample(unique_id=unique_id, text_a=claim, text_b=None, label=label, index=index, is_claim=True, is_evidence=False,cos=1.0))


            unique_id += 1
            i = i + 1



            evidence_data.append(InputExample(unique_id=unique_id, text_a=claim, text_b=None, label=label, index=index, is_claim=True, is_evidence=False,cos=1.0))
            sum_cos = 0
            sent_label = [1]


            for evidence in instance['evidences'][0:max_evi_num]:
                # examples.append(InputExample(unique_id=unique_id, text_a=evidence, text_b=None, label=label, index=index, is_claim=False, is_evidence=True))
                # unique_id += 1
                cos = compute_similarity(claim,evidence)

                sum_cos = sum_cos + cos
                i = i + 1
                examples.append(InputExample(unique_id=unique_id, text_a=evidence, text_b=claim, label=label, index=index, is_claim=True, is_evidence=True, cos=cos))
                evidence_data.append(InputExample(unique_id=unique_id, text_a=evidence, text_b=claim, label=label, index=index, is_claim=True, is_evidence=True, cos=cos))
                unique_id += 1
                sent_label.append(1)

            # Bayes
            # 归一化
            k = 0
            for example in evidence_data:
                k = k+1
                example.cos /= sum_cos  # Normalize cosine similarity
            num_evidences = len(evidence_data) - 1  # 减去 claim 的部分
            ave = sum_cos / k
            model = build_bayes_network(num_evidences)
            data = convert_to_dataframe(evidence_data, num_evidences)

            trained_model = train_bayes_network(model, data, num_evidences)

            evidence_data_for_query = {f'evidence{i}': int(evidence_data[i + 1].cos > ave) for i in
                                       range(num_evidences)}
            control_variable_probs = {f'evidence{i}': int(evidence_data[i + 1].cos > ave) for i in range(num_evidences)}
            posterior_prob = compute_posterior_probabilities_with_control(trained_model, evidence_data_for_query,
                                                                          control_variable_probs)

            # print(f"Posterior probability for instance {index}: {posterior_prob}")
            for j, example in enumerate(evidence_data):
                example.cos = posterior_prob.values[j] if j < len(posterior_prob.values) else 0

            examples[start_index:start_index + len(
                    evidence_data)] = evidence_data  # Replace corresponding part of the main list

            # # Build a Bayesian Network model
            # evidence_probabilities = [example.cos for example in evidence_data]
            # if not evidence_probabilities:
            #     continue
            # # 先验分布
            # prior_claim = 1.0 / len(evidence_probabilities)
            # # 贝叶斯定理计算后验概率
            # posterior_probabilities = [prior_claim * evidence_prob / sum(evidence_probabilities) for evidence_prob
            #                                in evidence_probabilities]
            # # 更新每个证据的概率
            # for i, example in enumerate(evidence_data):
            #     example.cos = posterior_probabilities[i]
            #
            #
            # examples[start_index:start_index + len(evidence_data)] = evidence_data  # Replace corresponding part of the main list

            for i in range(max_evi_num-len(instance['evidences'][0:max_evi_num])):
                examples.append(InputExample(unique_id=unique_id, text_a="[PAD]", text_b=None, label=label, index=index, is_claim=True, is_evidence=True,cos=0.0))
                unique_id += 1
                i = i + 1
                sent_label.append(0)
            evi_label = instance['evi_labels'][0:max_evi_num]
            evi_label = evi_label + [0] * (max_evi_num - len(evi_label))
            evi_label = [1] + evi_label
            all_sent_labels.append(sent_label)
            all_evi_labels.append(evi_label)

    return examples, all_sent_labels, all_evi_labels


def build_dataset(file_path, evi_max_num, seq_length=512):
    tokenizer = BertTokenizer.from_pretrained("pretrained_models/BERT-Pair", do_lower_case=True)

    # Read examples and convert them to features
    examples, all_sent_labels, all_evi_labels = read_examples(file_path, evi_max_num)
    # for example in examples[:1]:
    #     print(
    #         f"Claim: {example.text_a}, Label: {example.label}, Is Claim: {example.is_claim}, Is Evidence: {example.is_evidence}")
    #
    # print("Sentence Labels:", all_sent_labels[:1])
    # print("Evidence Labels:", all_evi_labels[:1])

    features = convert_examples_to_features(examples=examples, seq_length=seq_length, tokenizer=tokenizer)

    # Convert features to tensors
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.input_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    # Add cosine similarity as a tensor
    all_cosine_similarities = torch.tensor(
        [f.cos for f in features], dtype=torch.float
    )

    # Reshape tensors to match the expected shape
    all_input_ids = all_input_ids.view(-1, 1 + evi_max_num, seq_length)  # [10000, 6, 512]
    all_input_mask = all_input_mask.view(-1, 1 + evi_max_num, seq_length)
    all_segment_ids = all_segment_ids.view(-1, 1 + evi_max_num, seq_length)
    all_labels = all_labels.view(-1, 1 + evi_max_num)[:, 0]

    # Reshape cosine similarity tensor to match the shape [batch_size, 1 + evi_max_num]
    all_cosine_similarities = all_cosine_similarities.view(-1, 1 + evi_max_num)

    # Convert sentence and evidence labels to tensors
    all_sent_labels = torch.LongTensor(all_sent_labels)
    all_evi_labels = torch.LongTensor(all_evi_labels)

    # Include cosine similarity tensor in the dataset
    train_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_labels,
        all_sent_labels, all_evi_labels, all_cosine_similarities
    )

    return train_data

from transformers import LlamaModel, LlamaTokenizer
def build_dataset_LLM(file_path, evi_max_num, seq_length=1):
    tokenizer = LlamaTokenizer.from_pretrained("/home/disk2/ghh/llama")

    # Read examples and convert them to features
    examples, all_sent_labels, all_evi_labels = read_examples(file_path, evi_max_num)
    features = convert_examples_to_features_LLM(examples=examples, seq_length=seq_length, tokenizer=tokenizer)

    # Convert features to tensors
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.input_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    # Add cosine similarity as a tensor
    all_cosine_similarities = torch.tensor(
        [f.cos for f in features], dtype=torch.float
    )

    # Reshape tensors to match the expected shape
    all_input_ids = all_input_ids.view(-1, 1 + evi_max_num, seq_length)  # [10000, 6, 512]
    all_input_mask = all_input_mask.view(-1, 1 + evi_max_num, seq_length)
    all_segment_ids = all_segment_ids.view(-1, 1 + evi_max_num, seq_length)
    all_labels = all_labels.view(-1, 1 + evi_max_num)[:, 0]

    # Reshape cosine similarity tensor to match the shape [batch_size, 1 + evi_max_num]
    all_cosine_similarities = all_cosine_similarities.view(-1, 1 + evi_max_num)

    # Convert sentence and evidence labels to tensors
    all_sent_labels = torch.LongTensor(all_sent_labels)
    all_evi_labels = torch.LongTensor(all_evi_labels)

    # Include cosine similarity tensor in the dataset
    train_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_labels,
        all_sent_labels, all_evi_labels, all_cosine_similarities
    )

    return train_data
