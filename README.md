---
language:
- en
- zh
- id
- vi
- th
- ms
license: other
tags:
- sea
- multilingual
license_name: seallms
license_link: https://huggingface.co/SeaLLMs/SeaLLM-13B-Chat/blob/main/LICENSE
model-index:
- name: SeaLLMs-v3-7B-Chat
  results:
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: IFEval (0-Shot)
      type: HuggingFaceH4/ifeval
      args:
        num_few_shot: 0
    metrics:
    - type: inst_level_strict_acc and prompt_level_strict_acc
      value: 43.77
      name: strict accuracy
    source:
      url: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard?query=SeaLLMs/SeaLLMs-v3-7B-Chat
      name: Open LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: BBH (3-Shot)
      type: BBH
      args:
        num_few_shot: 3
    metrics:
    - type: acc_norm
      value: 33.8
      name: normalized accuracy
    source:
      url: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard?query=SeaLLMs/SeaLLMs-v3-7B-Chat
      name: Open LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: MATH Lvl 5 (4-Shot)
      type: hendrycks/competition_math
      args:
        num_few_shot: 4
    metrics:
    - type: exact_match
      value: 15.11
      name: exact match
    source:
      url: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard?query=SeaLLMs/SeaLLMs-v3-7B-Chat
      name: Open LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: GPQA (0-shot)
      type: Idavidrein/gpqa
      args:
        num_few_shot: 0
    metrics:
    - type: acc_norm
      value: 6.49
      name: acc_norm
    source:
      url: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard?query=SeaLLMs/SeaLLMs-v3-7B-Chat
      name: Open LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: MuSR (0-shot)
      type: TAUR-Lab/MuSR
      args:
        num_few_shot: 0
    metrics:
    - type: acc_norm
      value: 10.47
      name: acc_norm
    source:
      url: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard?query=SeaLLMs/SeaLLMs-v3-7B-Chat
      name: Open LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: MMLU-PRO (5-shot)
      type: TIGER-Lab/MMLU-Pro
      config: main
      split: test
      args:
        num_few_shot: 5
    metrics:
    - type: acc
      value: 32.16
      name: accuracy
    source:
      url: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard?query=SeaLLMs/SeaLLMs-v3-7B-Chat
      name: Open LLM Leaderboard
---

# *SeaLLMs-v3* - Large Language Models for Southeast Asia

<p align="center">
<a href="https://damo-nlp-sg.github.io/SeaLLMs/" target="_blank" rel="noopener">Website</a>
&nbsp;&nbsp;
<a href="https://huggingface.co/SeaLLMs/SeaLLMs-v3-7B-Chat" target="_blank" rel="noopener">Model</a>
&nbsp;&nbsp;
<a href="https://huggingface.co/spaces/SeaLLMs/SeaLLM-Chat" target="_blank" rel="noopener"> 🤗 DEMO</a>
&nbsp;&nbsp;
<a href="https://github.com/DAMO-NLP-SG/SeaLLMs" target="_blank" rel="noopener">Github</a>
&nbsp;&nbsp;
<a href="https://arxiv.org/pdf/2407.19672" target="_blank" rel="noopener">[NEW] Technical Report</a>
</p>

We introduce **SeaLLMs-v3**, the latest series of the SeaLLMs (Large Language Models for Southeast Asian languages) family. It achieves state-of-the-art performance among models with similar sizes, excelling across a diverse array of tasks such as world knowledge, mathematical reasoning, translation, and instruction following. In the meantime, it was specifically enhanced to be more trustworthy, exhibiting reduced hallucination and providing safe responses, particularly in queries closed related to Southeast Asian culture.

## 🔥 Highlights
- State-of-the-art performance compared to open-source models of similar sizes, evaluated across various dimensions such as human exam questions, instruction-following, mathematics, and translation.
- Significantly enhanced instruction-following capability, especially in multi-turn settings.
- Ensures safety in usage with significantly reduced instances of hallucination and sensitivity to local contexts.

## Uses

SeaLLMs is tailored for handling a wide range of languages spoken in the SEA region, including English, Chinese, Indonesian, Vietnamese, Thai, Tagalog, Malay, Burmese, Khmer, Lao, Tamil, and Javanese.

This page introduces the **SeaLLMs-v3-7B-Chat** model, specifically fine-tuned to follow human instructions effectively for task completion, making it directly applicable to your applications.

You may also refer to the [SeaLLMs-v3-1.5B-Chat](https://huggingface.co/SeaLLMs/SeaLLMs-v3-1.5B-Chat) model which requires much lower computational resources and can be easily loaded locally.


### Get started with `Transformers`

To quickly try the model, we show how to conduct inference with `transformers` below. Make sure you have installed the latest transformers version (>4.40).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
  "SeaLLMs/SeaLLMs-v3-7B-Chat", # can change to "SeaLLMs/SeaLLMs-v3-1.5B-Chat" if your resource is limited
  torch_dtype=torch.bfloat16, 
  device_map=device
)
tokenizer = AutoTokenizer.from_pretrained("SeaLLMs/SeaLLMs-v3-7B-Chat")

# prepare messages to model
prompt = "Hiii How are you?"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(device)
print(f"Formatted text:\n {text}")
print(f"Model input:\n {model_inputs}")

generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True, eos_token_id=tokenizer.eos_token_id)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print(f"Response:\n {response[0]}")
```

You can also utilize the following code snippet, which uses the streamer `TextStreamer` to enable the model to continue conversing with you:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextStreamer

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
  "SeaLLMs/SeaLLMs-v3-7B-Chat",  # can change to "SeaLLMs/SeaLLMs-v3-1.5B-Chat" if your resource is limited
  torch_dtype=torch.bfloat16, 
  device_map=device
)
tokenizer = AutoTokenizer.from_pretrained("SeaLLMs/SeaLLMs-v3-7B-Chat")

# prepare messages to model
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
]

while True:
    prompt = input("User:")
    messages.append({"role": "user", "content": prompt})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, streamer=streamer)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    messages.append({"role": "assistant", "content": response})
```

### Inference with `vllm`

You can also conduct inference with [vllm](https://docs.vllm.ai/en/stable/index.html), which is a fast and easy-to-use library for LLM inference and serving. To use vllm, first install the latest version via `pip install vllm`.

```python
from vllm import LLM, SamplingParams

prompts = [
    "Who is the president of US?",
    "Can you speak Indonesian?"
]

llm = LLM(ckpt_path, dtype="bfloat16")
sparams = SamplingParams(temperature=0.1, max_tokens=512)
outputs = llm.generate(prompts, sparams)

# print out the model response
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}\nResponse: {generated_text}\n\n")
```

### Bias, Risks, and Limitations
<blockquote style="color:red">
<p><strong style="color: red">Terms of Use and License</strong>: 
By using our released weights, codes, and demos, you agree to and comply with the terms and conditions specified in our <a href="https://huggingface.co/SeaLLMs/SeaLLM-Chat-13b/edit/main/LICENSE" target="_blank" rel="noopener">SeaLLMs Terms Of Use</a>.
</blockquote>

> **Disclaimer**:
> We must note that even though the weights, codes, and demos are released in an open manner, similar to other pre-trained language models, and despite our best efforts in red teaming and safety fine-tuning and enforcement, our models come with potential risks, including but not limited to inaccurate, misleading or potentially harmful generation.
> Developers and stakeholders should perform their own red teaming and provide related security measures before deployment, and they must abide by and comply with local governance and regulations.
> In no event shall the authors be held liable for any claim, damages, or other liability arising from the use of the released weights, codes, or demos.



## Evaluation

We conduct our evaluation along two dimensions:

1. **Model Capability**: We assess the model's performance on human exam questions, its ability to follow instructions, its proficiency in mathematics, and its translation accuracy.
2. **Model Trustworthiness**: We evaluate the model's safety and tendency to hallucinate, particularly in the context of Southeast Asia.

### Model Capability

#### Multilingual World Knowledge - M3Exam
[M3Exam](https://arxiv.org/abs/2306.05179) consists of local exam questions collected from each country. It reflects the model's world knowledge (e.g., with language or social science subjects) and reasoning abilities (e.g., with mathematics or natural science subjects).

| Model            |   en |    zh |   id |   th |   vi |   avg |   avg_sea |
|:-----------------|-----:|------:|-----:|-----:|-----:|------:|----------:|
| Sailor-7B-Chat   | 0.66 | 0.652 | 0.475 | 0.462 | 0.513 | 0.552 | 0.483 |
| gemma-7b         | 0.732 | 0.519 | 0.475 | 0.46 | 0.594 | 0.556 | 0.510 |
| SeaLLM-7B-v2.5   | 0.758 | 0.581 | 0.499 | 0.502 | 0.622 | 0.592 | 0.541 |
| Qwen2-7B         | 0.815 | 0.874 | 0.53 | 0.479 | 0.628 | 0.665 | 0.546 |
| Qwen2-7B-Instruct| 0.809 | 0.88 | 0.558 | 0.555 | 0.624 | 0.685 | 0.579 |
| Sailor-14B       | 0.748 | 0.84 | 0.536 | 0.528 | 0.621 | 0.655 | 0.562 |
| Sailor-14B-Chat  | 0.749 | 0.843 | 0.553 | 0.566 | 0.637 | 0.67 | 0.585 |
| SeaLLMs-v3-7B       | 0.809 | 0.863 | 0.545 | 0.530 | 0.628 | 0.675 | 0.568 |
| **SeaLLMs-v3-7B-Chat**  | 0.809 | 0.874 | 0.558 | 0.569 | 0.649 | 0.692 | **0.592** |


#### Multilingual Instruction-following Capability - SeaBench
SeaBench consists of multi-turn human instructions spanning various task types. It evaluates chat-based models on their ability to follow human instructions in both single and multi-turn settings and assesses their performance across different task types. The dataset and corresponding evaluation code will be released soon!

| model           |   id<br>turn1 |   id<br>turn2 |   id<br>avg |   th<br>turn1 |   th<br>turn2 |   th<br>avg |   vi<br>turn1 |   vi<br>turn2 |   vi<br>avg |   avg |
|:----------------|------------:|------------:|---------:|------------:|------------:|---------:|------------:|------------:|---------:|------:|
| Qwen2-7B-Instruct|         5.93 |         5.84 |     5.89 |         5.47 |         5.20 |     5.34 |         6.17 |         5.60 |     5.89 |  5.70 |
| SeaLLM-7B-v2.5  |         6.27 |         4.96 |     5.62 |         5.79 |         3.82 |     4.81 |         6.02 |         4.02 |     5.02 |  5.15 |
| Sailor-14B-Chat |         5.26 |         5.53 |     5.40 |         4.62 |         4.36 |     4.49 |         5.31 |         4.74 |     5.03 |  4.97 |
| Sailor-7B-Chat  |         4.60 |         4.04 |     4.32 |         3.94 |         3.17 |     3.56 |         4.82 |         3.62 |     4.22 |  4.03 |
| **SeaLLMs-v3-7B-Chat** |         6.73 |         6.59 |     6.66 |         6.48 |         5.90 |     6.19 |         6.34 |         5.79 |     6.07 |  **6.31** |


#### Multilingual Math
We evaluate the multilingual math capability using the MGSM dataset. MGSM originally contains Chinese and Thai testing sets only, we use Google Translate to translate the same English questions into other SEA languages. Note that we adopt the tradition of each country to represent the number, e.g., in Indonesian and Vietnamese, dots are used as thousands separators and commas as decimal separators, the opposite of the English system.

| MGSM                      |    en |    id |    ms |    th |    vi |    zh |   avg |
|:--------------------------|------:|------:|------:|------:|------:|------:|------:|
| Sailor-7B-Chat            |  33.6 |  22.4 |  22.4 |  21.6 |  25.2 |  29.2 |  25.7 |
| Meta-Llama-3-8B-Instruct  |  77.6 |  48   |  57.6 |  56   |  46.8 |  58.8 |  57.5 |
| glm-4-9b-chat             |  72.8 |  53.6 |  53.6 |  34.8 |  52.4 |  70.8 |  56.3 |
| Qwen1.5-7B-Chat           |  64   |  34.4 |  38.4 |  25.2 |  36   |  53.6 |  41.9 |
| Qwen2-7B-instruct         |  82   |  66.4 |  62.4 |  58.4 |  64.4 |  76.8 |  68.4 |
| aya-23-8B                 |  28.8 |  16.4 |  14.4 |   2   |  16   |  12.8 |  15.1 |
| gemma-1.1-7b-it           |  58.8 |  32.4 |  34.8 |  31.2 |  39.6 |  35.2 |  38.7 |
| SeaLLM-7B-v2.5            |  79.6 |  69.2 |  70.8 |  61.2 |  66.8 |  62.4 |  68.3 |
| **SeaLLMs-v3-7B-Chat**    |  74.8 |  71.2 |  70.8 |  71.2 |  71.2 |  79.6 |  **73.1** |


#### Translation
We use the test sets from Flores-200 for evaluation and report the zero-shot chrF scores for translations between every pair of languages. Each row in the table below presents the average results of translating from various source languages into the target languages. The last column displays the overall average results of translating from any language to any other language for each model.

| model                                          |    en |    id |    jv |    km |    lo |    ms |    my |    ta |    th |    tl |    vi |    zh |   avg |
|:-----------------------------------------------|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|
|Meta-Llama-3-8B-Instruct            | 51.54 | 49.03 | 22.46 | 15.34 |  5.42 | 46.72 | 21.24 | 32.09 | 35.75 | 40.8  | 39.31 | 14.87 | 31.22 |
|Qwen2-7B-Instruct                         | 50.36 | 47.55 | 29.36 | 19.26 | 11.06 | 42.43 | 19.33 | 20.04 | 36.07 | 37.91 | 39.63 | 22.87 | 31.32 |
|Sailor-7B-Chat                            | 49.4  | 49.78 | 28.33 |  2.68 |  6.85 | 47.75 |  5.35 | 18.23 | 38.92 | 29    | 41.76 | 20.87 | 28.24 |
|SeaLLM-7B-v2.5                         | 55.09 | 53.71 | 18.13 | 18.09 | 15.53 | 51.33 | 19.71 | 26.1  | 40.55 | 45.58 | 44.56 | 24.18 | 34.38 |
|**SeaLLMs-v3-7B-Chat**                 | 54.68 | 52.52 | 29.86 | 27.3  | 26.34 | 45.04 | 21.54 | 31.93 | 41.52 | 38.51 | 43.78 | 26.1 | **36.52** |


### Model Trustworthiness

#### Hallucination
Performance of whether a model can refuse questions about the non-existing entity. The following is the F1 score. We use refuse as the positive label. Our test set consists of ~1k test samples per language. Each unanswerable question is generated by GPT4o. The ratio of answerable and unanswerable questions are 1:1. We define keywords to automatically detect whether a model-generated response is a refusal response. 

| Refusal-F1 Scores    |    en |    zh |    vi |    th |    id |    avg |
|:---------------------|------:|------:|------:|------:|------:|-------:|
| Qwen1.5-7B-Instruct  | 53.85 | 51.70 | 52.85 | 35.50  | 58.40  | 50.46  |
| Qwen2-7B-Instruct    | 58.79 | 33.08 | 56.21 | 44.60  | 55.98 | 49.73 |
| SeaLLM-7B-v2.5       | 12.90 |  0.77 |  2.45 | 19.42 |  0.78 |  7.26  |
| Sailor-7B-Chat       | 33.49 | 18.82 |  5.19 |  9.68 | 16.42 | 16.72  |
| glm-4-9b-chat        | 44.48 | 37.89 | 18.66 |  4.27 |  1.97 | 21.45  |
| Llama-3-8B-Instruct  | 72.08 |  0.00 |  1.23 |  0.80 |  3.91 | 15.60  |
| gemma-1.1-7b-it      | 52.39 | 27.74 | 23.96 | 22.97 | 31.72 | 31.76  |
| **SeaLLMs-v3-7B-Chat**      | 71.36 | 78.39 | 77.93 | 61.31 | 68.95 | **71.59** |


#### Safety
Multijaildataset consists of harmful prompts in multiple languages. We take those relevant prompts in SEA languages here and report their safe rate (the higher the better).  

| Model                   |     en |     jv |     th |     vi |     zh |    avg |
|:------------------------|-------:|-------:|-------:|-------:|------:|-------:|
| Qwen2-7B-Instruct       | 88.57 | 43.81 | 63.81 | 73.02 | 87.30  | 71.30  |
| Sailor-7B-Chat          | 78.73 | 54.92 | 62.22 | 67.62 | 76.19 | 67.94 |
| Meta-Llama-3-8B-Instruct| 88.25 | 26.35 | 71.11 | 69.84 | 77.14 | 66.54 |
| Sailor-14B-Chat         | 86.98 | 30.48 | 53.65 | 60.95 | 72.70  | 60.95 |
| glm-4-9b-chat           | 77.14 | 21.27 | 30.16 | 60.63 | 74.92 | 52.82 |
| **SeaLLMs-v3-7B-Chat**  | 88.89 | 60.00 | 73.33 | 83.81 | 92.70  | **79.75** |


## Acknowledgement to Our Linguists
We would like to express our special thanks to our professional and native linguists, Tantong Champaiboon, Nguyen Ngoc Yen Nhi and Tara Devina Putri, who helped build, evaluate, and fact-check our sampled pretraining and SFT dataset as well as evaluating our models across different aspects, especially safety.


## Citation

If you find our project useful, we hope you would kindly star our repo and cite our work as follows: 
```
@article{damonlp2024seallm3,
  author = {Wenxuan Zhang*, Hou Pong Chan*, Yiran Zhao*, Mahani Aljunied*,
            Jianyu Wang*, Chaoqun Liu, Yue Deng, Zhiqiang Hu, Weiwen Xu,
            Yew Ken Chia, Xin Li, Lidong Bing},
  title = {SeaLLMs 3: Open Foundation and Chat Multilingual Large Language Models for Southeast Asian Languages},
  year = {2024},
  url = {https://arxiv.org/abs/2407.19672}
}
```
Corresponding Author: l.bing@alibaba-inc.com
# [Open LLM Leaderboard Evaluation Results](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
Detailed results can be found [here](https://huggingface.co/datasets/open-llm-leaderboard/details_SeaLLMs__SeaLLMs-v3-7B-Chat)

|      Metric       |Value|
|-------------------|----:|
|Avg.               |23.63|
|IFEval (0-Shot)    |43.77|
|BBH (3-Shot)       |33.80|
|MATH Lvl 5 (4-Shot)|15.11|
|GPQA (0-shot)      | 6.49|
|MuSR (0-shot)      |10.47|
|MMLU-PRO (5-shot)  |32.16|

