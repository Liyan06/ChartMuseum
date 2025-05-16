# ChartMuseum


## Overview

**ChartMuseum** is a chart question answering benchmark designed to evaluate reasoning capabilities of large vision-language models
(LVLMs) over real-world chart images. The benchmark consists of 1162 *(image, question, short answer)* tuples and exclusively targets at questions that requires non-trivial text and visual reasoning skills. The dataset is collectively annotated by a team of 13 researchers in computer science.


## Installation

To set up the environment for accessing the dataset and evaluating model performance, follow these steps:

```bash
# Create and activate a new conda environment
conda create -n chartmuseum python=3.11
conda activate chartmuseum

# Clone the repository
git clone https://github.com/username/chartmuseum.git
cd chartmuseum

# Install dependencies
pip install -r requirements.txt
```


## Benchmark Access

Our benchmark is available at Hugging Face 🤗. More benchmark details can be found [here](https://huggingface.co/datasets/lytang/ChartMuseum).


```python
from datasets import load_dataset
dataset = load_dataset("lytang/ChartMuseum")
```

The benchmark contains the following fields:
|Field| Description |
|--|--|
|image| an image where the question is based on|
|question| a question on an image|
|answer| an answer to a question|
|reasoning_type| the reasoning skill that is primarily required to answer the question - *text*, *visual/text*, *synthesis*, *visual*|
|source| the website where we collect the image |
|hash| a unique identifier for the example |

The question answering prompt we used for all models is included in [prompt.py](prompt.py).

## Output Evaluation Instruction

Once your model predictions on the benchmark are ready, we provide an evaluation script to compute the accuracy of model answers, which is very simple to run. We use `gpt-4.1-mini-2025-04-14` checkpoint from OpenAI as the LLM-as-a-Judge model for our benchmark. Make sure to set up your OpenAI API key in your environment variables.


```python
export OPENAI_API_KEY=your_api_key_here
python evaluate.py --prediction_path /path/to/predictions.json --split dev/test
```

Optionally, you can specify `--save_dir /path/to/save_dir` to save the evaluation results. The evaluation script will extract the short answers from predictions and compare them with the ground truth answers. The evaluation script will output:
* The cost and time spent on the evaluation. The estimated cost is \$0.03 and \$0.16 on dev and test set, respectively. The evaluation typically takes around 5s for the dev set and 12s for the test set.
* The accuracy of the model on the dev/test set.


The prediction file should contain a list of strings, where each string correpsonds to an answer of a question in the dataset. The order of the answers should match the order of the questions in the dataset. Note that we require each string to contain an answer wrapped in the `<answer></answer>` tags. As our evaluation script will automatically extract the answer from the string.

```
[
    "...<answer>predicted short answer 1</answer>...",
    "...<answer>predicted short answer 2</answer>...",
    ...
]
```

### Evaluation Demo

To demonstrate the evaluation process, we'll use outputs from Claude-3.7-Sonnet on the dev set. The example below shows how to run the evaluation with our provided [example_outputs](example_outputs) (both formatting styles are supported):

```python
export OPENAI_API_KEY=your_api_key_here
python evaluate.py \
    --prediction_path example_outputs/claude-3-7-sonnet-20250219-dev-full-output.json \
    --split dev
```

<details>
<summary>Expected Output (shortened)</summary>

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • Time Elapsed 0:00:05 • Time Remaining 0:00:00                                     
Requests: Total: 162 • Cached: 0✓ • Success: 162✓ • Failed: 0✗ • In Progress: 0⋯ • Req/min: 1703.1 • Res/min: 1703.1                                                    
                Final Curator Statistics                
╭────────────────────────────┬─────────────────────────╮
│ Section/Metric             │ Value                   │
├────────────────────────────┼─────────────────────────┤
│ Model                      │                         │
│ Name                       │ gpt-4.1-mini-2025-04-14 │
│ Rate Limit (RPM)           │ 12000                   │
│ Rate Limit (TPM)           │ 4000000                 │
│ Requests                   │                         │
│ Total Processed            │ 162                     │
│ Successful                 │ 162                     │
│ Failed                     │ 0                       │
│ Tokens                     │                         │
│ Total Tokens Used          │ 0                       │
│ Total Input Tokens         │ 0                       │
│ Total Output Tokens        │ 0                       │
│ Average Tokens per Request │ 0                       │
│ Average Input Tokens       │ 0                       │
│ Average Output Tokens      │ 0                       │
│ Costs                      │                         │
│ Total Cost                 │ $0.027                  │
│ Average Cost per Request   │ $0.000                  │
│ Input Cost per 1M Tokens   │ $0.400                  │
│ Output Cost per 1M Tokens  │ $1.600                  │
│ Performance                │                         │
│ Total Time                 │ 5.71s                   │
│ Average Time per Request   │ 0.04s                   │
│ Requests per Minute        │ 1700.9                  │
│ Responses per Minute       │ 1700.9                  │
│ Max Concurrent Requests    │ 35                      │
│ Input Tokens per Minute    │ 0.0                     │
│ Output Tokens per Minute   │ 0.0                     │
╰────────────────────────────┴─────────────────────────╯
Final Accuracy: 0.6296
```
</details>

The inference time with our evaluation script takes 5s and 12s for the dev and test set, respectively.


## License

Our benchmark is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.en). Copyright of all included charts is retained by their original authors and sources.