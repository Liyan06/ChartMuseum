QA_PROMPT = """Please answer the question using the chart image.

Question: [QUESTION]

Please first generate your reasoning process and then provide the user with the answer. Use the following format:

<think> 
... your thinking process here ... 
</think> 
<answer> 
... your final answer (entity(s) or number) ...
</answer>"""