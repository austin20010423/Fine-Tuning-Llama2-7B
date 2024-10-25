import datasets
import pandas as pd

dataset = datasets.load_dataset("openai/gsm8k", 'main')
dataset = dataset["train"]
dataset = dataset.select(range(10))

print(pd.DataFrame(dataset))