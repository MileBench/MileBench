<!--
**MileBench/MileBench** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->


# MileBench 🛣️

A Multi-modal Long-context Benchmark for MLLMs

<center>

![Python 3.10+](https://img.shields.io/badge/Python-3.10-lightblue) ![Pytorch 2.1.1](https://img.shields.io/badge/PyTorch-2.1-lightblue) ![transformers](https://img.shields.io/badge/transformers-4.37.0.dev0%2B-lightblue) ![accelerate](https://img.shields.io/badge/accelerate-0.28.0-lightblue)
</center>

## 🌈 Update

* **[2024.4.15]** 🎉🎉🎉 MileBench is public!🎉🎉🎉


## ℹ️ How to evaluate

### Environment Setup

<details><summary>Click to expand</summary>
   
Install required packages:
```bash
pip install -r requirements.txt
```
Update `transformers` (we used `4.37.0.dev0`):
```bash
pip install git+https://github.com/huggingface/transformers
```
</details>

### Modify model configuration file

<details><summary>Click to expand</summary>

In `configs/model_configs.yaml`:

```yaml
# Add a new model "my_model"
my_model:
    model_name: "my_model"
    model_dir: "path/to/full/model" # HuggingFace model weights
    cfg_path: "path/to/full/model_config"   # can be none
    gen_kwargs:
        max_new_tokens: 512
        min_new_tokens: 1
        do_sample: false
```
</details>

### Modify model worker

<details><summary>Click to expand</summary>

In `workers/model_workers.py`:
1. add a new model class

```python
class MyModel(BaseWorker):

    def init_components(self, config) -> None:
        # init the model components

    def forward(self, questions: list[str], image_paths: list[list], device, gen_kwargs) -> list[str]:
        # Prepare images and text for generate function

```

2. for github packages of different VLM models, we recommand you to save them to `./packages` directory. Then you don't need to install pip packages in your env.
</details>


### Modify utils.py

<details><summary>Click to expand</summary>

In `utils.py`:
import your model

```python
from workers.model_workers import MyModel   # modify here

name2worker = {
    "my_model": MyModel,  # modify here
    }
```
</details>

### Generate response

<details><summary>Click to expand</summary>
Set GPU num in `/configs/accelerate_configs.yaml`:

```yaml
num_processes: GPU_NUM    # modify here
```

Modify `eval.sh`:

```bash

gpu_num=GPU_NUM  # modify here

for model in my_model; do  # modify here
    for dataset_name in dataset_name; do  # modify here
...
```

and run:

```bash
source eval.sh
```

</details>


### Run evaluation

<details><summary>Click to expand</summary>

run:
```bash
python score.py \
    --result-dir outputs \
    --models my_model  # models to eval
# Result saved to outputs/result.csv
```

</details>
