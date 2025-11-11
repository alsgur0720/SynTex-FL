# SynTeX-FL: Cross-Modal Text Transfer in Federated Learning for Medical Visual Question Answering


## Environment


- Install Package: Create conda environment

```Shell
conda create -n syntex=3.10 -y
conda activate syntex
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

Clients 1 and 2 consist of the X-ray dataset.
Clients 3 and 4 consist of the X-ray dataset.
Clients 5 and 6 consist of the X-ray dataset.




The commands are as follows.

```
# Training for AdaShield-FL in Federated Learning

sh training.sh


# Testing for AdaShield-FL in Federated Learning

python llava/eval/model_vqa.py --conv-mode mistral_instruct --model-path training_model --question-file ./data/eval/llava_med_eval_qa50_qa.jsonl --image-folder ./data/images --answers-file ./path/to/answer-file.jsonl --temperature 0.0

```

For detailed parameter adjustments, refer to training.sh in the experiments directory.
