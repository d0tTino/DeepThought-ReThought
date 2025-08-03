# Social Perception Training

This guide explains how to fine-tune the social perception classifier using
`train/social_perception_train.py`.

## Dataset

The script expects a dataset containing two fields: `text` and `label`.
Labels are encoded as integers:

```
0 -> flirtation
1 -> avoidance
2 -> manipulation
3 -> sarcasm
4 -> supportiveness
```

Any Hugging Face dataset or local JSON/CSV file with these columns can be
used. A small public example is available at
[`dtrt/social-cues`](https://huggingface.co/datasets/dtrt/social-cues).

## Running the script

Install dependencies first:

```bash
pip install -r requirements.txt
```

Launch training with:

```bash
python train/social_perception_train.py \
    --dataset-path dtrt/social-cues \
    --model-name distilbert-base-uncased \
    --epochs 3 \
    --batch-size 8
```

The fine-tuned model will be saved to `./social_perception_model` by
default. Adjust `--output-dir` to change the location.

## Enable the classifier

Point the application to the trained weights by setting
`SOCIAL_PERCEPTION_MODEL` to the model directory:

```bash
export SOCIAL_PERCEPTION_MODEL=$(pwd)/social_perception_model
```

When this variable is unset the system falls back to neutral perception
scores.
