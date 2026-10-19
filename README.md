# Named Entity Recognition: Comparing Three Approaches

This CS4120 project compares a rule-based model, a Conditional Random Field (CRF), and DistilBERT on the CoNLL-2003 English Named Entity Recognition dataset.

## Setup

Install the data preparation dependency:

```bash
python -m pip install datasets==4.5.0
```

## Dataset

We use the existing training, validation, and test splits from:
https://huggingface.co/datasets/tomaarsen/conll2003

| Split | Sentences | Tokens |
| --- | ---: | ---: |
| Training | 14,041 | 203,621 |
| Validation | 3,250 | 51,362 |
| Test | 3,453 | 46,435 |

The dataset contains four entity types:

- PER: Person
- ORG: Organization
- LOC: Location
- MISC: Miscellaneous entity

This dataset version uses BIO2 labels:

- B-TYPE: Beginning of an entity
- I-TYPE: Continuation of an entity
- O: Outside any entity

## Data Preparation

Run:

```bash
python data.py
```
This script downloads the dataset and turns the numeric NER and POS labels into readable tags. It checks that every token lines up with its label and that the BIO2 sequences are valid.
Tokens, capitalization, punctuation, sentence order, and the train/validation/test splits all stay as they were.

When it finishes, you'll find three files in the `data` folder:

- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`

If you run the script again, it overwrites them.

## Output Format

Each JSONL file contains one sentence per line with these fields:

| Field | Description |
| --- | --- |
| id | Unique sentence ID within the prepared dataset |
| document_id | Original document identifier |
| sentence_id | Original sentence identifier |
| tokens | List of original words and punctuation |
| pos_tags | Part-of-speech tag for each token |
| ner_tags | Correct BIO2 entity label for each token |

Read a prepared file using:

```python
import json

with open("data/train.jsonl", encoding="utf-8") as file:
    train_data = [json.loads(line) for line in file]

print(train_data[0]["tokens"])
print(train_data[0]["ner_tags"])
```