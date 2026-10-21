import json
import time

import numpy as np
from datasets import Dataset
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

#hyperparamters from proposal
MODEL_NAME = "distilbert-base-uncased"
EPOCHS=2
LEARNING_RATE = 2e-5
BATCH_SIZE =16


#reads one of the jsonl files from data.py, one sentence per line
def load_split(name):
    with open(f"data/{name}.jsonl", encoding="utf-8") as file:
        records=[]

        for line in file:
            records.append(json.loads(line))
    return records


train_data = load_split("train")
validation_data = load_split( "validation")

test_data = load_split("test")

# build the label list from the training tags
labels = []

for record in train_data:
    for tag in record["ner_tags"]:
        if tag not in labels:

            labels.append(tag)
labels.sort()

label2id = {}
id2label= {}

for i in range(len(labels)):
    label2id[labels[i]] = i
    id2label[i] = labels[i]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# turns words into distilbert tokens and lines the labels up with them
def tokenize_and_align(batch):
    # split each word into subword peices
    encoded = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True)

    all_labels = []
    for i, tags in enumerate(batch["ner_tags"]):
        aligned = []

        previous_word=None
        for word_id in encoded.word_ids(batch_index=i):
            # only the first piece of a word keeps its label
            if word_id is None or word_id == previous_word:
                aligned.append(-100)
            else:
                aligned.append(label2id[tags[word_id]])

            previous_word = word_id
        all_labels.append(aligned)


    encoded["labels"] = all_labels
    return encoded


# turns the sentences into model inputs like token ids + aligned labels
def to_dataset(records):
    dataset = Dataset.from_list(records)
    return dataset.map(
        tokenize_and_align, batched=True, remove_columns=dataset.column_names
    )

train_dataset = to_dataset(train_data)
validation_dataset=to_dataset(validation_data)
test_dataset= to_dataset(test_data)


#changes model output back into tag names so seqeval can read them
def to_tag_sequences(logits, label_ids):

    #turn model output back into one tag per word
    predictions= np.argmax(logits, axis=-1)
    true_tags = []
    pred_tags =[]
    for row in range(len(label_ids)):
        sentence_true = []
        sentence_pred=[]
        for col in range( len(label_ids[row])):

            if label_ids[row][col] ==-100:
                continue
            sentence_true.append(id2label[label_ids[row][col]])
            sentence_pred.append(id2label[predictions[row][col]])

        true_tags.append(sentence_true)
        pred_tags.append(sentence_pred)
    return true_tags, pred_tags


#  the trainer calls compute_metrics to get precision recall and f1
def compute_metrics(eval_pred):
    logits, label_ids= eval_pred

    true_tags, pred_tags =to_tag_sequences(logits, label_ids)

    return {

        "precision": precision_score(true_tags, pred_tags),
        "recall" : recall_score(true_tags, pred_tags),
        "f1": f1_score(true_tags, pred_tags),
    }


model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=len(labels), id2label=id2label, label2id=label2id
)

# settings for how the model gets trained
training_args = TrainingArguments(
    output_dir = "distilbert-output", 
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size =BATCH_SIZE,
    eval_strategy="epoch",  
    save_strategy ="no",  
    logging_steps = 100,  
    report_to="none", 
)

# the trainer runs the training loop for us
trainer = Trainer(
    model =model,
    args =training_args,
    train_dataset = train_dataset,
    eval_dataset = validation_dataset,
    data_collator = DataCollatorForTokenClassification(tokenizer),
    compute_metrics = compute_metrics, 
)

# train and check validation scores after each epcoh
start = time.time()
trainer.train()
train_time = time.time()-start

# run on the test set once
start= time.time()
output =trainer.predict(test_dataset)
inference_time = time.time() - start

print( "\ntest results")
print(f"precision: {output.metrics['test_precision']:.4f}")
print(f"recall: {output.metrics['test_recall']:.4f}")
print(f"f1: {output.metrics['test_f1']:.4f}")
print(f"training time: {train_time:.1f}s")
print(f"inference time: {inference_time:.1f}s")

# per entity breakdown (PER, ORG, LOC, MISC)
true_tags, pred_tags = to_tag_sequences(output.predictions, output.label_ids)
print("\nper entity results")
print(classification_report(true_tags, pred_tags, digits=4))

