import json
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter


BASE_DIR = Path(__file__).resolve().parent


def load_data(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def extract_entities(tags):
    entities = set()
    index = 0

    while index < len(tags):
        if not tags[index].startswith("B-"):
            index += 1
            continue

        entity_type = tags[index][2:]
        end = index + 1

        while end < len(tags) and tags[end] == f"I-{entity_type}":
            end += 1

        entities.add((index, end, entity_type))
        index = end

    return entities


def build_dictionary(train_data):
    entity_counts = defaultdict(Counter)

    for example in train_data:
        tokens = example["tokens"]

        for start, end, entity_type in sorted(
            extract_entities(example["ner_tags"])
        ):
            name = tuple(tokens[start:end])
            entity_counts[name][entity_type] += 1

    entity_dictionary = {}

    for name, counts in entity_counts.items():
        best_type = sorted(
            counts,
            key=lambda entity_type: (
                -counts[entity_type],
                entity_type,
            ),
        )[0]

        entity_dictionary[name] = best_type

    return entity_dictionary


def predict(tokens, entity_dictionary, max_entity_length):
    predictions = ["O"] * len(tokens)
    index = 0

    while index < len(tokens):
        max_length = min(max_entity_length, len(tokens) - index)
        matched = False

        for length in range(max_length, 0, -1):
            name = tuple(tokens[index:index + length])
            entity_type = entity_dictionary.get(name)

            if entity_type is None:
                continue

            predictions[index] = f"B-{entity_type}"

            for offset in range(1, length):
                predictions[index + offset] = f"I-{entity_type}"

            index += length
            matched = True
            break

        if not matched:
            index += 1

    return predictions


def calculate_scores(gold, predicted):
    correct = len(gold & predicted)

    precision = correct / len(predicted) if predicted else 0.0
    recall = correct / len(gold) if gold else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return precision, recall, f1


# Load the prepared data.
train_data = load_data(BASE_DIR / "data" / "train.jsonl")
validation_data = load_data(BASE_DIR / "data" / "validation.jsonl")

# Build the dictionary and measure construction time.
build_start = perf_counter()

entity_dictionary = build_dictionary(train_data)
max_entity_length = max(
    (len(name) for name in entity_dictionary),
    default=0,
)

build_seconds = perf_counter() - build_start

# Predict validation labels and measure inference time.
prediction_start = perf_counter()

all_predictions = [
    predict(
        example["tokens"],
        entity_dictionary,
        max_entity_length,
    )
    for example in validation_data
]

prediction_seconds = perf_counter() - prediction_start

# Save predictions.
output_dir = BASE_DIR / "predictions"
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "rule_based_validation.jsonl"

with output_path.open("w", encoding="utf-8") as file:
    for example, predicted_tags in zip(validation_data, all_predictions):
        record = {
            "id": example["id"],
            "tokens": example["tokens"],
            "pred_tags": predicted_tags,
        }
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

gold_entities = set()
predicted_entities = set()

for sentence_index, (example, predicted_tags) in enumerate(
    zip(validation_data, all_predictions)
):
    for start, end, entity_type in extract_entities(example["ner_tags"]):
        gold_entities.add((sentence_index, start, end, entity_type))

    for start, end, entity_type in extract_entities(predicted_tags):
        predicted_entities.add((sentence_index, start, end, entity_type))

# Print timings and evaluation results.
print(f"Dictionary entries: {len(entity_dictionary):,}")
print(f"Dictionary build time: {build_seconds:.4f} seconds")
print(f"Validation prediction time: {prediction_seconds:.4f} seconds")
print(f"Saved predictions to: {output_path}")

print("\nValidation entity-level results")
print(f"{'Type':10} {'Precision':>10} {'Recall':>10} {'F1':>10}")

for entity_type in ["Overall", "PER", "ORG", "LOC", "MISC"]:
    if entity_type == "Overall":
        gold = gold_entities
        predicted = predicted_entities
    else:
        gold = {
            entity for entity in gold_entities
            if entity[3] == entity_type
        }
        predicted = {
            entity for entity in predicted_entities
            if entity[3] == entity_type
        }

    precision, recall, f1 = calculate_scores(gold, predicted)

    print(
        f"{entity_type:10} {precision:10.4f} "
        f"{recall:10.4f} {f1:10.4f}"
    )