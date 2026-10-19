import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

def main():
    dataset = load_dataset("tomaarsen/conll2003")

    output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(exist_ok=True)

    for split_name, split in dataset.items():
        ner_names = split.features["ner_tags"].feature.names
        pos_names = split.features["pos_tags"].feature.names

        entity_counts = Counter()
        token_count = 0
        outside_count = 0

        output_path = output_dir / f"{split_name}.jsonl"

        with output_path.open("w", encoding="utf-8") as file:
            for index, example in enumerate(split):
                tokens = example["tokens"]
                ner_tags = [
                    ner_names[tag_id] for tag_id in example["ner_tags"]
                ]
                pos_tags = [
                    pos_names[tag_id] for tag_id in example["pos_tags"]
                ]

                if not tokens or not (
                    len(tokens) == len(ner_tags) == len(pos_tags)
                ):
                    raise ValueError(
                        f"Invalid token/label lengths in "
                        f"{split_name}, sentence {index}"
                    )

                previous_tag = "O"

                for tag in ner_tags:
                    if tag.startswith("I-"):
                        entity_type = tag[2:]

                        if previous_tag not in (
                            f"B-{entity_type}",
                            f"I-{entity_type}",
                        ):
                            raise ValueError(
                                f"Invalid BIO2 sequence in "
                                f"{split_name}, sentence {index}: "
                                f"{previous_tag} followed by {tag}"
                            )

                    previous_tag = tag

                token_count += len(tokens)

                for tag in ner_tags:
                    if tag == "O":
                        outside_count += 1
                    elif tag.startswith("B-"):
                        entity_counts[tag[2:]] += 1

                record = {
                    "id": f"{split_name}-{index}",
                    "document_id": example["document_id"],
                    "sentence_id": example["sentence_id"],
                    "tokens": tokens,
                    "pos_tags": pos_tags,
                    "ner_tags": ner_tags,
                }

                file.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Display statistics
        print(f"\n{split_name.upper()}")
        print(f"  Sentences: {len(split):,}")
        print(f"  Tokens: {token_count:,}")

        for entity_type in ["PER", "ORG", "LOC", "MISC"]:
            print(
                f"  {entity_type} mentions: "
                f"{entity_counts[entity_type]:,}"
            )

        outside_percentage = (
            100 * outside_count / token_count if token_count else 0
        )
        print(f"  O tokens: {outside_percentage:.2f}%")
        print(f"  Saved: {output_path}")


    print("\nData preparation complete.")


if __name__ == "__main__":
    main()