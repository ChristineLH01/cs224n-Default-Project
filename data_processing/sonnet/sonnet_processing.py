from datasets import load_dataset

dataset_name = "Lambent/shakespeare_sonnets_diffused"


def format_sonnets_txt(dataset, output_path, start_index=1):
    lines = []
    for i, example in enumerate(dataset):
        sonnet_num = str(start_index + i)
        sonnet_lines = [l for l in example['Variation Text'].strip().split('\n') if l.strip()]

        if len(sonnet_lines) >= 2:
            sonnet_lines[-2] = '  ' + sonnet_lines[-2].lstrip()
            sonnet_lines[-1] = '  ' + sonnet_lines[-1].lstrip()

        lines.append(sonnet_num)
        lines.append('')
        lines.append('\n'.join(sonnet_lines))
        lines.append('')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def merge_sonnet_files(base_path, additional_path, output_path):
    """Append the contents of additional_path to base_path and write to output_path."""
    with open(base_path, 'r', encoding='utf-8') as f:
        base_content = f.read().rstrip()

    with open(additional_path, 'r', encoding='utf-8') as f:
        additional_content = f.read().strip()

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(base_content + '\n\n' + additional_content + '\n')


if __name__ == "__main__":
    dataset = load_dataset(dataset_name, split="train")
    format_sonnets_txt(dataset, "data/diffused_sonnets.txt", start_index=132)
    merge_sonnet_files("data/sonnets.txt", "data/diffused_sonnets.txt", "data/sonnets_merged.txt")
