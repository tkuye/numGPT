import argparse
import requests
import os
import json

VOCAB_FILE_LOCATION = "https://huggingface.co/gpt2/resolve/main/vocab.json"
MERGES_FILE_LOCATION = "https://huggingface.co/gpt2/resolve/main/merges.txt"

def main():
    """
    Simple function to download the vocabulary and merges file to a default folder
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', default='vocab.json')
    parser.add_argument('--merges_file', default='merges.txt')
    parser.add_argument('--output_dir', default='vocab')
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, 0o777)
    
    print("Fetching vocabulary and merges from huggingface.co")
    vocab_data = requests.get(VOCAB_FILE_LOCATION).json()
    with open(os.path.join(args.output_dir, args.vocab_file), 'w') as f:
        json.dump(vocab_data, f)

    merges_data = requests.get(MERGES_FILE_LOCATION).text
    with open(os.path.join(args.output_dir, args.merges_file), 'w') as f:
        f.write(merges_data)
    
    print("Completed vocabulary and merges download.")


if __name__ == '__main__':
    main()


    