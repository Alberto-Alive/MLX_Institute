import pandas as pd
import json

def save_captions_for_sentencepiece(input_csv, output_txt):
    """
    Extracts captions from all splits in the dataset CSV file and saves them to a text file, one caption per line.
    
    Parameters:
    - input_csv (str): Path to the input CSV file containing the dataset.
    - output_txt (str): Path to the output text file for SentencePiece training.
    
    Returns:
    - None
    """
    # Load the dataset
    df = pd.read_csv(input_csv)
    
    # Open the output file for writing
    with open(output_txt, 'w', encoding='utf-8') as f:
        # Iterate over each row in the dataset
        for _, row in df.iterrows():
            # Parse the `raw` field as JSON to get the list of captions
            captions = json.loads(row['raw'])
            
            # Write each caption to the output file, one per line
            for caption in captions:
                f.write(caption + '\n')
                
    print(f"Captions saved to {output_txt} for SentencePiece training.")

# Usage
input_csv = '..\data\/flickr_annotations_30k.csv'  # Replace with the path to your dataset CSV file
output_txt = '../data/captions_4_vocab.txt'  # Path to the output file for SentencePiece training
save_captions_for_sentencepiece(input_csv, output_txt)

