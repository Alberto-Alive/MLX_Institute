import torch 
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import sentencepiece as spm
import ast
from tqdm import tqdm
import time

# Import the updated ENCODER and DECODER classes
from encoder import ENCODER as Encoder
from decoder import DECODER as Decoder

torch.manual_seed(29)

# Define the device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device set to:", device)

##########################################################################################
# LoadPatchCaptionTrainPairs Dataset Class
##########################################################################################

class LoadPatchCaptionTrainPairs(Dataset):
    def __init__(self, image_folder_path, captions_dict, tokenizer, patch_size=56, max_images=None, max_seq_length=30):
        self.image_folder_path = image_folder_path
        self.captions_dict = captions_dict
        self.tokenizer = tokenizer
        self.patch_size = patch_size
        self.max_images = max_images
        self.max_seq_length = max_seq_length
        self.pairs = self.generate_pairs()

    def generate_patches(self, image):
        image_width, image_height = image.size
        patches = []
        num_patches_x = image_width // self.patch_size
        num_patches_y = image_height // self.patch_size
        for y in range(num_patches_y):
            for x in range(num_patches_x):
                left = x * self.patch_size
                top = y * self.patch_size
                right = left + self.patch_size
                bottom = top + self.patch_size
                patch = image.crop((left, top, right, bottom))
                patch = transforms.ToTensor()(patch)
                patches.append(patch)
        patches = torch.stack(patches)
        return patches

    def generate_pairs(self):
        pairs = []
        count = 0
        for image_filename in os.listdir(self.image_folder_path):
            image_path = os.path.join(self.image_folder_path, image_filename)
            if image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = Image.open(image_path).convert('RGB')  # Ensure RGB
                patches = self.generate_patches(image)
                captions = self.captions_dict.get(image_filename, [])
                for caption in captions:
                    pairs.append((patches, caption))
                count += 1
                if self.max_images and count >= self.max_images:
                    break  # Stop once we reach the maximum number of images
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        patches, caption = self.pairs[idx]
        if isinstance(caption, str):
            caption = self.tokenizer.encode(caption)
            sos_id = self.tokenizer.piece_to_id('<s>')
            eos_id = self.tokenizer.piece_to_id('</s>')
            pad_id = self.tokenizer.pad_id()
            caption = [sos_id] + caption + [eos_id]
            if len(caption) > self.max_seq_length:
                caption = caption[:self.max_seq_length]
                caption[-1] = eos_id
            else:
                caption += [pad_id] * (self.max_seq_length - len(caption))
            caption = torch.tensor(caption, dtype=torch.long)
        return patches, caption

##########################################################################################
# Transformer Model
##########################################################################################

class TransformerModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(TransformerModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, target_sequence):
        batch_size, num_patches, C, H, W = images.size()
        images = images.view(batch_size, num_patches, -1)  # Flatten patches
        encoder_output = self.encoder(images)  # [batch_size, num_patches, embedding_dim]
        decoder_output = self.decoder(target_sequence, encoder_output)  # [batch_size, seq_length, vocab_size]
        return decoder_output

##########################################################################################
# Initialization and Training
##########################################################################################

def main():
    # Paths
    image_dir = './data/224by224'
    captions_file = './data/flickr_annotations_30k.csv'
    tokenizer_path = './models/caption_tokenizer_updated.model'
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    vocab_size = tokenizer.get_piece_size()
    print(f"Vocabulary Size: {vocab_size}")
    captions_df = pd.read_csv(captions_file)

    captions_dict = {}

    for _, row in captions_df.iterrows():
        image_filename = row['filename']
        try:
            captions = ast.literal_eval(row['raw'])
            if not isinstance(captions, list):
                captions = [captions]
        except Exception as e:
            print(f"Error parsing captions for image {image_filename}: {e}")
            continue

        if image_filename in captions_dict:
            captions_dict[image_filename].extend(captions)
        else:
            captions_dict[image_filename] = captions

    max_caption_seq_length = 30  # Adjust as needed
    patch_size = 56
    flickr_dataset = LoadPatchCaptionTrainPairs(
        image_folder_path=image_dir,
        captions_dict=captions_dict,
        tokenizer=tokenizer,
        patch_size=patch_size,  # Adjust as needed
        max_images=None,  # Or specify a limit
        max_seq_length=max_caption_seq_length
    )
    batch_size = 100  # Adjust based on your GPU memory
    def collate_fn(batch):
        images, captions = zip(*batch)
        # Stack images and captions
        images = torch.stack(images)  # [batch_size, num_patches, C, H, W]
        captions = torch.stack(captions)  # [batch_size, max_seq_length]
        return images, captions

    data_loader = DataLoader(
        flickr_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # Ensure all batches are full
        num_workers=0,  # Set to 0 for debugging
        collate_fn=collate_fn
    )
    decoder_embedding_dim = 128  # Adjust as needed
    encoder_embedding_dim = 128  # Should match the encoder's embedding_dim
    max_seq_length = 30  # As defined in dataset
    patch_colors = 3
    num_heads_encoder = 8  # Adjust as needed
    num_heads_decoder = 8  # Adjust as needed
    num_patches = (224 // patch_size) * (224 // patch_size) 
    encoder = Encoder(
        embedding_dim=encoder_embedding_dim,
        num_heads=num_heads_encoder,  # Adjust as needed
        num_patches=num_patches,
        patch_embs_C_H_W=patch_colors * patch_size * patch_size
    ).to(device)

    decoder = Decoder(
        vocab_size=vocab_size,
        decoder_embedding_dim=decoder_embedding_dim,
        encoder_embedding_dim=encoder_embedding_dim,
        num_heads=num_heads_decoder,  # Adjust as needed
        max_sequence_length=max_seq_length  # For captions
    ).to(device)
    model = TransformerModel(
        encoder,
        decoder
    ).to(device)
    def initialize_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
            nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
    def truncate_at_eos(prediction, eos_token_id):
        # Truncate the prediction at the first occurrence of <eos>
        if eos_token_id in prediction:
            prediction = prediction[:prediction.index(eos_token_id) + 1]
        return prediction

    model.apply(initialize_weights)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id(), label_smoothing=0.1)
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 2000  # Adjust as needed
    global_time = 0
    global_loss = 0
    for epoch in range(num_epochs):
        model.train()  # Set to training mode
        total_loss = 0
         # Synchronize CUDA and record the start time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        with tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100) as pbar:
            for batch_idx, (images, target_labels) in pbar:
                optimizer.zero_grad()
                images = images.to(device)          # [batch_size, num_patches, C, H, W]
                target_labels = target_labels.to(device)  # [batch_size, max_seq_length]
                input_sequences = target_labels[:, :-1]  # [batch_size, max_seq_length -1]
                target_outputs = target_labels[:, 1:]    # [batch_size, max_seq_length -1]
                output = model(images, input_sequences)  # [batch_size, max_seq_length -1, vocab_size]
                output = output.view(-1, vocab_size)  # [(batch_size * (max_seq_length -1)), vocab_size]
                target_outputs = target_outputs.reshape(-1)  # [(batch_size * (max_seq_length -1))]
                loss = criterion(output, target_outputs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({
                    "Batch": f"{batch_idx + 1}/{len(data_loader)}",
                    "Loss": f"{loss.item():.4f}"
                })
        # Synchronize CUDA and record the end time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()

        # Calculate epoch duration
        epoch_duration = end_time - start_time
        global_time += epoch_duration 
        # Calculate and display average loss per epoch
        avg_loss = total_loss / len(data_loader)
        global_loss += total_loss
        print(f"Epoch [{epoch + 1}/{num_epochs}] Completed - Average Loss: {avg_loss:.4f} - Time Taken: {epoch_duration:.2f} seconds")
        if (epoch + 1) % 100 == 0:
            checkpoint_filename = f"./models/epoch_{epoch+1}_loss_{avg_loss:.4f}.pth"
            torch.save(model.state_dict(), checkpoint_filename)
            print(f"Checkpoint saved: {checkpoint_filename}")
        model.eval()  # Switch to evaluation mode for inference
        with torch.no_grad():
            sample_images, sample_targets = next(iter(data_loader))
            sample_images = sample_images.to(device)
            sample_targets = sample_targets.to(device)
            sample_input_sequences = sample_targets[:, :-1]
            sample_output = model(sample_images, sample_input_sequences)
            sample_predictions = sample_output.argmax(-1)
            special_tokens = ['<s>', '</s>', '<pad>']
            special_ids = [tokenizer.piece_to_id(token) for token in special_tokens]
            for i in range(min(3, sample_images.size(0))):  # Display up to 3 examples
                pred_tokens = sample_predictions[i].tolist()
                truncated_tokens = truncate_at_eos(pred_tokens, tokenizer.piece_to_id('</s>'))
                decoded_pred_caption = tokenizer.decode(truncated_tokens)
                # filtered_pred_tokens = [token for token in pred_tokens if token not in special_ids]
                # decoded_pred_caption = tokenizer.decode(filtered_pred_tokens)
                pred_caption = decoded_pred_caption.replace('▁', ' ').strip()
                target_tokens = sample_targets[i, 1:].tolist()  # Skip <SOS> token
                filtered_target_tokens = [token for token in target_tokens if token not in special_ids]
                # Filter out special tokens
                pred_tokens = [t for t in truncated_tokens if t not in special_ids]
                target_tokens = [t for t in target_tokens if t not in special_ids]

                # Precision, Recall, and F1
                same_tokens = sum(1 for pred, target in zip(pred_tokens, target_tokens) if pred == target)
                precision = same_tokens / len(pred_tokens) if pred_tokens else 0
                recall = same_tokens / len(target_tokens) if target_tokens else 0
                f1_score = 2 * (precision * recall) / (precision + recall + 1e-8) if (precision + recall) > 0 else 0
                
                print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
                
                # BLEU Score
                bleu_score = sentence_bleu([target_tokens], pred_tokens)
                print(f"BLEU Score: {bleu_score:.4f}")
                decoded_target_caption = tokenizer.decode(filtered_target_tokens)
                target_caption = decoded_target_caption.replace('▁', ' ').strip()
                print(f"Sample {i + 1} - Predicted: {pred_caption}")
                print(f"Sample {i + 1} - Target:    {target_caption}\n")
        model.train()  # Switch back to training mode

    # Save the model
    print(f"Global Time: {global_time:.2f}")
    print(f"Global Loss: {global_loss:2f}")
    model_filename = f"./models/ps{patch_size}_bs{batch_size}_de{decoder_embedding_dim}_ee{encoder_embedding_dim}_nhd{num_heads_decoder}_nhd{num_heads_encoder}_msl{max_seq_length}_lr{learning_rate:.4f}_ep{num_epochs}_ls{loss:.4f}.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as {model_filename}")

if __name__ == "__main__":
    main()
