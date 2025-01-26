import torch
import os
import pandas as pd
from PIL import Image

class ComDiffDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, train_transforms, tokenize_captions):
        self.image_dir = image_dir
        self.metadata = self.from_file(image_dir)
        self.column_names = ['pos_prompt', 'pos_img', 'neg_img']
        self.train_transforms = train_transforms
        self.tokenize_captions = tokenize_captions
        
    def from_file(self, image_dir):
        metadata = pd.DataFrame()
        prompt_types = next(os.walk(image_dir))[1]
        for prompt_type in prompt_types:
            sub_path = os.path.join(image_dir, prompt_type, 'metadata.jsonl')
            if os.path.exists(sub_path):
                print(f'Loading metadata from {sub_path}')
                metadata = pd.concat([metadata, pd.read_json(sub_path, lines=True)])
                
        if len(metadata) == 0:
            raise ValueError(f'No metadata found in {image_dir}')
        return metadata
        
    def __getitem__(self, idx):
        item = self.metadata.iloc[idx]
        pos_image = Image.open(item['pos_img'].replace("projects","repo")).convert("RGB")
        neg_image = Image.open(item['neg_img'].replace("projects","repo")).convert("RGB")
        
        example = {}
        example["pos_pixel_values"] = self.train_transforms(pos_image)
        example["neg_pixel_values"] = self.train_transforms(neg_image)
        example["pos_input_ids"] = self.tokenize_captions(item['pos_prompt'])
        
        return example

    def __len__(self):
        return len(self.metadata)
