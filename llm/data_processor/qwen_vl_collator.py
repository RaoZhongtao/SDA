# here put the import lib
from dataclasses import dataclass
from typing import Any, List, Dict, Sequence, Tuple
import torch
import transformers
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

@dataclass
class QwenVLTrainCollator(object):
    """Collate examples for supervised fine-tuning."""

    processor: transformers.ProcessorMixin
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        batched_sample = {}
        
        # Extract tokenizer pad token ID
        pad_token_id = self.processor.tokenizer.pad_token_id
        
        # Prepare lists to store values for each key
        input_ids_list = []
        attention_mask_list = []
        pixel_values_list = []
        image_grid_thw_list = []
        moe_task_ids_list = []
        item_ids = torch.tensor([instance["item_id"].item() for instance in instances])
        for instance in instances:
            sample = instance['contrastive_sample_input']
            input_ids_list.extend(sample['input_ids'].tolist())  # Flatten (2, N) -> (2 * N,)
            attention_mask_list.extend(sample['attention_mask'].tolist())
            if 'pixel_values' in sample:
                pixel_values_list.append(sample['pixel_values'])  # (1, X, Y) -> Keep shape
                image_grid_thw_list.append(sample['image_grid_thw'])  # (1, 3) -> Keep shape    
            if 'moe_task_ids' in instance:
                 moe_task_ids_list.append(instance['moe_task_ids'])
            
        # Convert lists to tensors before padding
        input_ids_tensor = [torch.tensor(seq) for seq in input_ids_list]
        attention_mask_tensor = [torch.tensor(seq) for seq in attention_mask_list]
        
        # Pad input_ids and attention_mask to max length in batch
        batched_sample['input_ids'] = pad_sequence(input_ids_tensor, batch_first=True, padding_value=pad_token_id, padding_side=self.processor.tokenizer.padding_side)
        batched_sample['attention_mask'] = pad_sequence(attention_mask_tensor, batch_first=True, padding_value=False, padding_side=self.processor.tokenizer.padding_side)
        
        # Stack pixel_values and image_grid_thw without padding
        if pixel_values_list:
            batched_sample['pixel_values'] = torch.cat(pixel_values_list, dim=0)  # Shape (32, X, Y)
            batched_sample['image_grid_thw'] = torch.cat(image_grid_thw_list, dim=0)  # Shape (32, 3)
        if moe_task_ids_list:
            batched_sample['moe_task_ids'] = torch.cat(moe_task_ids_list, dim=0)  # Shape (32, 2)
        batched_sample['item_ids'] = item_ids
        
        # print(f"input_ids shape: {batched_sample['input_ids'].shape}")
        # print(f"attention_mask shape: {batched_sample['attention_mask'].shape}")
        # print(f"pixel_values shape: {batched_sample['pixel_values'].shape}")
        # print(f"image_grid_thw shape: {batched_sample['image_grid_thw'].shape}")
        # print(f"batched_sample: {batched_sample}")
        
        return batched_sample 
    
        



@dataclass
class QwenVLEvalCollator(QwenVLTrainCollator):

    processor: transformers.ProcessorMixin

    r"""
    Data collator for pairwise data.
    """

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        batched_sample = {}
        pad_token_id = self.processor.tokenizer.pad_token_id
        input_ids_list = []
        attention_mask_list = []
        pixel_values_list = []
        image_grid_thw_list = []
        item_ids = torch.tensor([instance["item_id"].item() for instance in instances])
        moe_task_ids_list = []
        for instance in instances:
            sample = instance['sample_input']
            input_ids_list.extend(sample['input_ids'].tolist())
            attention_mask_list.extend(sample['attention_mask'].tolist())
            if 'pixel_values' in sample:
                pixel_values_list.append(sample['pixel_values'])
                image_grid_thw_list.append(sample['image_grid_thw'])
            if 'moe_task_ids' in instance:
                 moe_task_ids_list.append(instance['moe_task_ids'])
            
        # Convert lists to tensors before padding
        input_ids_tensor = [torch.tensor(seq) for seq in input_ids_list]
        attention_mask_tensor = [torch.tensor(seq) for seq in attention_mask_list]
        
        # Pad input_ids and attention_mask to max length in batch
        batched_sample['input_ids'] = pad_sequence(input_ids_tensor, batch_first=True, padding_value=pad_token_id, padding_side=self.processor.tokenizer.padding_side)
        batched_sample['attention_mask'] = pad_sequence(attention_mask_tensor, batch_first=True, padding_value=False, padding_side=self.processor.tokenizer.padding_side)
        
        # Stack pixel_values and image_grid_thw without padding
        if pixel_values_list:
            batched_sample['pixel_values'] = torch.cat(pixel_values_list, dim=0)  # Shape (32, X, Y)
            batched_sample['image_grid_thw'] = torch.cat(image_grid_thw_list, dim=0)  # Shape (32, 3)
            
        if moe_task_ids_list:
            batched_sample['moe_task_ids'] = torch.cat(moe_task_ids_list, dim=0)  # Shape (32, 2)
        batched_sample['item_ids'] = item_ids

        return batched_sample


    

