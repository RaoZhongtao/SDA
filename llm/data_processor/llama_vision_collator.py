# here put the import lib
from dataclasses import dataclass
from typing import Any, List, Dict, Sequence, Tuple
import torch
import transformers
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

@dataclass
class LlamaVisionTrainCollator(object):
    """Collate examples for supervised fine-tuning."""

    processor: transformers.ProcessorMixin
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batched_sample = {}
        pad_token_id = self.processor.tokenizer.pad_token_id
        # Prepare lists to store values for each key
        image_instance_input_ids_list = []
        image_instance_attention_mask_list = []
        image_instance_cross_attention_mask_list = []
        image_instance_aspect_ratio_ids_list = []
        image_instance_aspect_ratio_mask_list = []
        image_instance_pixel_values_list = []
        image_instance_moe_task_ids_list = []

        text_instance_input_ids_list = []
        text_instance_attention_mask_list = []
        text_instance_moe_task_ids_list = []


        item_ids = torch.tensor([instance["item_id"].item() for instance in instances])
        
        for instance in instances:
            sample = instance['contrastive_sample_input']
            sample_first = sample[0]
            sample_second = sample[1]
            image_instance_input_ids_list.extend(sample_first['input_ids'].tolist())  # Flatten (2, N) -> (2 * N,)
            image_instance_attention_mask_list.extend(sample_first['attention_mask'].tolist())
            if 'pixel_values' in sample_first:
                image_instance_cross_attention_mask_list.extend(sample_first['cross_attention_mask'].tolist())
                image_instance_aspect_ratio_ids_list.append(sample_first['aspect_ratio_ids'])#.tolist())
                image_instance_aspect_ratio_mask_list.append(sample_first['aspect_ratio_mask'])#.tolist())
                image_instance_pixel_values_list.append(sample_first['pixel_values'])  # (1, X, Y) -> Keep shape
                # image_instance_pixel_values_list.append(torch.tensor(sample_first['pixel_values']))

            text_instance_input_ids_list.extend(sample_second['input_ids'].tolist())
            text_instance_attention_mask_list.extend(sample_second['attention_mask'].tolist())

            if 'moe_task_ids' in instance:
                image_instance_moe_task_ids_list.append(instance['moe_task_ids'][0])
                text_instance_moe_task_ids_list.append(instance['moe_task_ids'][1])
                
            # print(f"sample_first['input_ids'].shape {sample_first['input_ids'].shape}")
            # print(f"sample_first['attention_mask'].shape {sample_first['attention_mask'].shape}")
            # print(f"sample_first['pixel_values'].shape {sample_first['pixel_values'].shape}")
            # print(f"sample_first['aspect_ratio_ids'].shape {sample_first['aspect_ratio_ids'].shape}")
            # print(f"sample_first['aspect_ratio_mask'].shape {sample_first['aspect_ratio_mask'].shape}")
            # print(f"sample_first['cross_attention_mask'].shape {sample_first['cross_attention_mask'].shape}")
            # print(f"sample_second['input_ids'].shape {sample_second['input_ids'].shape}")
            # print(f"sample_second['attention_mask'].shape {sample_second['attention_mask'].shape}")
            

        # cat_pixel_value = torch.cat(image_instance_pixel_values_list, dim=0)
        # print(f"cat_pixel_value shape: {cat_pixel_value.shape}")
        # pad_cross_attn_mask = pad_sequence(image_instance_cross_attention_mask_list, batch_first=True, padding_value=False, padding_side=self.processor.tokenizer.padding_side)
        # print(f"pad_cross_attn_mask shape: {pad_cross_attn_mask.shape}")
        # print(f"image_instance_input_ids_list: {image_instance_input_ids_list}")
        # print(f"image_instance_attention_mask_list: {image_instance_attention_mask_list}")
        # print(f"image_instance_cross_attention_mask_list: {image_instance_cross_attention_mask_list}")
        # print(f"image_instance_aspect_ratio_ids_list: {image_instance_aspect_ratio_ids_list}")
        # print(f"image_instance_aspect_ratio_mask_list: {image_instance_aspect_ratio_mask_list}")
        # print(f"image_instance_pixel_values_list: {image_instance_pixel_values_list}")
        # print(f"text_instance_input_ids_list: {text_instance_input_ids_list}")
        # print(f"text_instance_attention_mask_list: {text_instance_attention_mask_list}")
        
        # Convert lists to tensors before padding
        image_instance_input_ids_tensor = [torch.tensor(seq) for seq in image_instance_input_ids_list]
        image_instance_attention_mask_tensor = [torch.tensor(seq) for seq in image_instance_attention_mask_list]
        image_instance_cross_attention_mask_list_tensor = [torch.tensor(seq) for seq in image_instance_cross_attention_mask_list]
        # image_instance_aspect_ratio_ids_list_tensor = [torch.clone(seq) for seq in image_instance_aspect_ratio_ids_list]
        # image_instance_aspect_ratio_mask_list_tensor = [torch.clone(seq) for seq in image_instance_aspect_ratio_mask_list]
        
        text_instance_input_ids_tensor = [torch.tensor(seq) for seq in text_instance_input_ids_list]
        text_instance_attention_mask_tensor = [torch.tensor(seq) for seq in text_instance_attention_mask_list]
        
        batched_sample['item_ids'] = item_ids
    
        batched_sample['samples_first'] = {
            'input_ids': pad_sequence(image_instance_input_ids_tensor, batch_first=True, padding_value=pad_token_id, padding_side=self.processor.tokenizer.padding_side),
            'attention_mask': pad_sequence(image_instance_attention_mask_tensor, batch_first=True, padding_value=False, padding_side=self.processor.tokenizer.padding_side),
            'cross_attention_mask': pad_sequence(image_instance_cross_attention_mask_list_tensor, batch_first=True, padding_value=False, padding_side=self.processor.tokenizer.padding_side) if image_instance_cross_attention_mask_list_tensor else None,
            'aspect_ratio_ids': torch.cat(image_instance_aspect_ratio_ids_list, dim=0) if image_instance_aspect_ratio_ids_list else None,
            'aspect_ratio_mask': torch.cat(image_instance_aspect_ratio_mask_list, dim=0) if image_instance_aspect_ratio_mask_list else None,
            'pixel_values': torch.cat(image_instance_pixel_values_list, dim=0) if image_instance_aspect_ratio_mask_list else None,  # Shape (N, C, H, W)
            'moe_task_ids': torch.stack(image_instance_moe_task_ids_list, dim=0) if image_instance_moe_task_ids_list else None
        }
        
        # print(f"batched_sample['samples_first']['input_ids'].shape: {batched_sample['samples_first']['input_ids'].shape}")
        # print(f"batched_sample['samples_first']['attention_mask'].shape: {batched_sample['samples_first']['attention_mask'].shape}")
        # print(f"batched_sample['samples_first']['cross_attention_mask'].shape: {batched_sample['samples_first']['cross_attention_mask'].shape}")
        # print(f"batched_sample['samples_first']['aspect_ratio_ids'].shape: {batched_sample['samples_first']['aspect_ratio_ids'].shape}")
        # print(f"batched_sample['samples_first']['aspect_ratio_mask'].shape: {batched_sample['samples_first']['aspect_ratio_mask'].shape}")
        # print(f"batched_sample['samples_first']['pixel_values'].shape: {batched_sample['samples_first']['pixel_values'].shape}")
        # print(f"batched_sample['samples_first']['moe_task_ids'].shape: {batched_sample['samples_first']['moe_task_ids'].shape if batched_sample['samples_first']['moe_task_ids'] is not None else None}")
        
        
        batched_sample['samples_second'] = {
            'input_ids': pad_sequence(text_instance_input_ids_tensor, batch_first=True, padding_value=pad_token_id, padding_side=self.processor.tokenizer.padding_side),
            'attention_mask': pad_sequence(text_instance_attention_mask_tensor, batch_first=True, padding_value=False, padding_side=self.processor.tokenizer.padding_side),
            'moe_task_ids': torch.stack(text_instance_moe_task_ids_list, dim=0) if text_instance_moe_task_ids_list else None
        }

        return batched_sample 
    
        



@dataclass
class LlamaVisionEvalCollator(LlamaVisionTrainCollator):

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
        # Prepare lists to store values for each key
        image_instance_input_ids_list = []
        image_instance_attention_mask_list = []
        image_instance_cross_attention_mask_list = []
        image_instance_aspect_ratio_ids_list = []
        image_instance_aspect_ratio_mask_list = []
        image_instance_pixel_values_list = []
        image_instance_moe_task_ids_list = []

        text_instance_input_ids_list = []
        text_instance_attention_mask_list = []
        text_instance_moe_task_ids_list = []


        item_ids = torch.tensor([instance["item_id"].item() for instance in instances])
        is_double_samples = False
        for instance in instances:
            sample = instance['sample_input']
            sample_first = sample[0]
            
            image_instance_input_ids_list.extend(sample_first['input_ids'].tolist())  # Flatten (2, N) -> (2 * N,)
            image_instance_attention_mask_list.extend(sample_first['attention_mask'].tolist())
            
            if 'pixel_values' in sample_first:
                image_instance_cross_attention_mask_list.extend(sample_first['cross_attention_mask'].tolist())
                image_instance_aspect_ratio_ids_list.append(sample_first['aspect_ratio_ids'].tolist())
                image_instance_aspect_ratio_mask_list.append(sample_first['aspect_ratio_mask'].tolist())
                image_instance_pixel_values_list.append(sample_first['pixel_values'])  # (1, X, Y) -> Keep shape

            if len(sample) == 2:
                is_double_samples = True
                sample_second = sample[1]
                text_instance_input_ids_list.extend(sample_second['input_ids'].tolist())
                text_instance_attention_mask_list.extend(sample_second['attention_mask'].tolist())

            if 'moe_task_ids' in instance:
                image_instance_moe_task_ids_list.append(instance['moe_task_ids'][0])
                text_instance_moe_task_ids_list.append(instance['moe_task_ids'][1])

        # Convert lists to tensors before padding
        image_instance_input_ids_tensor = [torch.tensor(seq) for seq in image_instance_input_ids_list]
        image_instance_attention_mask_tensor = [torch.tensor(seq) for seq in image_instance_attention_mask_list]
        image_instance_cross_attention_mask_list_tensor = [torch.tensor(seq) for seq in image_instance_cross_attention_mask_list]
        image_instance_aspect_ratio_ids_list_tensor = [torch.tensor(seq) for seq in image_instance_aspect_ratio_ids_list]
        image_instance_aspect_ratio_mask_list_tensor = [torch.tensor(seq) for seq in image_instance_aspect_ratio_mask_list]
        
        text_instance_input_ids_tensor = [torch.tensor(seq) for seq in text_instance_input_ids_list]
        text_instance_attention_mask_tensor = [torch.tensor(seq) for seq in text_instance_attention_mask_list]
        
        batched_sample['item_ids'] = item_ids
    
        batched_sample['samples_first'] = {
            'input_ids': pad_sequence(image_instance_input_ids_tensor, batch_first=True, padding_value=pad_token_id, padding_side=self.processor.tokenizer.padding_side),
            'attention_mask': pad_sequence(image_instance_attention_mask_tensor, batch_first=True, padding_value=False, padding_side=self.processor.tokenizer.padding_side),
            'cross_attention_mask': pad_sequence(image_instance_cross_attention_mask_list_tensor, batch_first=True, padding_value=False, padding_side=self.processor.tokenizer.padding_side) if image_instance_cross_attention_mask_list_tensor else None,
            'aspect_ratio_ids': torch.cat(image_instance_aspect_ratio_ids_list_tensor, dim=0) if image_instance_aspect_ratio_ids_list_tensor else None,
            'aspect_ratio_mask': torch.cat(image_instance_aspect_ratio_mask_list_tensor, dim=0) if image_instance_aspect_ratio_mask_list_tensor else None,
            'pixel_values': torch.cat(image_instance_pixel_values_list, dim=0) if image_instance_pixel_values_list else None,  # Shape (N, C, H, W)
            'moe_task_ids': torch.stack(image_instance_moe_task_ids_list, dim=0) if image_instance_moe_task_ids_list else None
        }
        if is_double_samples:
            batched_sample['samples_second'] = {
                'input_ids': pad_sequence(text_instance_input_ids_tensor, batch_first=True, padding_value=pad_token_id, padding_side=self.processor.tokenizer.padding_side),
                'attention_mask': pad_sequence(text_instance_attention_mask_tensor, batch_first=True, padding_value=False, padding_side=self.processor.tokenizer.padding_side),
                'moe_task_ids': torch.stack(text_instance_moe_task_ids_list, dim=0) if text_instance_moe_task_ids_list else None
            }

        return batched_sample 


    


