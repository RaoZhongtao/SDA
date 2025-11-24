# here put the import lib
import numpy as np
from tqdm import tqdm
import copy
import random
from PIL import Image

from transformers import MllamaProcessor



class LlamaVisionTrainMask(object):
    
    def __init__(self, data_args, model_args, training_args, processor: MllamaProcessor) -> None:
    
        self.data_args = data_args
        self.model_args = model_args
        self.attributes_input_column = "attributes_input"
        self.image_instruction_column = "image_instruction_input"
        self.image_url_column = "image_url"
        self.processor = processor
        self.pad_token_id = self.processor.tokenizer.pad_token
        self.multimodal_embed = training_args.multimodal_embed
        self.enable_task_gate = training_args.enable_task_gate
        self.enable_moelora = training_args.enable_moelora



    def __call__(self, examples):
        if self.multimodal_embed:
            return self.multimodal_mask(examples)

        return self.text_mask(examples)
    
    def multimodal_mask(self, examples):
        if self.enable_moelora and self.enable_task_gate:
            model_inputs = {
                "contrastive_sample_input": [],
                "moe_task_ids": [],
            }
        else:
            model_inputs = {
                "contrastive_sample_input": [],
            }
        failed_images_count = 0
        for i in range(len(examples[self.attributes_input_column])):
            if examples[self.attributes_input_column][i]:
                attributes_sample = examples[self.attributes_input_column][i]
                image_sample_instruction = examples[self.image_instruction_column][i]
                image_file_name = examples[self.image_url_column][i]
                if image_file_name == "":
                    image_url = f"/hpc2hdd/home/zrao690/MultimodalEmb/data/noise.jpg"
                else:
                    image_url = f"/hpc2hdd/home/zrao690/MultimodalEmb/data/{self.data_args.dataset_name}/handled/image/{image_file_name}"

                image_sample_input_template = self.apply_message_template(image_sample_instruction, image_url)
                attributes_sample_input_template = self.apply_message_template(attributes_sample)
                # constrastive_sample_input_template = [image_sample_input_template, attributes_sample_input_template]
                try:
                    image_sample_input_ids = self.processor.apply_chat_template(image_sample_input_template, 
                                                                                add_generation_prompt = True, 
                                                                                tokenize=True, 
                                                                                return_dict=True, 
                                                                                return_tensors="pt",
                                                                                padding=True,
                                                                                truncation=False)
                    text_sample_input_ids = self.processor.apply_chat_template(attributes_sample_input_template, 
                                                                                add_generation_prompt = True, 
                                                                                tokenize=True, 
                                                                                return_dict=True, 
                                                                                return_tensors="pt",
                                                                                padding=True,
                                                                                truncation=False)
                except Exception as e:
                    failed_images_count += 1
                    print(f"image_sample_input_ids: {e} file_name: {image_file_name}")
                    image_url = f"/hpc2hdd/home/zrao690/MultimodalEmb/data/noise.jpg"
                    image_sample_input_template = self.apply_message_template(image_sample_instruction, image_url)
                    attributes_sample_input_template = self.apply_message_template(attributes_sample)
                    # constrastive_sample_input_template = [image_sample_input_template, attributes_sample_input_template]
                    image_sample_input_ids = self.processor.apply_chat_template(image_sample_input_template, 
                                                                                add_generation_prompt = True, 
                                                                                tokenize=True, 
                                                                                return_dict=True, 
                                                                                return_tensors="pt",
                                                                                padding=True,
                                                                                truncation=False)
                    text_sample_input_ids = self.processor.apply_chat_template(attributes_sample_input_template, 
                                                                                add_generation_prompt = True, 
                                                                                tokenize=True, 
                                                                                return_dict=True, 
                                                                                return_tensors="pt",
                                                                                padding=True,
                                                                                truncation=False)  
                if self.enable_moelora and self.enable_task_gate:
                    moe_task_ids = [0, 1]
                    model_inputs["moe_task_ids"].append(moe_task_ids)
                model_inputs["contrastive_sample_input"].append([image_sample_input_ids, text_sample_input_ids])
                # model_inputs["contrastive_sample_input"].append([
                #     {
                #         "input_ids": image_sample_input_ids["input_ids"].squeeze(0),
                #         "attention_mask": image_sample_input_ids["attention_mask"].squeeze(0),
                #         "cross_attention_mask": image_sample_input_ids["cross_attention_mask"].squeeze(0),
                #         "aspect_ratio_ids": image_sample_input_ids["aspect_ratio_ids"].squeeze(0),
                #         "aspect_ratio_mask": image_sample_input_ids["aspect_ratio_mask"].squeeze(0),
                #         "pixel_values": image_sample_input_ids["pixel_values"].squeeze(0),
                #     },
                #     {
                #         "input_ids": text_sample_input_ids["input_ids"].squeeze(0),
                #         "attention_mask": text_sample_input_ids["attention_mask"].squeeze(0),
                #     }
                # ])
        return model_inputs
    
    def text_mask(self, examples):
        model_inputs = {
            "contrastive_sample_input": [],
        }

        for i in range(len(examples[self.attributes_input_column])):
            if examples[self.attributes_input_column][i]:
                attributes_sample = examples[self.attributes_input_column][i]
                if self.data_args.dataset_name == "tmall":
                    sample_input_1, sample_input_2 = self.split_keywords_and_drop(attributes_sample, self.data_args.dropout_ratio)
                else:
                    sample_input_1, sample_input_2 = self.dropout_feature(attributes_sample, self.data_args.dropout_ratio)
                
                sample_input_1 = self.apply_message_template(sample_input_1)
                sample_input_2 = self.apply_message_template(sample_input_2)
                # constrastive_sample_input_template = [sample_input_1, sample_input_2]
                
                sample_input_ids_1 = self.processor.apply_chat_template(sample_input_1, 
                                                                            add_generation_prompt = True, 
                                                                            tokenize=True, 
                                                                            return_dict=True, 
                                                                            return_tensors="pt",
                                                                            padding=True,
                                                                            truncation=False)
                sample_input_ids_2 = self.processor.apply_chat_template(sample_input_2, 
                                                                            add_generation_prompt = True, 
                                                                            tokenize=True, 
                                                                            return_dict=True, 
                                                                            return_tensors="pt",
                                                                            padding=True,
                                                                            truncation=False)
                
                model_inputs["contrastive_sample_input"].append([sample_input_ids_1, sample_input_ids_2])
        return model_inputs
    
    def apply_message_template(self, text, image = None):
        if image is not None:
            INPUT_TEMPLATE = [
                {
                    "role":"user",
                    "content":[
                        {"type":"image", "image": f"{image}"},
                        {"type":"text", "text":f"{text}"}
                    ],
                }
            ]
        else:
            INPUT_TEMPLATE = [
                {
                    "role":"user",
                    "content":[
                        {"type":"text", "text":f"{text}"}
                    ],
                }
            ]
        return INPUT_TEMPLATE
    
    def dropout_feature(self, item_str, ratio):

        instruction = item_str.split(":")[0]
        feat_list = item_str.split(":")[1].split(";")

        # get two copies, shuffle the order, remove the last ratio_N
        feat_list_1 = copy.deepcopy(feat_list)
        feat_list_2 = copy.deepcopy(feat_list)
        random.shuffle(feat_list_1)
        random.shuffle(feat_list_2)

        dropout_N = int(len(feat_list) * ratio)
        for _ in range(dropout_N):  # dropout N times
            feat_list_1.pop()
            feat_list_2.pop()

        # assemble the feat list to item string
        item_str_1 = instruction + "\n"
        item_str_2 = instruction + "\n"
        if len(feat_list) > 1:
            for i in range(len(feat_list_1)):
                item_str_1 += (feat_list_1[i] + ";")
                item_str_2 += (feat_list_2[i] + ";")

        return item_str_1[:-1], item_str_2[:-1] # remove the last ";"

    def split_keywords_and_drop(self, item_str, ratio):
        """
        Split the item string into keywords and drop some of them based on the ratio.
        """
        instruction = item_str.split(":")[0]
        keywords = item_str.split(":")[1].split(" ")
        keywords = [keyword.strip() for keyword in keywords if keyword.strip() != ""]
        # get two copies, shuffle the order, remove the last ratio_N
        feat_list_1 = copy.deepcopy(keywords)
        feat_list_2 = copy.deepcopy(keywords)
        random.shuffle(feat_list_1)
        random.shuffle(feat_list_2)

        dropout_N = int(len(keywords) * ratio)
        for _ in range(dropout_N):
            feat_list_1.pop()
            feat_list_2.pop()
        
        item_str_1 = instruction + "\n"
        item_str_2 = instruction + "\n"
        if len(keywords) > 1:
            for i in range(len(feat_list_1)):
                item_str_1 += (feat_list_1[i] + ";")
                item_str_2 += (feat_list_2[i] + ";")    
        return item_str_1[:-1], item_str_2[:-1]

    
class LlamaVisionEvalMask(object):
    
    def __init__(self, data_args, model_args, training_args, processor: MllamaProcessor) -> None:
    
        self.data_args = data_args
        self.model_args = model_args
        self.attributes_input_column = "attributes_input"
        self.image_instruction_column = "image_instruction_input"
        self.image_url_column = "image_url"
        self.processor = processor
        self.pad_token_id = self.processor.tokenizer.pad_token
        self.multimodal_embed = training_args.multimodal_embed
        self.seperate_eval = training_args.seperate_eval
        self.image_only = model_args.image_only
        self.enable_task_gate = training_args.enable_task_gate
        self.enable_moelora = training_args.enable_moelora


    def __call__(self, examples):
        if self.multimodal_embed and self.seperate_eval:
            return self.multimodal_seperate_mask(examples)
        elif self.multimodal_embed:
            return self.multimodal_mask(examples)
        elif self.image_only:
            return self.image_mask(examples)
        return self.text_mask(examples)

    def multimodal_seperate_mask(self, examples):
        if self.enable_moelora and self.enable_task_gate:
            model_inputs = {
                "sample_input": [],
                "moe_task_ids": [],
            }
        else:
            model_inputs = {
                "sample_input": [],
            }
        failed_images_count = 0
        for i in range(len(examples[self.attributes_input_column])):
            if examples[self.attributes_input_column][i]:
                attributes_sample = examples[self.attributes_input_column][i]
                image_sample_instruction = examples[self.image_instruction_column][i]
                image_file_name = examples[self.image_url_column][i]
                if image_file_name == "":
                    image_url = f"/hpc2hdd/home/zrao690/MultimodalEmb/data/noise.jpg"
                else:
                    image_url = f"/hpc2hdd/home/zrao690/MultimodalEmb/data/{self.data_args.dataset_name}/handled/image/{image_file_name}"

                image_sample_input_template = self.apply_message_template(image_sample_instruction, image_url)
                attributes_sample_input_template = self.apply_message_template(attributes_sample)
                # constrastive_sample_input_template = [image_sample_input_template, attributes_sample_input_template]
                try:

                    image_sample_input_ids = self.processor.apply_chat_template(image_sample_input_template, 
                                                                                add_generation_prompt = True, 
                                                                                tokenize=True, 
                                                                                return_dict=True, 
                                                                                return_tensors="pt",
                                                                                padding=True,
                                                                                truncation=False)
                    text_sample_input_ids = self.processor.apply_chat_template(attributes_sample_input_template, 
                                                                                add_generation_prompt = True, 
                                                                                tokenize=True, 
                                                                                return_dict=True, 
                                                                                return_tensors="pt",
                                                                                padding=True,
                                                                                truncation=False)
                except Exception as e:
                    failed_images_count += 1
                    print(f"image_sample_input_ids: {e} file_name: {image_file_name}")
                    image_url = f"/hpc2hdd/home/zrao690/MultimodalEmb/data/noise.jpg"
                    image_sample_input_template = self.apply_message_template(image_sample_instruction, image_url)
                    attributes_sample_input_template = self.apply_message_template(attributes_sample)
                    # constrastive_sample_input_template = [image_sample_input_template, attributes_sample_input_template]
                    image_sample_input_ids = self.processor.apply_chat_template(image_sample_input_template, 
                                                                                add_generation_prompt = True, 
                                                                                tokenize=True, 
                                                                                return_dict=True, 
                                                                                return_tensors="pt",
                                                                                padding=True,
                                                                                truncation=False)
                    text_sample_input_ids = self.processor.apply_chat_template(attributes_sample_input_template, 
                                                                                add_generation_prompt = True, 
                                                                                tokenize=True, 
                                                                                return_dict=True, 
                                                                                return_tensors="pt",
                                                                                padding=True,
                                                                                truncation=False)        
                if self.enable_moelora and self.enable_task_gate:
                    moe_task_ids = [0, 1]
                    model_inputs["moe_task_ids"].append(moe_task_ids)
                model_inputs["sample_input"].append([image_sample_input_ids, text_sample_input_ids])
        return model_inputs
    @DeprecationWarning
    def multimodal_mask(self, examples):
        model_inputs = {
            "sample_input": []
        }
        failed_images_count = 0

        for i in range(len(examples[self.attributes_input_column])):
            if examples[self.attributes_input_column][i]:
                attributes_sample = examples[self.attributes_input_column][i]
                # image_sample_instruction = examples[self.image_instruction_column][i]
                image_file_name = examples[self.image_url_column][i]

                image_url = (
                    f"/hpc2hdd/home/zrao690/MultimodalEmb/data/{self.data_args.dataset_name}/handled/image/{image_file_name}"
                    if image_file_name else
                    f"/hpc2hdd/home/zrao690/MultimodalEmb/data/noise.jpg"
                )

                input_template = self.apply_message_template(attributes_sample, image_url)

                try:
                    input_ids = self.processor.apply_chat_template(
                        input_template, 
                        add_generation_prompt=True, 
                        tokenize=True, 
                        return_dict=True, 
                        return_tensors="pt",
                        padding=True,
                        truncation=False
                    )
                except Exception as e:
                    failed_images_count += 1
                    print(f"Error processing image {image_file_name}: {e}")
                    image_url = f"/hpc2hdd/home/zrao690/MultimodalEmb/data/noise.jpg"
                    input_template = self.apply_message_template(attributes_sample, image_url)
                    input_ids = self.processor.apply_chat_template(
                        input_template, 
                        add_generation_prompt=True, 
                        tokenize=True, 
                        return_dict=True, 
                        return_tensors="pt",
                        padding=True,
                        truncation=False
                    )    

                model_inputs["sample_input"].append(input_ids)

        return model_inputs

    def text_mask(self, examples):
        if self.enable_moelora and self.enable_task_gate:
            model_inputs = {
                "sample_input": [],
                "moe_task_ids": [],
            }
        else:
            model_inputs = {
                "sample_input": [],
            }

        for i in range(len(examples[self.attributes_input_column])):
            if examples[self.attributes_input_column][i]:
                attributes_sample = examples[self.attributes_input_column][i]
                input_template = self.apply_message_template(attributes_sample)

                input_ids = self.processor.apply_chat_template(
                    input_template, 
                    add_generation_prompt=True, 
                    tokenize=True, 
                    return_dict=True, 
                    return_tensors="pt",
                    padding=True,
                    truncation=False
                )
                if self.enable_moelora and self.enable_task_gate:
                    moe_task_ids = [1]
                    model_inputs["moe_task_ids"].append(moe_task_ids)
                model_inputs["sample_input"].append([input_ids])

        return model_inputs
    
    def image_mask(self, examples):
        if self.enable_moelora and self.enable_task_gate:
            model_inputs = {
                "sample_input": [],
                "moe_task_ids": [],
            }
        else:
            model_inputs = {
                "sample_input": [],
            }

        for i in range(len(examples[self.attributes_input_column])):
            if examples[self.attributes_input_column][i]:
                image_sample_instruction = examples[self.image_instruction_column][i]
                image_file_name = examples[self.image_url_column][i]
                if image_file_name == "":
                    image_url = f"/hpc2hdd/home/zrao690/MultimodalEmb/data/noise.jpg"
                else:
                    image_url = f"/hpc2hdd/home/zrao690/MultimodalEmb/data/{self.data_args.dataset_name}/handled/image/{image_file_name}"

                image_sample_input_template = self.apply_message_template(image_sample_instruction, image_url)
                
                try:
                    input_ids = self.processor.apply_chat_template(image_sample_input_template, 
                                                                    add_generation_prompt = True, 
                                                                    tokenize=True, 
                                                                    return_dict=True, 
                                                                    return_tensors="pt",
                                                                    padding=True,
                                                                    truncation=False)
                except Exception as e:

                    print(f"image_sample_input_ids: {e} file_name: {image_file_name}")
                    image_url = f"/hpc2hdd/home/zrao690/MultimodalEmb/data/noise.jpg"
                    image_sample_input_template = self.apply_message_template(image_sample_instruction, image_url)
 
                    input_ids = self.processor.apply_chat_template(image_sample_input_template, 
                                                                    add_generation_prompt = True, 
                                                                    tokenize=True, 
                                                                    return_dict=True, 
                                                                    return_tensors="pt",
                                                                    padding=True,
                                                                    truncation=False)      
                if self.enable_moelora and self.enable_task_gate:
                    moe_task_ids = [0]
                    model_inputs["moe_task_ids"].append(moe_task_ids)
                model_inputs["sample_input"].append(input_ids)
        return model_inputs

    def apply_message_template(self, text, image=None):
        if image is not None:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"{image}"},
                        {"type": "text", "text": f"{text}"}
                    ],
                }
            ]
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"{text}"}],
            }
        ]

    
class DataArgs():
    def __init__(self):
        self.dataset_name = "beauty"
        self.max_source_length = 256

# if __name__ == "__main__":
    # generate_gaussian_noise_image()
    # data_file = "/home/raozhongtao/LLMEmb/data/beauty/handled/multimodal_item_str.jsonline"
    # examples = []
    # with jsonlines.open(data_file, 'r') as f:
    #     for l in f:
    #         examples.append(l)
    # processor = AutoProcessor.from_pretrained("/data/raozhongtao/shared/Qwen2.5-VL-7B-Instruct")
    # dataArgs = DataArgs()
    # mask = QwenVLTrainMask(data_args=dataArgs, model_args=None, processor=processor)
    # mask(examples)
