# here put the import lib
import os
import json
import pickle
import torch
import deepspeed
import numpy as np
import jsonlines
from datasets import load_dataset
import wandb
from llm.peft import (
    LoraConfig,
    PeftModel,
    MOELoraConfig
)
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import MllamaForConditionalGeneration

from llm.llama import LlamaRSEmb
from llm.qwen_vl import QwenVLRSEmb
from llm.llama_vision import LlamaVisionRSEmb
from llm.trainer_seq2seq import MedRecTrainer, LlamaRecTrainer
from llm.lora_cls import PeftModelForCLS
from llm.arguments import DataTrainingArguments, ModelArguments
from llm.data_processor.llama_mask import llama_train_mask, llama_eval_mask
from llm.data_processor.qwen_vl_mask import QwenVLTrainMask, QwenVLEvalMask
from llm.data_processor.llama_vision_mask import LlamaVisionEvalMask, LlamaVisionTrainMask
from llm.data_processor.llama_collator import LongestSequenceMaskCollator, PairwiseDataCollatorWithPadding
from llm.data_processor.qwen_vl_collator import QwenVLEvalCollator, QwenVLTrainCollator
from llm.data_processor.llama_vision_collator import LlamaVisionEvalCollator, LlamaVisionTrainCollator


class LossEarlyStoppingCallback(TrainerCallback):
    def __init__(self):
        self.threshold = 0.08



    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # 如果你想在每个 step 检查 loss，可以在这里加逻辑
        if state.log_history and 'loss' in state.log_history[-1]:
            if state.log_history[-1]['loss'] < self.threshold:
                print(f"Early stopping: loss {state.log_history[-1]['loss']} < {self.threshold}")
                control.should_training_stop = True
        return control

def initWandb(model_choice, dataset_name, args):
    wandb.login(key="5b696414a6b88f31b9fe512c3d1d4e5ff07b09ac")
    wandb_run_name = f"MoE_inference_training_{model_choice}_{dataset_name}"
    wandb_config = {
        "project": "MoE", 
        "entity": "MultimodelRec",
        "config": args,
        "name": wandb_run_name
    }
    
    wandb.init(**wandb_config)

def find_llama_cross_attn_layers(model, verbose=True):
    """
    Find cross-attention modules in the model.
    """
    cross_attn_modules = []
    for idx, layer_module in enumerate(model.language_model.model.layers):
        if hasattr(layer_module, 'cross_attn'):
            
            name = f"language_model.model.layers.{idx}."
            print(f"debugging cross_attn layer: {name} has cross_attn")
            cross_attn_modules.append(name)
    
    if verbose:
        print(f"Found {len(cross_attn_modules)} cross-attention modules: {cross_attn_modules}")
    
    return cross_attn_modules

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

# save model for PeftModel
class SavePeftModelCallback(TrainerCallback):
    
    # def __init__(self):
    #     self.threshold = 0.08
        
    def on_save(    
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            print('+++++++++++++++++save call back++++++++++++++++')
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )
            kwargs["model"].save_pretrained(checkpoint_folder)

            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            return control
    
    # def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     # 如果你想在每个 step 检查 loss，可以在这里加逻辑
    #     if state.log_history and 'loss' in state.log_history[-1]:
    #         if state.log_history[-1]['loss'] < self.threshold:
    #             print(f"Early stopping: loss {state.log_history[-1]['loss']} < {self.threshold}")
    #             control.should_training_stop = True
    #     return control
        

def train():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    initWandb(model_choice=model_args.model_choice, dataset_name=data_args.dataset_name, args=model_args)
    device_map = "cuda"
    ## Load Tokenizer ##


    ## Load Model ##
    if model_args.model_choice == "Llama":
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # define the padding direction
        model = LlamaRSEmb.from_pretrained(
            model_args.model_name_or_path,
            pool_type=model_args.pool_type,
            tau=model_args.tau,
        ).half().cuda()
        model.config.pad_token_id = tokenizer.pad_token_id
        processor = tokenizer
        if training_args.do_train:
            preprocess_func = llama_train_mask(data_args, model_args, tokenizer)
            data_collator = PairwiseDataCollatorWithPadding(tokenizer)

        else:
            preprocess_func = llama_eval_mask(data_args, model_args, tokenizer)
            data_collator = LongestSequenceMaskCollator(tokenizer)
    elif model_args.model_choice == "QwenVL":
        min_pixels = 20 * 20  # Adjusted minimum pixel count
        max_pixels = 256 * 256  # Adjusted maximum pixel count

        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, min_pixels=min_pixels, max_pixels=max_pixels)
        
        processor.tokenizer.padding_side = "left"
        # if training_args.enable_moelora and model_args.item_emb_path is None:
        #     raise ValueError("Please provide item embedding path for MOELora")
        model = QwenVLRSEmb.from_pretrained(
            model_args.model_name_or_path,
            pool_type=model_args.pool_type,
            tau=model_args.tau,
            multimodal_embed=training_args.multimodal_embed,
            seperate_eval=training_args.seperate_eval,
            concat_features=training_args.concat_features,
            image_only=model_args.image_only,
            enable_moelora=training_args.enable_moelora,
            item_emb_path=model_args.item_emb_path,
            attn_implementation="flash_attention_2",
            info_nce=training_args.info_nce,
        ).half().cuda()
        model.config.pad_token_id = processor.tokenizer.pad_token_id

        if training_args.do_train:
            preprocess_func = QwenVLTrainMask(data_args, model_args, training_args, processor)
            data_collator = QwenVLTrainCollator(processor)

        else:
            preprocess_func = QwenVLEvalMask(data_args, model_args, training_args, processor)
            data_collator = QwenVLEvalCollator(processor)
    elif model_args.model_choice == "LlamaVision":
        min_pixels = 20 * 20  # Adjusted minimum pixel count
        max_pixels = 256 * 256   # Adjusted maximum pixel count

        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, min_pixels=min_pixels, max_pixels=max_pixels)
        
        processor.tokenizer.padding_side = "left"
        # if training_args.enable_moelora and model_args.item_emb_path is None:
        #     raise ValueError("Please provide item embedding path for MOELora")
        # model = MllamaForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
        if training_args.do_train:
            model = LlamaVisionRSEmb.from_pretrained(
                model_args.model_name_or_path,
                pool_type=model_args.pool_type,
                tau=model_args.tau,
                multimodal_embed=training_args.multimodal_embed,
                seperate_eval=training_args.seperate_eval,
                concat_features=training_args.concat_features,
                image_only=model_args.image_only,
                enable_moelora=training_args.enable_moelora,
                item_emb_path=model_args.item_emb_path,
                info_nce=training_args.info_nce,
                output_hidden_states=True,
            ).half().cuda()
        else:
            model = LlamaVisionRSEmb.from_pretrained(
                model_args.model_name_or_path,
                pool_type=model_args.pool_type,
                tau=model_args.tau,
                multimodal_embed=training_args.multimodal_embed,
                seperate_eval=training_args.seperate_eval,
                concat_features=training_args.concat_features,
                image_only=model_args.image_only,
                enable_moelora=training_args.enable_moelora,
                item_emb_path=model_args.item_emb_path,
                info_nce=training_args.info_nce,
                output_hidden_states=True,
            ).float().cuda()
        model.config.pad_token_id = processor.tokenizer.pad_token_id

        # print(f"debugging model : {model}")
        # raise ValueError("LlamaVision is not supported yet, please use QwenVL or Llama")
            
        if training_args.do_train:
            preprocess_func = LlamaVisionTrainMask(data_args, model_args, training_args, processor)
            data_collator = LlamaVisionTrainCollator(processor)

        else:
            preprocess_func = LlamaVisionEvalMask(data_args, model_args, training_args, processor)
            data_collator = LlamaVisionEvalCollator(processor)
    else:
        raise ValueError("No such LLM model")

    if not model_args.zero_shot:
        if model_args.peft_path is not None:    # for test model
            # Resume_training
            
            if training_args.resume_from_checkpoint is not None:
                model = PeftModelForCLS.from_pretrained(model, model_args.peft_path, is_trainable=True)
            else:
                model = PeftModelForCLS.from_pretrained(model, model_args.peft_path, is_trainable=False)
        else:   # for train model
            # Load Lora Config
            exclude_modules = ['lm_head', 'embed_tokens', 'srs_emb']
            if model_args.model_choice == "LlamaVision":
                if (not training_args.multimodal_embed) or model_args.freeze_visual_tower:
                    exclude_modules.append('vision')
                    exclude_modules.append('multi_modal_projector')
                    # cross_attn_modules = find_llama_cross_attn_layers(model, verbose=True)
                    # exclude_modules.extend(cross_attn_modules)
                    exclude_modules.append('cross_attn')
                    model.vision_model.eval()  # 设置为评估模式
                    model.vision_model.to(device_map) 
                    print(f"model.vision_model device {model.vision_model.device}")
            elif model_args.model_choice == "QwenVL":
                if (not training_args.multimodal_embed) or model_args.freeze_visual_tower:
                    exclude_modules.append('visual')
                    model.visual.eval()  # 设置为评估模式
                    model.visual.to(device_map) 
                    print(f"model.visual device {model.visual.device}")
            target_modules = find_target_linear_names(model, lora_namespan_exclude=exclude_modules, num_lora_modules=-1, verbose=False)
            if training_args.enable_moelora:
                if training_args.enable_task_gate:
                    peft_config = MOELoraConfig(
                        r=model_args.lora_rank,
                        lora_alpha=model_args.lora_alpha,
                        expert_num=model_args.num_moe,
                        task_num=2,
                        task_embedding_dim=64,
                        target_modules=target_modules,
                        lora_dropout=model_args.lora_dropout,
                        task_type="SEQ_CLS",
                    )
                else:
                    peft_config = MOELoraConfig(
                        r=model_args.lora_rank,
                        lora_alpha=model_args.lora_alpha,
                        expert_num=4,
                        item_emb_dim=128,
                        target_modules=target_modules,
                        lora_dropout=model_args.lora_dropout,
                        task_type="SEQ_CLS",
                    )
            else:
                peft_config = LoraConfig(
                    r=model_args.lora_rank,
                    lora_alpha=model_args.lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=model_args.lora_dropout,
                    task_type="SEQ_CLS",
                )
            model = PeftModelForCLS(model, peft_config)  # LoRA wrapped llama
            # if not training_args.enable_moelora:
            # save_dir = "./saved_grads"
            # os.makedirs(save_dir, exist_ok=True)

            # # 你感兴趣的层索引

            # def save_grad_hook(name):
            #     def hook_fn(grad):
            #         filename = os.path.join(save_dir, f"image_{name}.pt")
            #         torch.save(grad.cpu(), filename)
            #         print(f"[HOOK] Saved gradient for {name} to {filename}")
            #     return hook_fn
            # moe_params = ["base_model.model.model.layers.27.self_attn.q_proj.moelora_B.default.linear_B.0.weight",
            #           "base_model.model.model.layers.27.self_attn.k_proj.moelora_B.default.linear_B.0.weight",
            #           "base_model.model.model.layers.27.self_attn.v_proj.moelora_B.default.linear_B.0.weight",
            #           "base_model.model.model.layers.27.self_attn.o_proj.moelora_B.default.linear_B.0.weight"]
            # lora_params = ["base_model.model.model.layers.27.self_attn.q_proj.lora_B.default.weight",
            #                "base_model.model.model.layers.27.self_attn.k_proj.lora_B.default.weight",
            #           "base_model.model.model.layers.27.self_attn.v_proj.lora_B.default.weight",
            #           "base_model.model.model.layers.27.self_attn.o_proj.lora_B.default.weight"]
            # for name, param in model.named_parameters():
            #     if name in lora_params:
            #         param.register_hook(save_grad_hook(name))

                            
        model.print_trainable_parameters()
    
    # print(f"model: {model}")
    # find_llama_cross_attn_layers(model, verbose=True)
    # raise ValueError("LlamaVision is not supported yet, please use QwenVL or Llama")
    if training_args.do_train:
        for name, param in model.named_parameters():    # activate the head attention parameters
            if "head_attn" in name:
                param.requires_grad = True
            if "tau" in name:
                try:
                    param.requires_grad = True
                except:
                    pass
            if "item_wte" in name:
                param.requires_grad = True
            if "projector" in name:
                param.requires_grad = True
            if "cls_head" in name:
                param.requires_grad = True


    


    ## Load Dataset ##
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file

    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    print("raw_datasets: ", raw_datasets)

    if training_args.do_train:
        target_dataset = raw_datasets["train"]
    elif training_args.do_eval:
        target_dataset = raw_datasets["eval"]
    elif training_args.do_predict:
        target_dataset = raw_datasets["test"]
    
    

    with training_args.main_process_first(desc="Dataset map pre-processing"):
        target_dataset = target_dataset.map(
            preprocess_func,
            batched=True,
            num_proc=8,  # 限制为 4 个进程，防止内存爆炸
            writer_batch_size=100,
            batch_size=500,
            # load_from_cache_file=False,  # 不读取已有 cache 
            desc="Running tokenizer on prediction dataset",
        )
    target_dataset.set_format("torch")
    
    training_args.remove_unused_columns = False  # important for pairwise dataset
    training_args.report_to="none"
    training_args.gradient_clipping = 5.0
    ## Set Trainer ##
    if model_args.model_choice == "LlamaVision":
        trainer = LlamaRecTrainer(
            model=model,
            args=training_args,
            train_dataset=target_dataset if training_args.do_train else None,
            processing_class=processor,
            data_collator=data_collator,
            compute_metrics=None,
            callbacks=([SavePeftModelCallback] if isinstance(model, PeftModel) else None), # substitute the original model saver
        )
    else:
        trainer = MedRecTrainer(
            model=model,
            args=training_args,
            train_dataset=target_dataset if training_args.do_train else None,
            processing_class=processor,
            data_collator=data_collator,
            compute_metrics=None,
            callbacks=([SavePeftModelCallback] if isinstance(model, PeftModel) else None), # substitute the original model saver
        )

    ## Train Model
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_state()

    if model_args.model_choice == "said":

        item_emb = model.item_wte.weight    # get the embedding
        item_emb = item_emb.detach().cpu().numpy().astype(float)  # convert to numpy
        
        item_emb = item_emb[1:, :]  # remove the padding item
        pickle.dump(item_emb, open("data/fashion/handled/{}.pkl".format(model_args.output_file), "wb"))

    ## Evaluation ##
    results = {}

    if training_args.do_predict:

        if model_args.model_choice == "said":

            item_emb = model.item_wte.weight    # get the embedding
            item_emb = item_emb.detach().cpu().numpy().astype(float)  # convert to numpy
            
            item_emb = item_emb[1:, :]  # remove the padding item
            pickle.dump(item_emb, open("data/fashion/handled/{}.pkl".format(model_args.output_file), "wb"))

            results = None

        else:

            list_test_samples = []
            with open(data_args.test_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = json.loads(line)
                    list_test_samples.append(line)

            # start_time = time.time()
            with torch.no_grad():
                predict_results = trainer.predict(
                    target_dataset,
                    metric_key_prefix="predict",
                )
            # end_time = time.time()

            if trainer.is_world_process_zero():
                predictions = predict_results.predictions
                assert len(predictions) == len(list_test_samples)
                hidden_states = predict_results.label_ids

                output_prediction_file = os.path.join(training_args.output_dir, model_args.output_file)

                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    for idx, p in enumerate(predictions):
                        samp = list_test_samples[idx]
                        #samp["target"] = ehr_tokenizer.med_voc.idx2word[p]
                        samp["hidden_states"] = hidden_states[idx].astype(float).tolist()
                        samp["target"] = p.astype(float).tolist()
                        res = json.dumps(samp, ensure_ascii=False)
                        writer.write(f"{res}\n")
                if training_args.multimodal_embed and training_args.seperate_eval:
                    dim = 3584 if model_args.model_choice == "QwenVL" else 4096
                    convert_emb(output_prediction_file, data_args.dataset_name, dim)
                else:
                    convert_emb_text(output_prediction_file, data_args.dataset_name)
                results = None

    return results

from sklearn.decomposition import PCA
from tqdm import tqdm

def read_data(file_path):
    """Read data from jsonlines file efficiently using a generator."""
    
    print("Parsing data...")
    with jsonlines.open(file_path, "r") as f:
        for meta_data in f:
            yield meta_data  # Yields one record at a time (lazy loading)
    print("Parsing data done!")

def save_pooled_emb(llmemb_file_path, dataset, dim):
    # file_name = "mm_0318_last"
    # dataset = "beauty"
    # file_path = os.path.join(f"results/{dataset}/llm-emb", f"{file_name}.json")
    output_file_name = os.path.splitext(os.path.basename(llmemb_file_path))[0]
    data = read_data(llmemb_file_path)
    new_data = {}
    llm_emb = []
    
    for meta_data in tqdm(data, desc="Reading hidden_states"):
        new_data[str(meta_data["item_id"])] = meta_data["hidden_states"]
        llm_emb.append(meta_data["hidden_states"])
        
    llm_emb = np.array(llm_emb)
    llm_image_emb, llm_text_emb = llm_emb[:, :dim], llm_emb[:, dim:]
    print(f"llm_image_emb shape: {llm_image_emb.shape}")
    print(f"llm_text_emb shape: {llm_text_emb.shape}")

    print(f"llm emb example: {llm_emb[0]}")

    # image_pca_emb = pca.fit_transform(llm_image_emb)
    # print("Image PCA finish")
    # text_pca_emb = pca.fit_transform(llm_text_emb)

    image_pca_emb = llm_image_emb
    text_pca_emb = llm_text_emb

    pca_llm_emb = np.concatenate((image_pca_emb, text_pca_emb), axis=1)
    
    output_dir = os.path.join(f"data/{dataset}/handled")
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    output_filename = "{}.pkl".format(output_file_name).replace("_concat", "_pooled_768")
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, "wb") as f:
        pickle.dump(pca_llm_emb, f)
    # pickle.dump(pca_llm_emb, open(os.path.join(f"data/{dataset}/handled/", "{}_pca.pkl".format(output_file_name)).replace("_concat", "_late_concat")), "wb")
    print("Data saved!")

def convert_emb(llmemb_file_path, dataset, dim):
    # file_name = "mm_0318_last"
    # dataset = "beauty"
    # file_path = os.path.join(f"results/{dataset}/llm-emb", f"{file_name}.json")
    output_file_name = os.path.splitext(os.path.basename(llmemb_file_path))[0]
    data = read_data(llmemb_file_path)
    new_data = {}
    llm_emb = []
    
    for meta_data in tqdm(data, desc="Reading hidden_states"):
        new_data[str(meta_data["item_id"])] = meta_data["hidden_states"]
        llm_emb.append(meta_data["hidden_states"])
        
    llm_emb = np.array(llm_emb)
    llm_image_emb, llm_text_emb = llm_emb[:, :dim], llm_emb[:, dim:]
    print(f"llm_image_emb shape: {llm_image_emb.shape}")
    print(f"llm_text_emb shape: {llm_text_emb.shape}")
    print("Going to do PCA")
    print(f"llm emb example: {llm_emb[0]}")
    pca = PCA(n_components=768)
    image_pca_emb = pca.fit_transform(llm_image_emb)
    print("Image PCA finish")
    text_pca_emb = pca.fit_transform(llm_text_emb)
    print("Text PCA finish")
    
    print("PCA finish")
    pca_llm_emb = np.concatenate((image_pca_emb, text_pca_emb), axis=1)
    
    output_dir = os.path.join(f"data/{dataset}/handled")
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    output_filename = "{}_pca.pkl".format(output_file_name).replace("_concat", "_late_concat")
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, "wb") as f:
        pickle.dump(pca_llm_emb, f)
    # pickle.dump(pca_llm_emb, open(os.path.join(f"data/{dataset}/handled/", "{}_pca.pkl".format(output_file_name)).replace("_concat", "_late_concat")), "wb")
    print("Data saved!")
    
def convert_emb_text(llmemb_file_path, dataset):
    output_file_name = os.path.splitext(os.path.basename(llmemb_file_path))[0]
    data = read_data(llmemb_file_path)
    new_data = {}
    llm_emb = []

    for meta_data in tqdm(data, desc="Reading hidden_states"):
        new_data[str(meta_data["item_id"])] = meta_data["hidden_states"]
        llm_emb.append(meta_data["hidden_states"])
    
    llm_emb = np.array(llm_emb)
    pca = PCA(n_components=1536)
    pca_llm_emb = pca.fit_transform(llm_emb)
    
    output_dir = os.path.join(f"data/{dataset}/handled")
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    output_filename = "{}_pca.pkl".format(output_file_name)
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, "wb") as f:
        pickle.dump(pca_llm_emb, f)
    
if __name__ == "__main__":

    train()







