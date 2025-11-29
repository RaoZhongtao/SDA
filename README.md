# Prompt Template

We concatenate fields such as the **title**, **category**, and **brand**, prefixed by:

> **"The product has the following attributes:"**

For the visual modality, the image is introduced with:

> **"The product has the image:"**


---

# Hyperparameter Settings

| Hyperparameter | Value |
|----------------|--------|
| LoRA trainable modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `down_proj`, `gate_proj`, `up_proj` |
| LoRA dropout | `0.1` |
| Learning rate | `2e-4` |
| Number of experts | `8` |
| Max sequence length | `256` |
| Batch size | `32` |
| Gradient accumulation steps | `1` |

---

# Implementation Details

All multimodal baseline implementations and evaluations are conducted using **MMRec**, a unified open-source framework for multimodal recommendation.

- For **SASRec** and **BERT4Rec**, we adopt their official implementations.
- All experiments run on **Intel Xeon Platinum 8358P** CPUs and **NVIDIA A800 80GB** GPUs.
- Embedding dimension for all models: **128**.
- Textual features: **all-MiniLM-L6-v2** (Sentence-Transformers).  
- Visual features: **Amazon Reviews** dataset features extracted with **AlexNet**.

Training details within MMRec:

- Batch size: **2048**
- Early stopping patience: **20**
- Max epochs: **1000**
- Exhaustive grid search performed for each model.

For our model:

- Max fine-tuning steps: **4000**
- Vision encoder (**ViT**) is **frozen**
- Number of experts in SDA: **8**
- Training batch size: **32**

---

# Statistics of Datasets

| Dataset | #Users | #Items | #Interactions | Sparsity |
|--------|--------|--------|----------------|----------|
| Beauty | 50,498 | 57,019 | 391,956 | 99.986% |
| Sports | 80,097 | 83,006 | 582,372 | 99.991% |
| Toys | 53,983 | 68,556 | 404,955 | 99.989% |
| Tmall | 10,440 | 42,748 | 68,835 | 99.980% |

*Table: Statistics of the datasets.*


# Additional Experiments

Below is a Markdown-converted version of the experimental table.  
Logos are replaced with `[QwenVLVL]` and `[LLaMA]`.

> **Notation**  
> - **Bold** = Best   
> - `H@10` = Hit Ratio@10  
> - `N@10` = NDCG@10  

---

## Beauty Dataset

| Model | Variant | H@10 (Overall) | N@10 (Overall) | H@10 (Tail) | N@10 (Tail) |
|-------|---------|----------------|----------------|-------------|-------------|
| **SMORE** | Base | 0.5524 | 0.3816 | 0.2464 | 0.1288 |
| SMORE | w/ SDA [QwenVL] | _0.5748_ | _0.3983_ | _0.2884_ | _0.1570_ |
| SMORE | w/ SDA [LLaMA] | **0.5759** | **0.4006** | **0.2976** | **0.1641** |
| **PGL** | Base | 0.5307 | 0.3440 | 0.2055 | 0.0910 |
| PGL | w/ SDA [QwenVL] | **0.5612** | **0.3723** | **0.2765** | **0.1304** |
| PGL | w/ SDA [LLaMA] | _0.5566_ | _0.3672_ | _0.2616_ | _0.1209_ |
| **MMGCN** | Base | 0.4720 | 0.2999 | 0.0461 | 0.0223 |
| MMGCN | w/ SDA [QwenVL] | **0.4907** | _0.3139_ | _0.0721_ | _0.0368_ |
| MMGCN | w/ SDA [LLaMA] | _0.4884_ | **0.3152** | **0.0744** | **0.0377** |
| **SLMRec** | Base | 0.4258 | 0.2852 | 0.2003 | 0.1020 |
| SLMRec | w/ SDA [QwenVL] | **0.4805** | **0.3295** | **0.3191** | **0.1766** |
| SLMRec | w/ SDA [LLaMA] | _0.4781_ | _0.3292_ | _0.3080_ | _0.1685_ |
| **VBPR** | Base | 0.4572 | 0.3112 | 0.0804 | 0.0419 |
| VBPR | w/ SDA [QwenVL] | _0.5375_ | _0.3745_ | **0.3981** | **0.2570** |
| VBPR | w/ SDA [LLaMA] | **0.5393** | **0.3762** | _0.3963_ | _0.2550_ |
| **SASRec** | Base | 0.4141 | 0.2760 | 0.0462 | 0.0229 |
| SASRec | w/ SDA [QwenVL] | _0.5752_ | **0.3951** | **0.4692** | **0.3118** |
| SASRec | w/ SDA [LLaMA] | **0.5769** | _0.3936_ | _0.4564_ | _0.2972_ |
| **BERT4Rec** | Base | 0.3955 | 0.2349 | 0.0031 | 0.0011 |
| BERT4Rec | w/ SDA [QwenVL] | **0.5535** | **0.3521** | **0.4070** | **0.2329** |
| BERT4Rec | w/ SDA [LLaMA] | _0.5211_ | _0.3234_ | 0.3390 | 0.1830 |


---

## Sports Dataset

| Model | Variant | H@10 (Overall) | N@10 (Overall) | H@10 (Tail) | N@10 (Tail) |
|-------|---------|----------------|----------------|--------------|--------------|
| SMORE | Base | 0.5862 | 0.3945 | 0.2695 | 0.1326 |
| SMORE | w/ SDA [QwenVLVL] | **0.6063** | **0.4070** | **0.3265** | **0.1657** |
| SMORE | w/ SDA [LLaMA] | _0.6057_ | _0.4055_ | _0.3215_ | _0.1611_ |
| PGL | Base | 0.5623 | 0.3613 | 0.2113 | 0.0899 |
| PGL | w/ SDA [QwenVLVL] | **0.5962** | **0.3871** | **0.2972** | **0.1335** |
| PGL | w/ SDA [LLaMA] | _0.5941_ | _0.3856_ | _0.2952_ | _0.1329_ |
| MMGCN | Base | 0.4939 | 0.3044 | 0.0435 | _0.0197_ |
| MMGCN | w/ SDA [QwenVLVL] | **0.5325** | _0.3385_ | _0.0765_ | **0.0384** |
| MMGCN | w/ SDA [LLaMA] | _0.5293_ | **0.3342** | **0.0793** | **0.0384** |
| SLMRec | Base | 0.4632 | 0.2962 | 0.2926 | 0.1429 |
| SLMRec | w/ SDA [QwenVLVL] | **0.5115** | **0.3367** | _0.3545_ | _0.1880_ |
| SLMRec | w/ SDA [LLaMA] | _0.5097_ | _0.3282_ | **0.3716** | **0.1904** |
| VBPR | Base | 0.4969 | 0.3307 | 0.0839 | 0.0399 |
| VBPR | w/ SDA [QwenVLVL] | **0.5731** | **0.3781** | _0.4306_ | **0.2598** |
| VBPR | w/ SDA [LLaMA] | _0.5718_ | _0.3773_ | **0.4336** | _0.2593_ |
| SASRec | Base | 0.4441 | 0.2882 | 0.0462 | 0.0216 |
| SASRec | w/ SDA [QwenVLVL] | _0.6192_ | **0.3964** | **0.5209** | **0.3284** |
| SASRec | w/ SDA [LLaMA] | **0.6200** | _0.3950_ | _0.5188_ | _0.3234_ |
| BERT4Rec | Base | 0.3877 | 0.2246 | 0.0017 | 0.0006 |
| BERT4Rec | w/ SDA [QwenVLVL] | **0.5847** | **0.3515** | **0.4344** | **0.2413** |
| BERT4Rec | w/ SDA [LLaMA] | 0.5362 | 0.3096 | 0.3717 | 0.1933 |


---

## Toys Dataset

| Model | Variant | H@10 (Overall) | N@10 (Overall) | H@10 (Tail) | N@10 (Tail) |
|-------|---------|----------------|----------------|--------------|--------------|
| SMORE | Base | 0.5590 | 0.3893 | 0.3151 | 0.1900 |
| SMORE | w/ SDA [QwenVLVL] | _0.5799_ | _0.4023_ | _0.3574_ | _0.2226_ |
| SMORE | w/ SDA [LLaMA] | **0.5818** | **0.4062** | **0.3651** | **0.2278** |
| PGL | Base | 0.5465 | 0.3573 | 0.2971 | 0.1529 |
| PGL | w/ SDA [QwenVLVL] | _0.5709_ | _0.3769_ | **0.3531** | **0.1913** |
| PGL | w/ SDA [LLaMA] | **0.5738** | **0.3790** | _0.3512_ | _0.1895_ |
| MMGCN | Base | 0.4711 | 0.2939 | 0.0981 | 0.0539 |
| MMGCN | w/ SDA [QwenVLVL] | **0.4972** | **0.3180** | **0.1254** | _0.0715_ |
| MMGCN | w/ SDA [LLaMA] | _0.4956_ | _0.3170_ | _0.1253_ | **0.0751** |
| SLMRec | Base | 0.4617 | 0.3115 | 0.3315 | 0.1890 |
| SLMRec | w/ SDA [QwenVLVL] | **0.5048** | **0.3453** | **0.3988** | **0.2357** |
| SLMRec | w/ SDA [LLaMA] | _0.5004_ | _0.3419_ | _0.3949_ | _0.2320_ |
| VBPR | Base | 0.4536 | 0.3061 | 0.1252 | 0.0751 |
| VBPR | w/ SDA [QwenVLVL] | **0.5556** | **0.3915** | **0.4761** | **0.3308** |
| VBPR | w/ SDA [LLaMA] | _0.5510_ | _0.3898_ | _0.4633_ | _0.3215_ |
| SASRec | Base | 0.4126 | 0.2758 | 0.0782 | 0.0452 |
| SASRec | w/ SDA [QwenVLVL] | _0.6125_ | _0.4333_ | _0.5522_ | _0.3914_ |
| SASRec | w/ SDA [LLaMA] | **0.6148** | **0.4367** | **0.5818** | **0.4196** |
| BERT4Rec | Base | 0.3554 | 0.2032 | 0.0045 | 0.0017 |
| BERT4Rec | w/ SDA [QwenVLVL] | **0.6138** | **0.4190** | **0.5288** | **0.3609** |
| BERT4Rec | w/ SDA [LLaMA] | _0.5613_ | _0.3622_ | _0.4941_ | _0.3148_ |

---

### Tmall Results

| Model | Variant | H@10 (Overall) | N@10 (Overall) | H@10 (Tail) | N@10 (Tail) |
|-------|---------|--------------|--------------|-----------|-----------|
| **SMORE** | Base | 0.2069 | 0.1658 | 0.0815 | 0.0420 |
| SMORE | CAMEL (QwenVL) | **0.3275** | **0.2646** | _0.2255_ | **0.1520** |
| SMORE | CAMEL (LLaMA) | _0.3262_ | **0.2646** | **0.2263** | _0.1512_ |
| **PGL** | Base | 0.3185 | 0.2086 | 0.2373 | 0.1103 |
| PGL | CAMEL (QwenVL) | **0.4396** | **0.3030** | **0.4085** | **0.2262** |
| PGL | CAMEL (LLaMA) | _0.4185_ | _0.2862_ | _0.3797_ | _0.2043_ |
| **MMGCN** | Base | 0.1374 | 0.0960 | 0.0000 | 0.0000 |
| MMGCN | CAMEL (QwenVL) | **0.1504** | _0.1093_ | 0.0000 | 0.0000 |
| MMGCN | CAMEL (LLaMA) | _0.1501_ | **0.1107** | 0.0000 | 0.0000 |
| **SLMRec** | Base | 0.2997 | 0.2051 | 0.2203 | 0.1036 |
| SLMRec | CAMEL (QwenVL) | **0.3824** | **0.2685** | **0.3295** | **0.1763** |
| SLMRec | CAMEL (LLaMA) | _0.3742_ | _0.2657_ | _0.3116_ | _0.1680_ |
| **VBPR** | Base | 0.3404 | 0.2487 | 0.3069 | 0.1931 |
| VBPR | CAMEL (QwenVL) | **0.4745** | **0.3922** | **0.4830** | **0.3680** |
| VBPR | CAMEL (LLaMA) | _0.4682_ | _0.3871_ | _0.4741_ | _0.3612_ |
| **SASRec** | Base | 0.3299 | 0.3151 | 0.2049 | 0.2020 |
| SASRec | CAMEL (QwenVL) | **0.7045** | **0.6154** | **0.6532** | **0.5587** |
| SASRec | CAMEL (LLaMA) | _0.6740_ | 0.5602 | 0.6261 | 0.5128 |
| **BERT4Rec** | Base | 0.2779 | 0.1790 | 0.1704 | 0.1098 |
| BERT4Rec | CAMEL (QwenVL) | **0.6870** | **0.5857** | **0.6302** | **0.5252** |
| BERT4Rec | CAMEL (LLaMA) | _0.6707_ | _0.5491_ | _0.6185_ | _0.4915_ |
| **Avg. Improv.** | Base | **25.70%** | **30.97%** | **51.72%** | **75.69%** |



# Notes

- `[QwenVL]` = QwenVLVL2.5-VL 7B  
- `[LLaMA]` = LLaMA-3.2 11B Vision  

