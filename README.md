

## Overview

<p align="center"><img src="./teaser.png" alt="teaser" width="500px" /></p>

Hallucination remains a significant challenge for Multimodal Large Language Models (MLLMs), hindering their reliability across various tasks. Despite extensive research from various perspectives, the underlying causes remain unclear. In this paper, we conduct empirical analyses and identify a progressive attention shift in the decoding process, where the decoder’s attention over visual tokens gradually diverges from the vision encoder’s. Based on these observations, we infer that this shift systematically reduces the model’s focus on semantically important visual tokens, leading to hallucinations. Building on this finding, we propose Align Attention with Image (AAI), a decoding-time method that explicitly aligns the decoder’s attention over visual tokens with the self-attention of the vision encoder. Specifically, AAI caches the encoder’s visual self-attention and leverages it as a reference signal to guide the decoder’s attention distribution toward that of the image. AAI is decoding-agnostic and can be seamlessly integrated with both classical and modern decoding strategies across different MLLMs. We evaluate AAI on widely used hallucination benchmarks and show that it consistently reduces hallucinations without sacrificing semantic completeness.All relevant experimental code is included in the supplementary appendix and will be released publicly.

## Setup

The main implementation of AAI is in `extract.py` and `attention.py`.

So it is convenient to use AAI method.
```
conda env create -f environment.yml
conda activate aai
```
#### Note: to implement OPERA on other version of transformers, you can follow the steps as the follows:
- Find the file at `transformers-4.29.2/src/transformers/generation/utils.py`.
- Add the arguments in `transformers.generate` function [here](https://github.com/shikiw/OPERA/blob/aa968c7501f4d3d8362f4b3bcab855024f4da5f6/transformers-4.29.2/src/transformers/generation/utils.py#L1156-L1162).
- Add the code in `transformers.generate` function [here](https://github.com/shikiw/OPERA/blob/aa968c7501f4d3d8362f4b3bcab855024f4da5f6/transformers-4.29.2/src/transformers/generation/utils.py#L1619-L1665).
- Copy and paste the `opera_decoding` function [here](https://github.com/shikiw/OPERA/blob/aa968c7501f4d3d8362f4b3bcab855024f4da5f6/transformers-4.29.2/src/transformers/generation/utils.py#L3116-L3674).

## TL;DR
After setup the environment, you can directly use AAI on your own MLLM model by:
```
# copy `extract.py` and `attention.py` and use followed function before generate

```
extract_vision(model_name,model,image, questions)
```
```
llama_modify(
        model,
        start_layer,
        end_layer,
        use_aai,
        alpha,
        beta,
        img_start_idx,
        img_end_idx,
        vis_ref
    )
```



## Evaluation

The following evaluation requires for MSCOCO 2014 dataset. Please download [here](https://cocodataset.org/#home) and extract it in your data path.

Besides, it needs you to prepare the following checkpoints of 7B base models:

- Download [LLaVA-1.5 merged 7B model](https://huggingface.co/liuhaotian/llava-v1.5-7b) and specify it.
- Download [Vicuna 7B v0 model](https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main) and specify it.
- Download [MiniGPT-4 7B pretrained weights](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing) and specify it.
- Download [Shikra merged 7B model](https://github.com/shikras/shikra#checkpoint) and and specify it.

### CHAIR
- Generate the MLLM's responses and save them in a jsonl file:
```bash
python chair_eval.py --model  MODEL_NAME --data-path /path/to/COCO/val2014 --use-aai --alpha 0.3 --start-layer 2 --end-layer 32 --beam 1 --beta 1.0

```


- Calculate CHAIR using the generated jsonl file:
```bash
python chair.py --cap_file /path/to/jsonl --image_id_key image_id --caption_key caption --coco_path /path/to/COCO/annotations_trainval2014/annotations/ --save_path /path/to/save/jsonl
```

### POPE
```bash
python pope_eval.py --model MODEL_NAME --data_path /path/to/COCO --pope-type random --gpu-id GPU_IDs --beam 5 --scale_factor 50 --threshold 15 --num_attn_candidates 5 --penalty_weights 1
```





## Acknowledgement
This repo is based on the MLLM codebase of [OPERA](https://github.com/shikiw/OPERA/) and [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) and the CHAIR code of [PAI](https://github.com/LALBJ/PAI). Thanks for their impressive works!

<!-- ## Citation
``` -->