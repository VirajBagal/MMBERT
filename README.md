# MMBERT

## MMBERT: Multimodal BERT Pretraining for Improved Medical VQA
Yash Khare*, Viraj Bagal*, Minesh Mathew, Adithi Devi, U Deva Priyakumar, CV Jawahar

![alt text](https://github.com/VirajBagal/MMBERT/blob/main/qualitative.png?raw=true)

Abstract: *Images in the medical domain are fundamentally different from the general domain images. Consequently, it is infeasible to directly employ general domain Visual Question Answering ( VQA ) models for the medical domain. Additionally,medical images annotation is a costly and time-consuming
process. To overcome these limitations, we propose a solution inspired by self-supervised pretraining of Transformer-style architectures for NLP , V ision and L anguage tasks. Our
method involves learning richer medical image and text semantic representations using Masked Language Modeling
(MLM) with image features as the pretext task on a large medical image+caption dataset. The proposed solution achieves
new state-of-the-art performance on two VQA datasets for
radiology images – VQA - M ed 2019 and VQA - RAD , outperforming even the ensemble models of previous best solutions.
Moreover, our solution provides attention maps which help
in model interpretability.*

## Train on VQARAD

```
python train_vqarad.py --run_name give_name --mixed_precision --use_pretrained --lr set_lr  --epochs set_epochs
```

## Train on MedVQA 2019

```
python train.py --run_name  give_name --mixed_precision --lr set_lr --category cat_name --batch_size 16 --num_vis set_visual_feats --hidden_size hidden_dim_size
```

## Evaluate 

```
python eval.py --run_name give_name --mixed_precision --category cat_name --hidden_size hidden_dim_size --use_pretrained
```

## VQARAD Results

MMBERT General, which is a single model for both the question types
in the dataset, outperforms the existing approaches including
the ones which have a dedicated model for each question
type.

| Method | Dedicated Models | Open Acc. | Closed Acc. | Overall Acc. |
| --- | --- | --- | --- | --- | 
| MEVF + SAN | - | 40.7 | 74.1 | 60.8 |
| MEVF + BAN | - | 43.9 | 75.1 | 62.7 |
| Conditional Reasoning | :heavy_check_mark: | 60.0 | 79.3 | 71.6 |
| MMBERT General | :x: | 63.1 | 77.9 | 72.0 | 



| Method | Dedicated Models | Modality Acc. | Modality Bleu | Modality Acc. | Modality Bleu | Modality Acc. | Modality Bleu | Modality Acc. | Modality Bleu | Modality Acc. | Modality Bleu | Modality Acc. | Modality Bleu | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| MEVF + SAN | - | 40.7 | 74.1 | 40.7 | 74.1 | 40.7 | 74.1 | 40.7 | 74.1 | 40.7 | 74.1 | 40.7 | 74.1 |
