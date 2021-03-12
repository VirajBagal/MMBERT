# MMBERT
MMBERT: Multimodal BERT Pretraining for Improved Medical VQA

# Train on VQARAD

```
python train_vqarad.py --run_name give_name --mixed_precision --use_pretrained --lr set_lr  --epochs set_epochs

```

# Train on MedVQA 2019

```
python train.py --run_name  give_name --mixed_precision --lr set_lr --category cat_name --batch_size 16 --num_vis set_visual_feats --hidden_size hidden_dim_size

```

