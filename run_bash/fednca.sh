#!/bin/bash

CODE_PTH="${CODE_PTH:-$(pwd)}"          # repo root (run from repo root, or export CODE_PTH)
DATA_PTH="${DATA_PTH:-/path/to/dataset}" # dataset root (export DATA_PTH=/your/data)

cd $CODE_PTH

declare CLIENT_LABEL_LIST=('actinic keratoses' 'basal cell carcinoma' 'benign keratosis-like lesions' 'dermatofibroma'
'melanocytic nevi' 'melanoma' 'vascular lesions')

export TORCH_HOME=$CODE_PTH/.torch
export XDG_CACHE_HOME=$CODE_PTH/.cache

declare -a SEEDS
for i in 1 2 3 4 5; do
    SEEDS+=($RANDOM)
done

for s in "${SEEDS[@]}"; do
    echo ">>> Running with seed ${s}"
    accelerate launch \
    --config_file accelerate_config.yaml \
    ./main/main.py \
    --encoder_name resnet18 \
    --pretrained_model_name 'imagenet' \
    --data_dir $DATA_PTH \
    --json_text_folder multi_label_2_all_50_NON_IID_data_distribution_10_client_1_step_alpha_0dot1_w_missing_cls_5 \
    --json_img_folder multi_label_2_all_50_NON_IID_data_distribution_10_client_1_step_alpha_0dot1_w_missing_cls_5 \
    --do_train \
    --output_dir ./logs \
    --batch_size 32 \
    --comm_rounds 100 \
    --local_epochs 1 \
    --lr 1e-4 \
    --optimizer_mode full \
    --seed "${s}" \
    --splits train val test \
    --dataset_name medmnist_derma \
    --num_fl_tasks 10 \
    --num_cl_tasks 1 \
    --customer_label_list "${CLIENT_LABEL_LIST[@]}" \
    --exp_name DermaMNIST_alpha0dot1_fednca_seed"${s}" \
    --loss_type binary_ce_w_dual_reg_rejection_n_contrastive_topK_PSC \
    --classifier_type MultiLabel_ETF_Classifier \
    --cl_method direct_ft \
    --fl_method standard_aggregation \
    --get_pred_probability_function sigmoid \
    --pred_threshold_type fix_05 \
    --evaluate_only_first_client_flag \
    --obtain_class_wise_feature \
    --projection_type DETR_Attention_fixed_etf_init \
    --hyparam_rejection_loss 0.01 \
    --hyparam_contractive_loss 1
done
