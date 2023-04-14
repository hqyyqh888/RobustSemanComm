TF_ENABLE_ONEDNN_OPTS=0 CUDA_VISIBLE_DEVICES=1  python3  run_class_main.py \
    --model  ViT_FIM_model  \
    --output_dir ckpt_record   \
    --data_set cifar_S32 \
    --batch_size 50 \
    --input_size 224 \
    --lr  3e-5 \
    --epochs 200  \
    --opt_betas 0.95 0.99  \
    --save_freq 2   \
    --mask_ratio 0.0 \
    --train_type fim_train \
    --resume /Data1/zhangguangyi/SemanRes1/PaperCode_V2/ckpt_record/ckpt_fim_train/checkpoint-55.pth \
    --if_attack_train \
    --if_attack_test \
    --eval

    