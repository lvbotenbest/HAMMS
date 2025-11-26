TRAIN_FILE=train-v1.txt
MODEL_DIR=/model-v1
OUT_DIR=/output_dir




CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8  torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 train.py --train_dataset $TRAIN_FILE \
                    --model_name_or_path $MODEL_DIR \
                    --output_dir $OUT_DIR \
                    --do_train true \
                    --cutoff_len 1024 \
                    --per_device_train_batch_size 1 \
                    --gradient_accumulation_steps 8 \
                    --learning_rate 1.0e-4 \
                    --num_train_epochs 3.0 \
                    --lr_scheduler_type cosine \
                    --warmup_ratio 0.1 \
                    --ddp_timeout 180000000 \
                    --preprocessing_num_workers 8 \
                    --overwrite_cache True \
                    --logging_steps 100 \
                    --save_steps 1000 \
                    --plot_loss true \
                    --use_hyper true \
                    --lora_train_hyper true \
                    --use_img true \
                    --save_only_model true \
                    --save_safetensors False \
                    --per_lang_pair_batch_size 1 \
                    --hyper_classification true \
                    --hyper_classification_loss_ratio 0.4




