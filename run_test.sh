
CUDA_VISIBLE_DEVICES=0  python test.py \
                       --use_hyper true \
                       --model_name_or_path /home/liunayu/copy/lvbo/llama3-hamms/pre_model/img_model-v1 \
                       --test_checkpoint ./checkpoint/classification/v1-0.4-warm/checkpoint-7000 \
                       --output_prediction_path ./results/results/classification/v1-0.4-1 \
                       --hyper_predict true \
                       --use_img true \

