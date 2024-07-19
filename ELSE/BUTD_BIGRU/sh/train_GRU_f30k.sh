DATASET_NAME='f30k'
DATA_PATH='../data/f30k'
MODEL_NAME='runs/bigru_f30k'
VOCAB_PATH='../data/vocab'

cd ../
CUDA_VISIBLE_DEVICES=1 python3 train.py \
  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH}\
  --logger_name ${MODEL_NAME}/log --model_name ${MODEL_NAME} \
  --num_epochs=30 --lr_update=15 --learning_rate=.0005 --workers 10 \
  --log_step 200 --embed_size 1024 --vse_mean_warmup_epochs 1 --batch_size 128 --hardnum 2

python3 eval.py --dataset coco --data_path ${DATA_PATH} --model_name ${MODEL_NAME}





