####for tnews
#python naive_bayes_classifier.py tnews inv

##lr=0.0005 0.55 0.53
#CUDA_VISIBLE_DEVICES=1 python train_classifier.py \
#       --lstm \
#       --dataset tnews\
#       --embedding data/sgns.weibo.word.bz2 \
#       --save_path data/tnews/lstm-0.3 \
#       --dropout 0.3 \
#       --lr 0.001\
#       --test
#
##完全同音，考虑音调:
#CUDA_VISIBLE_DEVICES=0 python train_classifier.py \
#       --lstm \
#       --dataset tnews\
#       --embedding data/sgns.weibo.word.bz2 \
#       --save_path data/tnews/lstm-0.3 \
#       --dropout 0.3 \
#       --lr 0.001\
#       --noise_type add

#CUDA_VISIBLE_DEVICES=0 python train_classifier.py \
#       --cnn \
#       --dataset tnews\
#       --embedding data/sgns.renmin.word.bz2 \
#       --save_path data/tnews/cnn-0.3 \
#       --dropout 0.3 \
#       --lr 0.001\
#       --test

#CUDA_VISIBLE_DEVICES=0 python train_classifier.py \
#       --cnn \
#       --dataset tnews\
#       --embedding data/sgns.weibo.word.bz2 \
#       --save_path data/tnews/cnn-weibo-0.3 \
#       --dropout 0.3 \
#       --lr 0.001



##完全同音，考虑音调: 0.47 , 0.44
##
#CUDA_VISIBLE_DEVICES=1 python train_classifier.py \
#       --cnn \
#       --dataset tnews\
#       --embedding data/sgns.weibo.word.bz2 \
#       --save_path data/tnews/cnn-weibo-0.3 \
#       --dropout 0.3 \
#       --lr 0.001 \
#       --noise_type add
#
#python train_classifier.py \
#       --cnn \
#       --dataset tnews\
#       --embedding data/sgns.renmin.word.bz2 \
#       --save_path data/tnews/cnn-0.3 \
#       --dropout 0.3 \
#       --lr 0.001
#
##epoch 1: 0.56, 0.54
##epoch 2: 0.57 0.56
##epoch 3: 0.56 0.56
#CUDA_VISIBLE_DEVICES=0 python run_classifier.py \
#        --data_dir ../data/tnews \
#        --bert_model bert-base-chinese \
#        --task_name tnews \
#        --output_dir results/tnews-e2 \
#        --cache_dir pytorch_cache \
#        --do_train \
#        --do_eval \
#        --do_lower_case \
#        --num_train_epochs 2

##noise_type: del, noise
CUDA_VISIBLE_DEVICES=0 python run_classifier.py \
        --data_dir ../data/tnews \
        --bert_model bert-base-chinese \
        --task_name tnews \
        --output_dir results/tnews-e4 \
        --cache_dir pytorch_cache \
        --do_resume\
        --do_eval \
        --do_lower_case \
        --num_train_epochs 4 \
        --noise_type typos

for max_iters in 100
do
{
  nohup srun -p inspur -w inspur-gpu-07 python classification_attack.py \
        --dataset_path data/tnews/test.json  \
        --word_embeddings_path data/sgns.weibo.word.bz2 \
        --target_model wordLSTM \
        --counter_fitting_cos_sim_path tnews-mat.txt \
        --target_dataset tnews \
        --target_model_path data/tnews/lstm-0.3 \
        --output_dir adv_results \
        --output_name tnews-lstm-adv-${max_iters} \
        --counter_fitting_embeddings_path tnews-counter-fitted-vectors.txt \
        --USE_cache_path " " \
        --max_seq_length 256 \
        --sim_score_window 40 \
        --nclasses 15 \
        --max_iters ${max_iters} \
        --data_size 1000 > tmp_file/tnews-lstm-adv-${max_iters}-out &
}
done

for max_iters in 100
do
{
  nohup srun -p inspur -w inspur-gpu-07 python classification_attack.py \
        --dataset_path data/tnews/test.json  \
        --word_embeddings_path data/sgns.renmin.word.bz2 \
        --target_model wordCNN \
        --counter_fitting_cos_sim_path tnews-mat.txt \
        --target_dataset tnews \
        --target_model_path data/tnews/cnn-0.3 \
        --output_dir adv_results \
        --output_name tnews-cnn-adv-${max_iters} \
        --counter_fitting_embeddings_path tnews-counter-fitted-vectors.txt \
        --USE_cache_path " " \
        --max_seq_length 256 \
        --sim_score_window 40 \
        --nclasses 15 \
        --max_iters ${max_iters} \
        --data_size 1000 > tmp_file/tnews-cnn-adv-${max_iters}-out &
}
done

for max_iters in 100
do
{
   nohup srun -p inspur -w inspur-gpu-07 python classification_attack.py \
        --bert_model bert-base-chinese\
        --dataset_path data/tnews/test.txt  \
        --word_embeddings_path data/sgns.weibo.word.bz2 \
        --target_model bert \
        --counter_fitting_cos_sim_path tnews-mat.txt \
        --target_dataset tnews \
        --target_model_path BERT/results/tnews-e4 \
        --output_dir adv_results \
        --output_name tnews-bert-adv-${max_iters} \
        --counter_fitting_embeddings_path  tnews-counter-fitted-vectors.txt \
        --USE_cache_path " " \
        --max_seq_length 128 \
        --sim_score_window 40 \
        --max_iters ${max_iters} \
        --nclasses 15 \
        --data_size 1000 > tmp_file/tnews-bert-adv-${max_iters}-out &
}
done