####for weibo2018
#python naive_bayes_classifier.py weibo2018 add
#
#
CUDA_VISIBLE_DEVICES=0 python train_classifier.py \
       --cnn \
       --dataset weibo2018\
       --embedding data/sgns.weibo.word.bz2 \
       --save_path data/weibo2018/cnn-0.3 \
       --dropout 0.3 \
       --lr 0.0005 \
       --noise_type add

CUDA_VISIBLE_DEVICES=1 python train_classifier.py \
       --lstm \
       --dataset weibo2018\
       --embedding data/sgns.weibo.word.bz2 \
       --save_path data/weibo2018/lstm-0.3 \
       --dropout 0.3 \
       --lr 0.001 \
       --noise_type add
#
##epoch 3: 0.884
##epoch 4: 0.906
##epoch 6: 0.92 0.90
##epoch 7: 0.91 0.90
#python run_classifier.py \
#        --data_dir ../data/weibo2018 \
#        --bert_model bert-base-chinese \
#        --task_name weibo2018 \
#        --output_dir results/weibo2018-e3 \
#        --cache_dir pytorch_cache \
#        --do_train \
#        --do_eval \
#        --do_lower_case \
#        --num_train_epochs 3

CUDA_VISIBLE_DEVICES=0 python run_classifier.py \
        --data_dir ../data/weibo2018 \
        --bert_model bert-base-chinese \
        --task_name weibo2018 \
        --output_dir results/weibo2018-e6 \
        --cache_dir pytorch_cache \
        --do_resume \
        --do_eval \
        --do_lower_case \
        --num_train_epochs 6\
        --noise_type add


#for max_iters in 100
#do
#{
#  nohup srun -p inspur -w inspur-gpu-05 python classification_attack.py \
#        --dataset_path data/weibo2018/test.txt  \
#        --word_embeddings_path data/sgns.weibo.word.bz2 \
#        --target_model wordCNN \
#        --counter_fitting_cos_sim_path weibo2018-mat.txt \
#        --target_dataset weibo2018 \
#        --target_model_path data/weibo2018/cnn-0.3 \
#        --output_dir adv_results \
#        --output_name weibo2018-cnn-adv-${max_iters} \
#        --counter_fitting_embeddings_path weibo2018-counter-fitted-vectors.txt \
#        --USE_cache_path " " \
#        --max_seq_length 128 \
#        --sim_score_window 40 \
#        --nclasses 2 \
#        --max_iters ${max_iters} \
#        --data_size 500 > tmp_file/weibo2018-cnn-adv-${max_iters}-out &
#}
#done

#For target model using TFIDF wordLSTM on dataset window size 40 with WP val 0 top words 1000000 qrs 1000000
#: original accuracy: 88.200%, adv accuracy: 63.600%, random avg  change: 5.947% avg changed rate: 5.484%,
#num of queries: 145.2, random_sims: 0.800%, final_sims : 0.817%
#for max_iters in 100
#do
#{
#  nohup srun -p inspur -w inspur-gpu-05 python classification_attack.py \
#        --dataset_path data/weibo2018/test.txt  \
#        --word_embeddings_path data/sgns.weibo.word.bz2 \
#        --target_model wordLSTM \
#        --counter_fitting_cos_sim_path weibo2018-mat.txt \
#        --target_dataset weibo2018 \
#        --target_model_path data/weibo2018/lstm-0.3 \
#        --output_dir adv_results \
#        --output_name weibo2018-lstm-adv-${max_iters} \
#        --counter_fitting_embeddings_path weibo2018-counter-fitted-vectors.txt \
#        --USE_cache_path " " \
#        --max_seq_length 128 \
#        --sim_score_window 40 \
#        --nclasses 2 \
#        --max_iters ${max_iters} \
#        --data_size 500 \
#        --cuda_idx 1 > tmp_file/weibo2018-lstm-adv-${max_iters}-out &
#}
#done


##nohup srun -p inspur -w inspur-gpu-05
for max_iters in 100
do
{
  nohup srun -p inspur -w inspur-gpu-07 python classification_attack.py \
        --bert_model bert-base-chinese\
        --dataset_path data/weibo2018/test.txt  \
        --word_embeddings_path data/sgns.weibo.word.bz2 \
        --target_model bert \
        --counter_fitting_cos_sim_path weibo2018-mat.txt \
        --target_dataset weibo2018 \
        --target_model_path BERT/results/weibo2018-e6 \
        --output_dir adv_results \
        --output_name weibo2018-bert-adv-${max_iters} \
        --counter_fitting_embeddings_path  weibo2018-counter-fitted-vectors.txt \
        --USE_cache_path " " \
        --max_seq_length 128 \
        --sim_score_window 40 \
        --max_iters ${max_iters} \
        --nclasses 2 \
        --data_size 500 > tmp_file/weibo2018-bert-adv-${max_iters}-out &
}
done