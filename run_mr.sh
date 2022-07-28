###mr
#python naive_bayes_classifier.py mr_all train

#python train_classifier.py \
#       --cnn \
#       --dataset mr\
#       --embedding data/glove.6B.200d.txt \
#       --save_path data/mr_all/cnn-mr-0.3-1 \
#       --dropout 0.3 \
#       --lr 0.001
#
#
#python train_classifier.py \
#       --lstm \
#       --dataset mr\
#       --embedding data/glove.6B.200d.txt \
#       --save_path data/mr_all/lstm-mr-0.3-1 \
#       --dropout 0.3 \
#       --lr 0.001
#
#
#CUDA_VISIBLE_DEVICES=1 python train_classifier.py \
#       --cnn \
#       --dataset mr\
#       --embedding data/glove.6B.200d.txt \
#       --save_path data/mr_all/cnn-mr-0.3 \
#       --dropout 0.3 \
#       --lr 0.001\
#       --noise_type add
#
#
#python train_classifier.py \
#       --lstm \
#       --dataset mr\
#       --embedding data/glove.6B.200d.txt \
#       --save_path data/mr_all/lstm-mr-0.3 \
#       --dropout 0.3 \
#       --lr 0.0005 \
#       --test
#
#CUDA_VISIBLE_DEVICES=0 python train_classifier.py \
#       --lstm \
#       --dataset mr\
#       --embedding data/glove.6B.200d.txt \
#       --save_path data/mr_all/lstm-mr-0.3 \
#       --dropout 0.3 \
#       --lr 0.0005\
#       --noise_type add

#python run_classifier.py \
#          --data_dir ../data/mr_all  \
#          --bert_model bert-base-uncased  \
#          --task_name mr \
#          --output_dir ../data/mr_all/ret-3/ \
#          --cache_dir pytorch_cache \
#          --do_train --do_eval  \
#          --do_lower_case \
#          --num_train_epochs 3
#
#python run_classifier.py \
#          --data_dir ../data/mr_all  \
#          --bert_model bert-base-uncased  \
#          --task_name mr \
#          --output_dir ../data/mr_all/ret-3/ \
#          --cache_dir pytorch_cache \
#          --do_train --do_eval  \
#          --do_lower_case \
#          --num_train_epochs 3

#CUDA_VISIBLE_DEVICES=1 python run_classifier.py \
#          --data_dir ../data/mr_all  \
#          --bert_model bert-base-uncased  \
#          --task_name mr \
#          --output_dir ../data/mr_all/ret-3/ \
#          --cache_dir pytorch_cache \
#          --do_resume\
#          --do_eval  \
#          --do_lower_case \
#          --num_train_epochs 3\
#          --noise_type add

###
##we need to generate the adversarial training dataset!!!
##max_iters=10:
##For target model using TFIDF wordLSTM on dataset window size 40 with WP val 0 top words 1000000 qrs 1000000 :
## original accuracy: 79.300%, adv accuracy: -5.900%, random avg  change: 15.359%
## avg changed rate: 13.338%, num of queries: 1131.5, random_sims: 0.533%, final_sims : 0.608%
##max_iters=100
#for max_iters in 100
#do
#{
#  nohup srun -p inspur -w inspur-gpu-06 python classification_attack.py \
#        --dataset_path data/mr_all/dev.csv  \
#        --word_embeddings_path data/glove.6B.200d.txt \
#        --target_model wordLSTM \
#        --counter_fitting_cos_sim_path mat.txt \
#        --target_dataset mr \
#        --target_model_path data/mr_all/lstm-mr-0.3 \
#        --output_dir adv_results \
#        --output_name mr-lstm-adv-${max_iters} \
#        --counter_fitting_embeddings_path  counter-fitted-vectors.txt \
#        --USE_cache_path " " \
#        --max_seq_length 128 \
#        --sim_score_window 40 \
#        --nclasses 2 \
#        --max_iters ${max_iters} \
#        --data_size 1067 > tmp_file/mr-lstm-adv-${max_iters}-out &
#}
#done
#
#for max_iters in 100
#do
#{
#  nohup srun -p inspur -w inspur-gpu-06 nohup python classification_attack.py \
#        --dataset_path data/mr_all/dev.csv  \
#        --word_embeddings_path data/glove.6B.200d.txt \
#        --target_model wordCNN \
#        --counter_fitting_cos_sim_path mat.txt \
#        --target_dataset mr \
#        --target_model_path data/mr_all/cnn-mr-0.3 \
#        --output_dir adv_results \
#        --output_name mr-cnn-adv-${max_iters} \
#        --counter_fitting_embeddings_path  counter-fitted-vectors.txt \
#        --USE_cache_path " " \
#        --max_seq_length 128 \
#        --sim_score_window 40 \
#        --nclasses 2 \
#        --max_iters ${max_iters} \
#        --data_size 1067 > tmp_file/mr-cnn-adv-${max_iters}-out &
#}
#done

for max_iters in 100
do
{
  nohup srun -p inspur -w inspur-gpu-05 python classification_attack.py \
        --bert_model bert-base-uncased\
        --dataset_path data/mr_all/dev.csv \
        --word_embeddings_path glove.6B.200d.txt \
        --target_model bert \
        --counter_fitting_cos_sim_path mat.txt \
        --target_dataset mr \
        --target_model_path data/mr_all/ret-3 \
        --output_dir adv_results \
        --output_name mr-bert-adv-${max_iters} \
        --counter_fitting_embeddings_path  counter-fitted-vectors.txt \
        --USE_cache_path " " \
        --max_seq_length 256 \
        --sim_score_window 40 \
        --max_iters ${max_iters} \
        --nclasses 2 \
        --data_size 1067 > tmp_file/mr-bert-adv-${max_iters}-out &
}
done