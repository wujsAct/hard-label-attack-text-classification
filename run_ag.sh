###ag news
##python naive_bayes_classifier.py ag_news_all inv
#python naive_bayes_classifier.py ag_news_all train
##CUDA_VISIBLE_DEVICES=1 python train_classifier.py \
##       --cnn \
##       --dataset ag_news\
##       --embedding data/glove.6B.200d.txt \
##       --save_path data/ag_news_all/cnn-ag-0.3 \
##       --dropout 0.3 \
##       --lr 0.001\
##       --max_epoch 100
#
#CUDA_VISIBLE_DEVICES=1 python train_classifier.py \
#       --cnn \
#       --dataset ag_news\
#       --embedding data/glove.6B.200d.txt \
#       --save_path data/ag_news_all/cnn-ag-0.3 \
#       --dropout 0.3 \
#       --lr 0.001\
#       --max_epoch 100 \
#       --noise_type del

#
#CUDA_VISIBLE_DEVICES=1 python train_classifier.py \
#       --lstm \
#       --dataset ag_news\
#       --embedding data/glove.6B.200d.txt \
#       --save_path data/ag_news_all/lstm-ag-0.3 \
#       --dropout 0.3 \
#       --lr 0.001\
#       --max_epoch 100


#CUDA_VISIBLE_DEVICES=0 python train_classifier.py \
#       --lstm \
#       --dataset ag_news\
#       --embedding data/glove.6B.200d.txt \
#       --save_path data/ag_news_all/lstm-ag-0.3 \
#       --dropout 0.3 \
#       --lr 0.001\
#       --max_epoch 100\
#       --noise_type del

###
##epoch1:
##
#CUDA_VISIBLE_DEVICES=2 python run_classifier.py \
#        --data_dir ../data/ag_news_all \
#        --bert_model bert-base-uncased \
#        --task_name ag \
#        --output_dir ../data/ag_news_all/bert_ag_news-e1 \
#        --cache_dir pytorch_cache \
#        --do_train  \
#        --do_eval \
#        --do_lower_case\
#        --num_train_epochs 1\
#        --train_batch_size 32 #> bert_ag_news-e3-out &
#
#CUDA_VISIBLE_DEVICES=1 python run_classifier.py \
#        --data_dir ../data/ag_news_all \
#        --bert_model bert-base-uncased \
#        --task_name ag \
#        --output_dir ../data/ag_news_all/bert_ag_news-e1 \
#        --cache_dir pytorch_cache \
#        --do_resume  \
#        --do_eval \
#        --do_lower_case\
#        --num_train_epochs 1\
#        --train_batch_size 128 \
#        --noise_type del

# nohup srun -p inspur -w inspur-gpu-07
for max_iters in 100
do
{
     nohup srun -p inspur -w inspur-gpu-05 python classification_attack.py \
        --dataset_path data/ag_news_all/test.csv  \
        --word_embeddings_path data/glove.6B.200d.txt \
        --target_model wordLSTM \
        --counter_fitting_cos_sim_path mat.txt \
        --target_dataset ag \
        --target_model_path data/ag_news_all/lstm-ag-0.3 \
        --output_dir adv_results \
        --output_name ag-lstm-adv-${max_iters} \
        --counter_fitting_embeddings_path  counter-fitted-vectors.txt \
        --USE_cache_path " " \
        --max_seq_length 128 \
        --sim_score_window 40 \
        --nclasses 4 \
        --max_iters ${max_iters} \
        --data_size 1000 > tmp_file/ag-lstm-adv-${max_iters}-out &
}
done

#nohup srun -p inspur -w inspur-gpu-05
for max_iters in 100
do
{
  nohup srun -p inspur -w inspur-gpu-05 python classification_attack.py \
        --dataset_path data/ag_news_all/test.csv  \
        --word_embeddings_path data/glove.6B.200d.txt \
        --target_model wordCNN \
        --counter_fitting_cos_sim_path mat.txt \
        --target_dataset ag \
        --target_model_path data/ag_news_all/cnn-ag-0.3 \
        --output_dir adv_results \
        --output_name ag-cnn-adv-${max_iters} \
        --counter_fitting_embeddings_path  counter-fitted-vectors.txt \
        --USE_cache_path " " \
        --max_seq_length 128 \
        --sim_score_window 40 \
        --nclasses 4 \
        --max_iters ${max_iters} \
        --data_size 1000 > tmp_file/ag-cnn-adv-${max_iters}-out &
}
done

#nohup srun -p inspur -w inspur-gpu-05
for max_iters in 100
do
{
  nohup srun -p inspur -w inspur-gpu-05 python classification_attack.py \
        --bert_model bert-base-uncased\
        --dataset_path data/ag_news_all/dev.csv \
        --word_embeddings_path glove.6B.200d.txt \
        --target_model bert \
        --counter_fitting_cos_sim_path mat.txt \
        --target_dataset ag \
        --target_model_path data/ag_news_all/bert_ag_news-e1 \
        --output_dir adv_results \
        --output_name ag-bert-adv-${max_iters} \
        --counter_fitting_embeddings_path  counter-fitted-vectors.txt \
        --USE_cache_path " " \
        --max_seq_length 128 \
        --sim_score_window 40 \
        --max_iters ${max_iters} \
        --nclasses 4 \
        --data_size 1000 > tmp_file/ag-bert-adv-${max_iters}-out &
}
done