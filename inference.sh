# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

for seed_value in {0..9}
       do
              data_path='./'
              results_path="./infer_drug_toxicity_results/infer_drug_toxicity_results_epoch_${seed_value}"
              weight_path="./drug_toxicity_ckpt/drug_toxicity_ckpt_epoch_${seed_value}/checkpoint_best.pt"
              batch_size=256
              task_name='dataset'
              dict_name='dict.txt'
              conf_size=11
              only_polar=0
              
              export CUDA_VISIBLE_DEVICES=0
              python ./ToxScan/infer.py --user-dir ./ToxScan $data_path --task-name $task_name --valid-subset test \
                     --results-path $results_path \
                     --num-workers 1 --ddp-backend=c10d --batch-size $batch_size \
                     --task ToxScan --loss ToxScan --arch ToxScan_base \
                     --classification-head-name $task_name \
                     --dict-name $dict_name --conf-size $conf_size \
                     --only-polar $only_polar  \
                     --path $weight_path  \
                     --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
                     --log-interval 100 --log-format simple 
       done