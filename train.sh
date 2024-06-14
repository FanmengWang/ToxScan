# Copyright (c) DP Technology.
# This source code is licensed under the GPL-3.0 license found in the
# LICENSE file in the root directory of this source tree.

for seed_value in {0..9}
       do
              data_path='./'
              save_dir="./drug_toxicity_ckpt/drug_toxicity_ckpt_epoch_${seed_value}"
              n_gpu=1
              MASTER_PORT=7000
              dict_name='dict.txt'
              task_name='dataset'
              lr=1e-4
              batch_size=64
              epoch=50
              dropout=0.1
              warmup=0.06
              local_batch_size=64
              only_polar=0
              conf_size=11
              seed="${seed_value}"
              metric="valid_agg_auc"

              export NCCL_ASYNC_ERROR_HANDLING=1
              export OMP_NUM_THREADS=1
              update_freq=`expr $batch_size / $local_batch_size`
              python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path \
                     --task-name $task_name --user-dir ./ToxScan \
                     --train-subset train --valid-subset valid \
                     --conf-size $conf_size \
                     --num-workers 8 --ddp-backend=c10d \
                     --dict-name $dict_name \
                     --task ToxScan --loss ToxScan --arch ToxScan_base \
                     --classification-head-name $task_name \
                     --optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm 1.0 \
                     --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch \
                     --batch-size $local_batch_size --pooler-dropout $dropout\
                     --update-freq $update_freq --seed $seed \
                     --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
                     --tensorboard-logdir $save_dir/tsb --tmp-save-dir $save_dir/tmp \
                     --log-interval 100 --log-format simple \
                     --validate-interval 1 --keep-last-epochs 1 \
                     --best-checkpoint-metric $metric --patience 30 \
                     --save-dir $save_dir --only-polar $only_polar \
                     --maximize-best-checkpoint-metric \
                     --all-gather-list-size 65536
       done