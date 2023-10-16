python train.py --pretrain '' --resume_path '' \
    --model_name 'DSM_c48_4x' --scale_factor 4 --scale 4  \
    --epoch 301 \
    --n_steps 60 \
    --MGPU 2 \
    --batch_size 4 \
    --lr 2e-4 \
    --k_nbr 6
