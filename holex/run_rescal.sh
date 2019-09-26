
python ProjE_softmax_rescal.py --dim 64 --batch 128 --data ./data/FB15k/ --eval_per 1 --worker 24 --eval_batch 500 --max_iter 50 --generator 10 --lr 5e-4 --neg_weight 0.1 --dc 64 --haar 4  --save_dir waste/ --load_model ./sparsity-dim64-models/ProjE_DEFAULT_49.ckpt
 
