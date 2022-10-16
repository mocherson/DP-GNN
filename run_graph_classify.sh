

# source ~/torchv100/bin/activate
 

for fold in {0..9}
do
    python3 graph_classification.py --dataset MUTAG --hidden_dim 32 --phi MLP --device 1 --fold_idx $fold --folds 10 --lr 0.01 --agg cat --weight_decay 0.0  --wt 11001 &   # MUTAG
    python3 graph_classification.py --dataset PTC --hidden_dim 64 --phi MLP --device 5 --fold_idx $fold --folds 10 --lr 0.01 --agg cat --weight_decay 0.0  --wt 10011 &   # PTC
    python3 graph_classification.py --dataset NCI1 --hidden_dim 32 --phi MLP --device 6 --fold_idx $fold --folds 10 --lr 0.001 --agg cat --weight_decay 0.0 --first_phi --wt 11110 &   # NCI1
    python3 graph_classification.py --dataset PROTEINS --hidden_dim 16 --phi MLP --device 7 --fold_idx $fold --folds 10 --lr 0.01 --agg cat --weight_decay 0.0 --wt 11101 &   # PROTEINS
done
  

