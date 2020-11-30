#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model mymodel \
    --kernel-size 1 \
    --hidden-dim 15 \
    --epochs 10 \
    --weight-decay 0.001 \
    --momentum 0.9 \
    --batch-size 64 \
    --lr 0.00005 | tee mymodel.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
