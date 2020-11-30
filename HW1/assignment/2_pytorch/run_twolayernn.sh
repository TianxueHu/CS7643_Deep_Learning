#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model twolayernn \
    --hidden-dim 48 \
    --epochs 6 \
    --weight-decay 0.95 \
    --momentum 0.9 \
    --batch-size 512 \
    --lr 0.0001 | tee twolayernn.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
