Name: TianxueHu (thu82 on EvalAI) <br>
Email: thu82@gatech.edu <br>
Best prediction accuracy: 66.1% 
<br>

Architecture: Pretrained torchvision model Resnet18 at https://pytorch.org/docs/stable/torchvision/models.html 
<br>
Parameters: 
<br>
    --kernel-size 1 \
    --hidden-dim 15 \
    --epochs 10 \
    --weight-decay 0.001 \
    --momentum 0.9 \
    --batch-size 64 \
    --lr 0.00005 |
<br>
Optimization: SGD
<br>
Validation scores: Average loss: 0.4287, Accuracy: 8505/10000 (85%)
<br>

At forward step, using F.upsample() to resample image at (128,128), I was trying to use larger sizes but unfortunately Google Colab doesn't have enough GPU memory.