rm -f hw2_code.zip
rm -rf 1_cs231n/cs231n/datasets/cifar*
rm -rf 2_pytorch/cs231n/data/cifar*
jupyter-nbconvert --to pdf "1_cs231n/*.ipynb"
jupyter-nbconvert --to pdf "2_pytorch/*.ipynb"
zip -r hw2_code.zip . -x "*.git*" "*cs231n/datasets*" "*.ipynb_checkpoints*" "*README.md" "*collect_submission.sh" "*requirements.txt" ".env/*" "*.pyc" "*cs231n/build/*" "__pycache__/*" ".*" "__MACOSX"
