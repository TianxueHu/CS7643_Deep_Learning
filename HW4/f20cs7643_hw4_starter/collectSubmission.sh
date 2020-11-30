rm -f hw4.zip
jupyter-nbconvert --to pdf "*.ipynb"
zip -r hw4.zip . -x "*.git" "*cs231n/datasets*" "*.ipynb_checkpoints*" "*README.md" "*collectSubmission.sh" "*requirements.txt" ".env/*" "*.pyc" "*cs231n/build/*" "gt_7643/datasets*" "*imgs/*" "*__pycache__*"
