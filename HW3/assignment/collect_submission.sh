rm -f hw3_code.zip
jupyter-nbconvert --to pdf "*.ipynb"
zip -r hw3_code.zip . -x "*.ipynb_checkpoints*" "reference_output*" "*README.md" "*collect_submission.sh" ".env/*" "*.pyc" "__pycache__/*" "*.git*" "*cs7643/datasets*" "*cs7643/__pycache__*" ".*" "__MACOSX"
