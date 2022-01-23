FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.10.0-gpu-py38-cu113-ubuntu20.04-sagemaker

RUN pip3 install tensorboard

RUN pip3 install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html --force-reinstall

