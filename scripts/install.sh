# python 3.8
pip install --no-cache-dir torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -r requirements.txt --no-cache-dir
pip install future tensorboard
pip install grad-cam==1.3.6

pip install -U openmim
cd mmpretrain
mim install -e .
cd -

pip install onnxruntime
pip install onnxconverter_common