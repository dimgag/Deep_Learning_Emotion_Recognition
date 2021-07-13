
Architectures described in architecture.py

Examples for running python code:

Training of model 5 for 200 epochs:
python train.py 5 200

Evaluate quality of model 5, available model 0-6:
python evaluate.py 5

Plot feature maps for layers in model 5:
python visualize_layers.py 5

Other python scripts are helper scripts to 
pre-process data, analysis, make augmented data

Environment set up:

conda create --name CV3_8 python=3.8

conda activate CV3_8

conda install -c anaconda numpy
conda install -c anaconda matplotlib
conda install -c anaconda opencv
conda install -c anaconda pycodestyle
conda install -c anaconda scipy
conda install -c pytorch  pytorch
conda install -c anaconda scikit-learn