conda create -n robopoint python=3.10 -y
conda activate robopoint

pip install --upgrade pip  # enable PEP 660 support

# this is optional if you prefer to system built-in nvcc.
conda install -c nvidia cuda=12.1 -y

pip install -e .

# this is optional if you don't need to train the model
pip install -e ".[train]"
pip install flash-attn --no-build-isolation