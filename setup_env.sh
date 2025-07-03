pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install cython
echo -e " installing required packages" 
pip install -r requirements.txt
echo -e " installing pyxclib" 
pip install --upgrade pybind11 #hotfix for nmslib fails on python 3.11
pip install --verbose  'nmslib @ git+https://github.com/nmslib/nmslib.git#egg=nmslib&subdirectory=python_bindings'
pip install git+https://github.com/kunaldahiya/pyxclib
