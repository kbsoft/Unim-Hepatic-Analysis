name=$1
tensor="https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.0-cp35-cp35m-linux_x86_64.whl"
if [ "$name" == "gpu" ]; 
then
	tensor="https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp35-cp35m-linux_x86_64.whl"
fi

conda create -n test python=3.5 anaconda
source activate test

conda install pip
conda install geos
conda install -c conda-forge opencv
conda install -c anaconda scikit-image
conda install -c conda-forge matplotlib

pip install --ignore-installed --upgrade \
 $tensor
pip install shapely
