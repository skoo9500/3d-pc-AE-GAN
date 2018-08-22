TF_LIB= /home/ymkim/.virtualenvs/virtual-py3/lib/python3.5/site-packages/tensorflow/
nvcc = /usr/local/cuda-9.0/bin/nvcc/
cudalib = /usr/local/cuda-9.0/lib64/
tensorflow = /home/ymkim/.virtualenvs/virtual-py3/lib/python3.5/site-packages/tensorflow/include/

all: grouping/tf_grouping_so.so
.PHONY : all

grouping/tf_grouping_so.so : grouping/tf_grouping_g.cu.o grouping/tf_grouping.cpp
	g++ -std=c++11 grouping/tf_grouping.cpp grouping/tf_grouping_g.cu.o -o grouping/tf_grouping_so.so -shared -fPIC -I $(tensorflow) -ltensorflow_framework -L $(TF_LIB) -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -isystem /usr/local/cuda-9.0/include

grouping/tf_grouping_g.cu.o: grouping/tf_grouping_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o grouping/tf_grouping_g.cu.o grouping/tf_grouping_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2
