# How to write a tensorflow custom operation

	A simple example is in "zero_out.cc"

# To compile it it will follow steps like:
	
	TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
	TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

	g++ -std=c++11 -shared -undefined dynamic_lookup zero_out.cc -o zero_out.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2


	-undefined dynamic_lookup is required in MacOS