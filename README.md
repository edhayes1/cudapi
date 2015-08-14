# cudapi (linux only)

compile .cu files using nvcc. Comes with the nvidia toolkit version 7.
you'll need the -std=c++11 flag for both the nvcc and g++.


### Compilation:

First you need to install linux CUDA drivers and the Nvidia toolkit:
http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/#axzz3ihWNglZK

on ubuntu, it is easiest to download the "run" installer, and execute that, follow the on-screen instructions (after reading the documentation on the link, if you mess it up, your X server may break... Just like mine did...).

once you have the cuda driver and toolkit installed, try:
```
$ nvcc --version
```
For you to compile these samples, you need at least Cuda compilation tools release 7.0.

```
$ nvcc -std=c++11 cudafile.cu -o cudafile.exe -lcurand
```
will compile the Cuda programs.

```
$ g++ -std=c++11 cppfile.cpp -o cppfile.exe
```

will compile the cpp programs.

replace "cudafile.cu" with the name of the file you are compiling.
This could be the provided "test.cu" or "dualtest.cu" files.

Likewise for the "cppfile.cpp" you can choose either "20threads.cpp" or "singleC.cpp"
I should not need to speculate over what would happen if you attempted to run "20threads.cpp" without having 20 threads. However its very easy to change the number of threads in the source code (change the int decleration on line 13).
