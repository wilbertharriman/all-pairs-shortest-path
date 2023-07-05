NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61
LDFLAGS  := -lm

a.out: main.cu
        nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?
