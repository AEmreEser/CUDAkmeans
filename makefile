# Cuda Kmeans - A. Emre Eser
CC = g++
CXXFLAGS = -O3 -Wall -fopenmp
NVCC = /usr/local/cuda/bin/nvcc
arch?=sm_61 # my gpu
NVCCFLAGS = -O3 -arch=$(arch) # Adjust the architecture based on your GPU's compute capability
addflags?= # options: HOSTREDUCE, DEBUG, ALIGNED_MEM (does not increase the performance all that much)
k?=4
file?=input.txt
EXE ?= kmeans
SRC = $(EXE).cu
OBJ = $(EXE).o

.PHONY: debug, clean, run, all, hostreduce, rerun

all: $(EXE)

$(EXE): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) $(addflags) -Xcompiler "$(CXXFLAGS)" -o $(EXE) 

# obsolete:
# bsize: $(SRC)
# 	make $(EXE) addflags="-DBLOCKSIZE=$(k)"

# brun: clean bsize run

debug: $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -DDEBUG -Xcompiler "$(CXXFLAGS)" -o $(EXE) 

rerun: clean run

clean:
	-rm -f $(EXE)

run: $(EXE)
	./$(EXE) $(file) $(k)

hostreduce: $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -DHOSTREDUCE -Xcompiler "$(CXXFLAGS)" -o $(EXE) 
