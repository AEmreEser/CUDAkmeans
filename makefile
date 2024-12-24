CC = g++
CXXFLAGS = -O3 -Wall -fopenmp
NVCC = /usr/local/cuda/bin/nvcc
arch?=sm_61 # my gpu
NVCCFLAGS = -O3 -arch=$(arch) -DPINMEM # Adjust the architecture based on your GPU's compute capability
addflags?= # options: DEBUG, ALIGNED_MEM (does not increase the performance all that much)
k?=4
file?=input.txt
SRC = kmeans.cu
OBJ = kmeans.o
EXE = kmeans

.PHONY: debug, clean, run, all

all: $(EXE)

$(EXE): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) $(addflags) -Xcompiler "$(CXXFLAGS)" -o $(EXE) 


debug: $(SRC)
	make $(EXE) addflags=-DDEBUG

clean:
	-rm -f $(EXE)

run: $(EXE)
	./$(EXE) $(file) $(k)

