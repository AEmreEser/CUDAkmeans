CC = g++
CXXFLAGS = -O3 -Wall -fopenmp
NVCC = /usr/local/cuda/bin/nvcc
arch?=sm_61 # my gpu
NVCCFLAGS = -O3 -arch=$(arch) # Adjust the architecture based on your GPU's compute capability
k?=4
file?=input.txt
SRC = kmeans.cu
OBJ = kmeans.o
EXE = kmeans

all: $(EXE)

$(EXE): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) $(addflags) -Xcompiler "$(CXXFLAGS)" -o $(EXE) 

.PHONY: debug

debug: $(SRC)
	make $(EXE) addflags=-DDEBUG

clean:
	-rm -f $(EXE)

run: $(EXE)
	./$(EXE) $(file) $(k)

