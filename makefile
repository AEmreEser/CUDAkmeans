CC = g++
CXXFLAGS = -O3 -Wall -fopenmp
NVCC = /usr/bin/nvcc
arch?=sm_50 # my gpu
NVCCFLAGS = -O3 -arch=$(arch) # Adjust the architecture based on your GPU's compute capability
k?=4
file?=input.txt
SRC = kmeans.cu
OBJ = kmeans.o
EXE = kmeans

all: $(EXE)

$(EXE): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -Xcompiler "$(CXXFLAGS)" -o $(EXE) 

clean:
	-rm -f $(EXE)

run: $(EXE)
	./$(EXE) $(file) $(k)

