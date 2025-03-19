NVCC = nvcc
NVCC_FLAGS = -Wno-deprecated-gpu-targets  
SRCS = $(wildcard *.cu)                    
TARGETS   = $(SRCS:.cu=)

# Detect OS
ifeq ($(OS),Windows_NT)
    RM = del /Q
    OBJ_PATTERN = *.exp
	LIB_PATTERN = *.lib
	EXE_PATTERN = *.exe
else
    RM = rm -f
    OBJ_PATTERN = *.o
	EXE_PATTERN = $(TARGETS)
endif

all: $(TARGETS)

$(TARGETS): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

clean:
	@$(RM) $(TARGETS) $(OBJ_PATTERN) $(LIB_PATTERN) $(EXE_PATTERN) 2>nul || exit 0

.PHONY: all clean
