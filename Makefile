# Compiler and flags
NVCC := nvcc
CPPFLAGS := $(addprefix -I ,$(shell find include -type d))
CXXFLAGS = -arch $(ARCH) -rdc=true -O2
XCOMPILER_FLAGS := -Xcompiler -fopenmp -Xptxas -v
LDFLAGS := -dlto

# Directories
SRC_DIR := src
INC_DIR := include
BUI_DIR := build
TEST_DIR := test

# Source files and object files
OBJ_DIR := $(BUI_DIR)/object
BIN_DIR := $(BUI_DIR)/bin

# Source files and object files
SRC_FILES := $(shell find $(SRC_DIR) -type f -name "*.cu")  # All .cu files in src directory
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRC_FILES))

# Test programs
TEST_SRC_FILES := $(wildcard $(TEST_DIR)/*.cu)  # All .cu files in test directory
TEST_EXEC_FILES := $(patsubst $(TEST_DIR)/%.cu, $(BIN_DIR)/%, $(TEST_SRC_FILES))

# Targets
all: $(TEST_EXEC_FILES)

# Rule for compiling .cu files into .o files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(CPPFLAGS) $(CXXFLAGS) $(XCOMPILER_FLAGS) $(LDFLAGS) -c $< -o $@

# Rule for compiling test programs
$(BIN_DIR)/%: $(TEST_DIR)/%.cu $(OBJ_FILES)
	@mkdir -p $(dir $@)
	$(NVCC) $(CPPFLAGS) $(CXXFLAGS) $(XCOMPILER_FLAGS) $(LDFLAGS) $< $(OBJ_FILES) -o $@

# Default program
PROG ?= multi-gpu

# Run a specific test program
run: $(BIN_DIR)/$(PROG)
	./$(BIN_DIR)/$(PROG)

# Profile using nvprof (available for old GPU architectures)
pf-old: $(BIN_DIR)/$(PROG)
	nvprof ./$(BIN_DIR)/$(PROG)

clean:
	rm -rf $(BUI_DIR)

.PHONY: all clean

-include config.mk