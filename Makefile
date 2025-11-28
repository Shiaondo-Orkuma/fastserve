# FastServe v0.1 Makefile
# Neural Network Inference Engine

CXX = g++
CXXFLAGS = -O3 -march=native -mavx2 -mfma -std=c++17 -Wall -Wextra

# Source files
SRCS = main.cpp nn/layers.cpp nn/model.cpp nn/loader.cpp
OBJS = $(SRCS:.cpp=.o)

# Output
TARGET = fastserve

# Default target
all: $(TARGET)

# Link
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Dependencies (simplified)
main.o: main.cpp nn/model.h nn/loader.h core/tensor.h
nn/layers.o: nn/layers.cpp nn/layers.h core/tensor.h core/simd_matmul.h
nn/model.o: nn/model.cpp nn/model.h nn/layers.h core/tensor.h
nn/loader.o: nn/loader.cpp nn/loader.h core/tensor.h

# Clean
clean:
	rm -f $(OBJS) $(TARGET)

# Install (copy to parent directory for easy access)
install: $(TARGET)
	cp $(TARGET) ../fastserve

# Run inference on test images
run: $(TARGET)
	./$(TARGET) --model ../models/tiny_cnn_weights.txt --images ../models/test_images.txt --verbose

# Run benchmark
bench: $(TARGET)
	./$(TARGET) --model ../models/tiny_cnn_weights.txt --benchmark

# Help
help:
	@echo "FastServe v0.1 Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build fastserve (default)"
	@echo "  clean    - Remove build artifacts"
	@echo "  install  - Copy binary to parent directory"
	@echo "  run      - Run inference on test images"
	@echo "  bench    - Run benchmark"
	@echo "  help     - Show this message"

.PHONY: all clean install run bench help
