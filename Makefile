CXXFLAGS := -std=c++17 $(shell pkg-config --cflags opencv4)
LDLIBS := $(shell pkg-config --libs opencv4)

build/main: src/main.cpp | build
	g++ -o build/main $(CXXFLAGS) src/main.cpp $(LDLIBS)

build:
	mkdir -p build

run: build/main
	./build/main cones/im2.png cones/im2.png

.PHONY: run
