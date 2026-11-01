build/main: src/main.cpp | build
	g++ -o build/main src/main.cpp

build:
	mkdir -p build

run: build/main
	./build/main cones/im2.png cones/im2.png

.PHONY: run
