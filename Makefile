all:
	test -d build || mkdir build
	cd build && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/usr/local/Cellar/opencv3/3.2.0/share .. && make

debug:
	test -d build || mkdir build
	cd build && cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=/usr/local/Cellar/opencv3/3.2.0/share .. && make

clean:
	rm -rf build

.PHONY: all debug clean
