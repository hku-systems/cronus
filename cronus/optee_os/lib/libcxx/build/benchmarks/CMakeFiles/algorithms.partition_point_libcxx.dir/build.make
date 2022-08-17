# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jianyu/optee/optee_os/lib/libcxx/libcxx

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jianyu/optee/optee_os/lib/libcxx/build

# Include any dependencies generated for this target.
include benchmarks/CMakeFiles/algorithms.partition_point_libcxx.dir/depend.make

# Include the progress variables for this target.
include benchmarks/CMakeFiles/algorithms.partition_point_libcxx.dir/progress.make

# Include the compile flags for this target's objects.
include benchmarks/CMakeFiles/algorithms.partition_point_libcxx.dir/flags.make

benchmarks/CMakeFiles/algorithms.partition_point_libcxx.dir/algorithms.partition_point.bench.cpp.o: benchmarks/CMakeFiles/algorithms.partition_point_libcxx.dir/flags.make
benchmarks/CMakeFiles/algorithms.partition_point_libcxx.dir/algorithms.partition_point.bench.cpp.o: /home/jianyu/optee/optee_os/lib/libcxx/libcxx/benchmarks/algorithms.partition_point.bench.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jianyu/optee/optee_os/lib/libcxx/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object benchmarks/CMakeFiles/algorithms.partition_point_libcxx.dir/algorithms.partition_point.bench.cpp.o"
	cd /home/jianyu/optee/optee_os/lib/libcxx/build/benchmarks && /home/jianyu/optee/toolchains/aarch64/bin/aarch64-none-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/algorithms.partition_point_libcxx.dir/algorithms.partition_point.bench.cpp.o -c /home/jianyu/optee/optee_os/lib/libcxx/libcxx/benchmarks/algorithms.partition_point.bench.cpp

benchmarks/CMakeFiles/algorithms.partition_point_libcxx.dir/algorithms.partition_point.bench.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/algorithms.partition_point_libcxx.dir/algorithms.partition_point.bench.cpp.i"
	cd /home/jianyu/optee/optee_os/lib/libcxx/build/benchmarks && /home/jianyu/optee/toolchains/aarch64/bin/aarch64-none-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jianyu/optee/optee_os/lib/libcxx/libcxx/benchmarks/algorithms.partition_point.bench.cpp > CMakeFiles/algorithms.partition_point_libcxx.dir/algorithms.partition_point.bench.cpp.i

benchmarks/CMakeFiles/algorithms.partition_point_libcxx.dir/algorithms.partition_point.bench.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/algorithms.partition_point_libcxx.dir/algorithms.partition_point.bench.cpp.s"
	cd /home/jianyu/optee/optee_os/lib/libcxx/build/benchmarks && /home/jianyu/optee/toolchains/aarch64/bin/aarch64-none-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jianyu/optee/optee_os/lib/libcxx/libcxx/benchmarks/algorithms.partition_point.bench.cpp -o CMakeFiles/algorithms.partition_point_libcxx.dir/algorithms.partition_point.bench.cpp.s

# Object files for target algorithms.partition_point_libcxx
algorithms_partition_point_libcxx_OBJECTS = \
"CMakeFiles/algorithms.partition_point_libcxx.dir/algorithms.partition_point.bench.cpp.o"

# External object files for target algorithms.partition_point_libcxx
algorithms_partition_point_libcxx_EXTERNAL_OBJECTS =

benchmarks/algorithms.partition_point.libcxx.out: benchmarks/CMakeFiles/algorithms.partition_point_libcxx.dir/algorithms.partition_point.bench.cpp.o
benchmarks/algorithms.partition_point.libcxx.out: benchmarks/CMakeFiles/algorithms.partition_point_libcxx.dir/build.make
benchmarks/algorithms.partition_point.libcxx.out: lib/libc++.a
benchmarks/algorithms.partition_point.libcxx.out: lib/libc++experimental.a
benchmarks/algorithms.partition_point.libcxx.out: lib/libc++.a
benchmarks/algorithms.partition_point.libcxx.out: benchmarks/CMakeFiles/algorithms.partition_point_libcxx.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jianyu/optee/optee_os/lib/libcxx/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable algorithms.partition_point.libcxx.out"
	cd /home/jianyu/optee/optee_os/lib/libcxx/build/benchmarks && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/algorithms.partition_point_libcxx.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
benchmarks/CMakeFiles/algorithms.partition_point_libcxx.dir/build: benchmarks/algorithms.partition_point.libcxx.out

.PHONY : benchmarks/CMakeFiles/algorithms.partition_point_libcxx.dir/build

benchmarks/CMakeFiles/algorithms.partition_point_libcxx.dir/clean:
	cd /home/jianyu/optee/optee_os/lib/libcxx/build/benchmarks && $(CMAKE_COMMAND) -P CMakeFiles/algorithms.partition_point_libcxx.dir/cmake_clean.cmake
.PHONY : benchmarks/CMakeFiles/algorithms.partition_point_libcxx.dir/clean

benchmarks/CMakeFiles/algorithms.partition_point_libcxx.dir/depend:
	cd /home/jianyu/optee/optee_os/lib/libcxx/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jianyu/optee/optee_os/lib/libcxx/libcxx /home/jianyu/optee/optee_os/lib/libcxx/libcxx/benchmarks /home/jianyu/optee/optee_os/lib/libcxx/build /home/jianyu/optee/optee_os/lib/libcxx/build/benchmarks /home/jianyu/optee/optee_os/lib/libcxx/build/benchmarks/CMakeFiles/algorithms.partition_point_libcxx.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : benchmarks/CMakeFiles/algorithms.partition_point_libcxx.dir/depend
