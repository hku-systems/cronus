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

# Utility rule file for cxx-benchmarks.

# Include the progress variables for this target.
include benchmarks/CMakeFiles/cxx-benchmarks.dir/progress.make

cxx-benchmarks: benchmarks/CMakeFiles/cxx-benchmarks.dir/build.make

.PHONY : cxx-benchmarks

# Rule to build all files generated by this target.
benchmarks/CMakeFiles/cxx-benchmarks.dir/build: cxx-benchmarks

.PHONY : benchmarks/CMakeFiles/cxx-benchmarks.dir/build

benchmarks/CMakeFiles/cxx-benchmarks.dir/clean:
	cd /home/jianyu/optee/optee_os/lib/libcxx/build/benchmarks && $(CMAKE_COMMAND) -P CMakeFiles/cxx-benchmarks.dir/cmake_clean.cmake
.PHONY : benchmarks/CMakeFiles/cxx-benchmarks.dir/clean

benchmarks/CMakeFiles/cxx-benchmarks.dir/depend:
	cd /home/jianyu/optee/optee_os/lib/libcxx/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jianyu/optee/optee_os/lib/libcxx/libcxx /home/jianyu/optee/optee_os/lib/libcxx/libcxx/benchmarks /home/jianyu/optee/optee_os/lib/libcxx/build /home/jianyu/optee/optee_os/lib/libcxx/build/benchmarks /home/jianyu/optee/optee_os/lib/libcxx/build/benchmarks/CMakeFiles/cxx-benchmarks.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : benchmarks/CMakeFiles/cxx-benchmarks.dir/depend
