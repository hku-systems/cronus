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

# Utility rule file for install-cxx.

# Include the progress variables for this target.
include src/CMakeFiles/install-cxx.dir/progress.make

src/CMakeFiles/install-cxx: lib/libc++experimental.a
	cd /home/jianyu/optee/optee_os/lib/libcxx/build/src && /usr/local/bin/cmake -DCMAKE_INSTALL_COMPONENT=cxx -P /home/jianyu/optee/optee_os/lib/libcxx/build/cmake_install.cmake

install-cxx: src/CMakeFiles/install-cxx
install-cxx: src/CMakeFiles/install-cxx.dir/build.make

.PHONY : install-cxx

# Rule to build all files generated by this target.
src/CMakeFiles/install-cxx.dir/build: install-cxx

.PHONY : src/CMakeFiles/install-cxx.dir/build

src/CMakeFiles/install-cxx.dir/clean:
	cd /home/jianyu/optee/optee_os/lib/libcxx/build/src && $(CMAKE_COMMAND) -P CMakeFiles/install-cxx.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/install-cxx.dir/clean

src/CMakeFiles/install-cxx.dir/depend:
	cd /home/jianyu/optee/optee_os/lib/libcxx/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jianyu/optee/optee_os/lib/libcxx/libcxx /home/jianyu/optee/optee_os/lib/libcxx/libcxx/src /home/jianyu/optee/optee_os/lib/libcxx/build /home/jianyu/optee/optee_os/lib/libcxx/build/src /home/jianyu/optee/optee_os/lib/libcxx/build/src/CMakeFiles/install-cxx.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/install-cxx.dir/depend
