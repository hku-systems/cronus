project (rodinia_bfs C)

set (SRC host/main.c)

add_executable (${PROJECT_NAME} ${SRC})
add_executable (check_cpu check_cpu.c)
target_include_directories(${PROJECT_NAME}
			   PRIVATE cpumenclave/include
			   PRIVATE include)

target_link_libraries (${PROJECT_NAME} PRIVATE teec -pthread)

install (TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR})
install (TARGETS check_cpu DESTINATION ${CMAKE_INSTALL_BINDIR})
install (DIRECTORY data/ DESTINATION ${CMAKE_INSTALL_BINDIR}/../../data/tee/rodinia)
