project (rodinia_bfs C)

set (SRC host/main.c)

add_executable (${PROJECT_NAME} ${SRC})

target_include_directories(${PROJECT_NAME}
			   PRIVATE cpumenclave/include
			   PRIVATE include)

target_link_libraries (${PROJECT_NAME} PRIVATE teec)

install (TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR})
install (DIRECTORY data/ DESTINATION ${CMAKE_INSTALL_BINDIR}/../../data/tee/rodinia)
