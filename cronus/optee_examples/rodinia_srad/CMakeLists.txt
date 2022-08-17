project (rodinia_srad C)

set (SRC host/main.c)

add_executable (${PROJECT_NAME} ${SRC})

target_include_directories(${PROJECT_NAME}
			   PRIVATE cpumenclave/include
			   PRIVATE include)

target_link_libraries (${PROJECT_NAME} PRIVATE teec -pthread)

install (TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR})
