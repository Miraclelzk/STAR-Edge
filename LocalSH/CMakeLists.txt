cmake_minimum_required(VERSION 3.4)

project(LocalSH)
file( GLOB header_list *.h )
file( GLOB source_list *.cpp *.c)


project(${PROJECT_NAME} LANGUAGES CXX)

####pybind11####
add_subdirectory(pybind11)
pybind11_add_module(${PROJECT_NAME} ${header_list} ${source_list})

target_include_directories( ${PROJECT_NAME}
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory( nanoflann )
add_subdirectory( S2kit11 )
target_link_libraries( ${PROJECT_NAME} PRIVATE S2kit11 )

target_compile_definitions( ${PROJECT_NAME} PRIVATE _USE_MATH_DEFINES)

set(CMAKE_INSTALL_RPATH "/usr/local/lib/")

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/..)