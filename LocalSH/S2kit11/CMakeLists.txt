project( S2kit11 )

add_library( ${PROJECT_NAME} STATIC )


file( GLOB header_list include/s2kit/*.h  )
file( GLOB source_list src/*c src/legendre_polynomials/*.c src/legendre_polynomials/util/*.c src/legendre_transform/*.c src/util/*.c)

target_sources( ${PROJECT_NAME}
	PRIVATE
	${header_list}
	${source_list}
)

# set(fftw3_DIR "/usr/local/lib/cmake/fftw3/")
find_package(FFTW3 REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE FFTW3::fftw3)

target_include_directories( ${PROJECT_NAME}
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/include
		${FFTW3_INCLUDE_DIRS}
)

target_compile_definitions(${PROJECT_NAME} PUBLIC COMPILE_DEFINITIONS _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS )

# OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
	message(STATUS "OpenMP found")
	target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()