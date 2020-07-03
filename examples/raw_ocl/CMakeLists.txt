cmake_minimum_required(VERSION 3.13.4)

project(raw_OpenCL_examples)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

add_library(cl_check_err OBJECT
  cl_check_err.c
)

set(EXAMPLES
    cl_platform_ls
    vec_add
    matrix_mult
)

set(EXAMPLES_WITH_KERNELS
    vec_add
    matrix_mult
)

foreach(EXAMPLE_NAME IN LISTS EXAMPLES)
  set(EXEC_NAME ${EXAMPLE_NAME})
  set(SRC_NAME ${EXAMPLE_NAME}.c)
  add_executable(${EXEC_NAME} ${SRC_NAME} $<TARGET_OBJECTS:cl_check_err>)
  if(${EXAMPLE_NAME} IN_LIST EXAMPLES_WITH_KERNELS)
    set(KERNEL_SRC_NAME ../${EXAMPLE_NAME}_kernel.cl)
    target_compile_definitions(${EXEC_NAME} PRIVATE STD_KERNEL_FILENAME=\"${KERNEL_SRC_NAME}\")
  endif()
  target_link_libraries(${EXEC_NAME} ${OpenCL_LIBRARIES})
endforeach()

