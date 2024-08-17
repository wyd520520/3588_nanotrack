cmake_minimum_required(VERSION 3.10)

project(nanotrack_cpp)

# if (ENABLE_ASAN)
# 	message(STATUS "BUILD WITH ADDRESS SANITIZER")
# 	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
# 	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
# 	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
# endif ()

#-fPIC
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS"${CMAKE_CXX_FLAGS} -fPIC")
#set(CMAKE_BUILD_TYPE "Debug")
#################################################################pyb
# set(MY_PYBIND ${CMAKE_SOURCE_DIR}/pybind11)
# add_subdirectory(${MY_PYBIND})
# set(DEMO_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/postprocess.cc
#     ${CMAKE_CURRENT_SOURCE_DIR}/yolov6.cc
#     ${CMAKE_CURRENT_SOURCE_DIR}/mat_warper.cpp
# )
                
# # add demo module
# pybind11_add_module(${PROJECT_NAME} ${DEMO_SOURCES})
#################################################################pyb

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/utils/ utils.out)
set(CMAKE_INSTALL_RPATH "./lib")


include_directories(include)
# opencv
find_package(OpenCV REQUIRED)

# set rknn api path
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/3rdparty/rknpu2)
# set lib arch
set(LIB_ARCH aarch64)
# set rknn rt lib
set(RKNN_RT_LIB ${RKNN_API_PATH}/Linux/${LIB_ARCH}/librknnrt.so)
include_directories(${RKNN_API_PATH}/include)

# set rga path
set(RGA_PATH ${CMAKE_SOURCE_DIR}/3rdparty/librga)
# set rga lib
set(LIB_ARCH aarch64)
set(RGA_LIB ${RGA_PATH}/Linux/${LIB_ARCH}/librga.so)
include_directories( ${RGA_PATH}/include)

#eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})


#################################################################exe
add_executable(${PROJECT_NAME}
    demo_rknn.cpp
    nanotrack.cpp
)
#################################################################exe
target_link_libraries(${PROJECT_NAME} PRIVATE
    imageutils
    fileutils
    imagedrawing    
    dl
)

# link rga lib
target_link_libraries(${PROJECT_NAME}  PRIVATE ${RGA_LIB})
# link rknn rt lib
target_link_libraries(${PROJECT_NAME}  PRIVATE ${RKNN_RT_LIB})
# link opencv libraries
target_link_libraries(${PROJECT_NAME}  PRIVATE  ${OpenCV_LIBS})
# link eigen
target_link_libraries(${PROJECT_NAME}  PRIVATE  ${EIGEN3_LIBS})
message(STATUS "!!!!!!!!!!!CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME}  PRIVATE Threads::Threads)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
)

#target_compile_definitions(${PROJECT_NAME} PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO} PUBLIC)
#install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_SOURCE_DIR}/)
#install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/bus.jpg DESTINATION model)
#install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/coco_80_labels_list.txt DESTINATION model)
#file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
#install(FILES ${RKNN_FILES} DESTINATION model)