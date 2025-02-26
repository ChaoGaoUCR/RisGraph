cmake_minimum_required(VERSION 3.1)
project(RisGraph CXX)

set (CMAKE_CXX_STANDARD 17)
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)
link_libraries(Threads::Threads)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# Set the default build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -DDEBUG")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0 -DDEBUG")


find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIRS})
link_directories(${TBB_LIBRARY_DIRS})
link_libraries(${TBB_LIBRARIES})

find_package(AIO REQUIRED)
include_directories(${AIO_INCLUDE_DIRS})
link_directories(${AIO_LIBRARY_DIRS})
link_libraries(${AIO_LIBRARIES})

set(Boost_USE_STATIC_LIBS        ON)  # only find static libs
set(Boost_USE_DEBUG_LIBS         OFF) # ignore debug libs and
set(Boost_USE_RELEASE_LIBS       ON)  # only find release libs
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME     OFF)
find_package(Boost REQUIRED COMPONENTS fiber)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
link_libraries(${Boost_LIBRARIES})

find_package(PkgConfig)
# pkg_check_modules(UCX ucx>=1.8)
if(UCX_FOUND)
    include_directories(${UCX_INCLUDE_DIRS})
    link_directories(${UCX_LIBRARY_DIRS})
    link_libraries(${UCX_LIBRARIES})
endif()

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no march native support.")
endif()

try_compile(ARCH_HAS_RTM "${CMAKE_BINARY_DIR}/temp" "${CMAKE_SOURCE_DIR}/cmake/rtm_test.cpp")
if(ARCH_HAS_RTM)
    message(STATUS "Your CPU supports rtm.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DARCH_HAS_RTM")
else()
    message(STATUS "Your CPU doesn't support rtm, fallback to locks.")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-Ofast -g -DNDEBUG")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set(PROJECT_DEPS_DIR ${PROJECT_SOURCE_DIR}/deps)
include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_DEPS_DIR})

add_subdirectory(${PROJECT_DEPS_DIR}/abseil-cpp)
link_libraries(absl::flat_hash_map absl::node_hash_map)

option(ENABLE_MMAP "Enable default mmap pool" OFF)
if (ENABLE_MMAP)
    add_definitions("-DENABLE_MMAP")
endif()
add_library(mmapalloc STATIC core/mmapalloc.cpp)
link_libraries(mmapalloc)

option(ENABLE_KVS "Enable Key Value Store" OFF)
if (ENABLE_KVS)
    add_definitions("-DENABLE_KVS")
endif()

option(ENABLE_WAL "Enable Write-Ahead Log" OFF)
if (ENABLE_WAL)
    add_definitions("-DENABLE_WAL")
endif()

# add_executable(convert_to_binary utils/convert_to_binary.cpp)
# add_executable(convert_to_binary_random utils/convert_to_binary_random.cpp)
# add_executable(convert_to_binary_timestamp utils/convert_to_binary_timestamp.cpp)

# add_executable(wcc src/wcc.cpp)
# add_executable(wcc_inc src/wcc_inc.cpp)
# add_executable(wcc_inc_batch src/wcc_inc_batch.cpp)
# add_executable(wcc_inc_clf src/wcc_inc_clf.cpp)
# add_executable(wcc_inc_rt src/wcc_inc_rt.cpp)

# add_executable(bfs src/bfs.cpp)
# add_executable(bfs_inc src/bfs_inc.cpp)
# add_executable(bfs_inc_batch src/bfs_inc_batch.cpp)
# add_executable(bfs_inc_clf src/bfs_inc_clf.cpp)
# add_executable(bfs_inc_rt src/bfs_inc_rt.cpp)

# add_executable(sssp src/sssp.cpp)
# add_executable(sssp_inc src/sssp_inc.cpp)
# add_executable(sssp_risgraph_baseline src/sssp_risgraph_baseline.cpp)
# add_executable(sssp_inc_batch src/sssp_inc_batch.cpp)
# add_executable(sssp_inc_clf src/sssp_inc_clf.cpp)
# add_executable(sssp_inc_rt src/sssp_inc_rt.cpp)
# add_executable(Risgraph_baseline_SSSP src/Risgraph_baseline_SSSP.cpp)
add_executable(CommonGraph_baseline_direct_SSSP src/CommonGraph_baseline_direct_SSSP.cpp)
add_executable(CommonGraph_baseline_WH src/CommonGraph_baseline_WH.cpp)

add_executable(sswp src/sswp.cpp)
add_executable(sswp_inc src/sswp_inc.cpp)
add_executable(sswp_inc_batch src/sswp_inc_batch.cpp)
add_executable(sswp_inc_clf src/sswp_inc_clf.cpp)
add_executable(sswp_inc_rt src/sswp_inc_rt.cpp)


add_executable(bfs-core src/bfs-core.cpp)
add_executable(sssp-core src/core-graph-sssp.cpp)
add_executable(sswp-core src/sswp-core.cpp)
add_executable(ssnp-core src/ssnp-core.cpp)
add_executable(wcc-core src/wcc-core.cpp)
add_executable(viterbi-core src/viterbi-core.cpp)



add_executable(index-wcc src/index-wcc.cpp)
add_executable(index-sssp src/index-graph-sssp.cpp)
add_executable(index-sswp src/index-sswp.cpp)
add_executable(index-viterbi src/index-viterbi.cpp)
add_executable(index-bfs src/index-bfs.cpp)
add_executable(index-ssnp src/index-ssnp.cpp)
add_executable(correctness-sssp src/sssp_correctness.cpp)
add_executable(correctness-sswp src/sswp_correctness.cpp)
add_executable(correctness-ssnp src/ssnp_correctness.cpp)
add_executable(correctness-bfs src/bfs_correctness.cpp)
add_executable(correctness-viterbi src/viterbi_correctness.cpp)
add_executable(correctness-wcc src/wcc_correctness.cpp)
add_executable(test_sssp src/test_sssp.cpp)
add_executable(Indexing-baseline-sssp src/Indexing-Baseline-SSSP.cpp)

add_executable(fusion_sssp src/fusion_sssp_test.cpp)

