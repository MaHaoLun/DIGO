cmake_minimum_required(VERSION 3.16)
project(sdc_daily_snapshot VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SDC_SOURCES
    src/snapshot_main.cpp
    src/graph/graph_inputs.cpp
    src/graph/trajectory_reader.cpp
    src/graph/aggregation.cpp
    src/graph/snapshot_builder.cpp
    src/io/json.cpp
    src/io/paths.cpp
)
add_executable(sdc_snapshot ${SDC_SOURCES})
target_include_directories(sdc_snapshot PRIVATE include)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(sdc_snapshot PRIVATE -Wall -Wextra -Wpedantic)
endif()

set(CORPUS_SOURCES
    src/graph/graph_context.cpp
    src/graph/lpf.cpp
    src/routing/td_dijkstra.cpp
    src/sampling/hierarchical_grid_sampler.cpp
    src/sampling/uniform_grid_sampler.cpp
    src/sampling/cluster_sampler.cpp
    src/sampling/query_plan.cpp
    src/sampling/online_query_plan.cpp
    src/encoding/image_encoder.cpp
    src/io/corpus_writer.cpp
    src/io/npz.cpp
)
add_library(corpus_core STATIC ${CORPUS_SOURCES})
target_include_directories(corpus_core PUBLIC include)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(corpus_core PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_executable(sdc_offline src/offline_main.cpp src/io/json.cpp src/io/paths.cpp)
target_link_libraries(sdc_offline PRIVATE corpus_core)
add_executable(sdc_online src/online_main.cpp src/io/json.cpp src/io/paths.cpp)
target_link_libraries(sdc_online PRIVATE corpus_core)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(sdc_offline PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(sdc_online PRIVATE -Wall -Wextra -Wpedantic)
endif()

