cmake_minimum_required(VERSION 3.0.2)
project(visual_occupancy_map)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  nav_msgs
  roscpp
  sensor_msgs
  message_filters
)

## System dependencies are found with CMake's conventions
find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

find_package(Eigen3 3.1.0 REQUIRED)
find_package(PCL REQUIRED)
find_package(Boost COMPONENTS system)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(LIBS 
${OpenCV_LIBS} 
${EIGEN3_LIBS}
${Boost_LIBRARIES}
)

catkin_package(
 CATKIN_DEPENDS cv_bridge geometry_msgs nav_msgs roscpp sensor_msgs message_filters
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide

add_executable(occupancy_map_node src/occupancy_grid_map.cpp)

add_executable(semantic_occupancy_map_node src/semantic_occupancy_grid_map.cpp)

##add_executable(dense_occupancy_map_node src/dense_occupancy_grid_map.cpp)


## Specify libraries to link a library or executable target against
target_link_libraries(occupancy_map_node
   ${catkin_LIBRARIES}
${LIBS}
 )

 target_link_libraries(semantic_occupancy_map_node
   ${catkin_LIBRARIES}
${LIBS}
 )

## target_link_libraries(dense_occupancy_map_node
##   ${catkin_LIBRARIES}
## ${LIBS}
## )

