# Visual_occupancy_map

A ROS node for generating occupancy grid map and semantic occupancy grid map based on visual SLAM.

## Dependencies
Tested on Ubuntu 18.04, ROS Melodic, OpenCV 3.2.0, Eigen 3.2.10.

- OpenCV;
- Eigen3.

## Build
You can use provided Dockerfile or build using catkin build.

## Subscribed topics

- ` /KF_pose` - **nav_msgs::Odometry** - pose of camera in a keyframe;
- ` /KF_map_points` - **sensor_msgs::PointCloud** - map points that are visible on a keyframe;
- ` /semantic_image` - **sensor_msgs::Image** - image with result of semantic segmentation **(only for semantic_occupancy_map_node)**;

## Published topics

- ` /visual_occupancy_node/grid_map` - **nav_msgs::OccupancyGrid** - occupancy grid map;
- ` /visual_occupancy_node/trajectory_grid_map` - **sensor_msgs::Image** - image with occupancy grid map and trajectory of robot;
- ` /visual_occupancy_node/semantic_occupancy_map` - **sensor_msgs::Image** - semantic occupancy map **(only for semantic_occupancy_map_node)**;

## Parameters
You can change next parameters in .yaml file:
- `Scale_factor` - 1 / resolution of map - default: 20;
- `Tcw_form` - form of KF_pose, 0 equals Twc form - default: 0;
- `Cloud_max_x` - max value of map's horizontal in meters - default: 10;
- `Cloud_min_x` - min value of map's horizontal in meters - default: -10;
- `Cloud_max_z` - max value of map's vertical in meters - default: 16;
- `Cloud_min_z` - min value of map's vertical in meters - default: -5;
- `Free_thresh` - value above which cell determines as free area - default: 0.55;
- `Occupied_thresh` - value below which cell determines as occupied area - default: 0.5;
- `Use_local_counters` - using local counters in keyframes for reduce fake free cells - default: 0;
- `Use_gaussian_counters` - using gaussian smoothing for reduce discretization - default: 0;
- `Use_boundary_detection` - using Canny algorithm for detect boundary between free and unknown area and set it as occupancy area - default: 0;
- `Publish_trajectory` - publish image in '/visual_occupancy_node/trajectory_grid_map' topic - default: 0;
- `Visit_thresh` - visit counter value above which map points are provided for update occupancy map - default: 0;
- `Thresh_extension` - when difference between KF_pose and bound of map in meters below this value, extension of map is starting - default: 5;
- `Step_extension` - value which added to bound of map when extension is running - default: 10;
- `Gaussian_kernel_size` - kernel size for gaussian smoothing - default: 3;
- `Canny_thresh` - value of lower thresh in Canny detector, upper thresh equals 2 * lower thresh - default: 350;
- `Upper_height` - map points which have height value above this thresh not used - default: 1.0;
- `Lower_height` - map points which have height value below this thresh not used - default: 0.0;
- `Make_submaps` - make submaps whith colours in submaps.txt **(only for semantic_occupancy_map_node)** - default: 0;
- `Use_semantic_filter` - using only map points whith semantic labels in submaps.txt for update counters **(only for semantic_occupancy_map_node)** - default: 0.

## Examples
For running occupancy_map_node with provided config file sber_mseg.yaml use:
```
roslaunch visual_occupancy_map sber_mseg.launch
```
For running semantic_occupancy_map_node with provided config file sber_mseg.yaml use:
```
roslaunch visual_occupancy_map semantic_sber_mseg.launch
```
If you want to build additional submaps, setup submaps.txt and use:
```
roslaunch visual_occupancy_map submaps.launch
```
Also you can use configs occupancy_map_node and semantic_occupancy_map_node for rviz in folder 'rviz'.