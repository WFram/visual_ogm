#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/PoseArray.h"
#include "nav_msgs/OccupancyGrid.h"
#include "nav_msgs/Odometry.h"
#include "sensor_msgs/PointCloud.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/Image.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Eigen>
#include <opencv2/core/core.hpp>

#include <opencv2/highgui/highgui_c.h>
#include <opencv2/highgui/highgui.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

// parameters
float scale_factor = 20;
float cloud_max_x = 10.0;
float cloud_min_x = -10.0;
float cloud_max_z = 10.0;
float cloud_min_z = -10.0;
float free_thresh = 0.55;
float occupied_thresh = 0.50;
int visit_thresh = 0;
float step_extension = 10;
float thresh_extension = 5;
bool use_local_counters = false;
bool use_gaussian_counters = false;
bool use_boundary_detection = false;
bool publish_trajectory = false;
int gaussian_kernel_size = 3;
int canny_thresh = 350;
float upper_height = 1.0;
float lower_height = 0.0;
float height_cam = 0.5;
bool Tcw_form = false;

float grid_max_x, grid_min_x, grid_max_z, grid_min_z;
cv::Mat global_occupied_counter, global_visit_counter;
cv::Mat local_occupied_counter, local_visit_counter;
cv::Mat local_map_pt_mask;
cv::Mat grid_map, grid_map_int, grid_map_thresh;
cv::Mat gauss_kernel;
cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
float norm_factor_x, norm_factor_z;
int h, w;
unsigned int n_kf_received;
bool loop_closure_being_processed = false;
ros::Publisher pub_grid_map;
ros::Publisher pub_traj_map;
nav_msgs::OccupancyGrid grid_map_msg;
bool first_msg = true;
float kf_pos_x, kf_pos_z;
int kf_pos_grid_x, kf_pos_grid_z;
std::vector <cv::Point> trj;
std::vector <cv::Point> kf_poses;

using namespace std;

void ptsKFCallback(const sensor_msgs::PointCloud2::ConstPtr &MapPoints, const nav_msgs::Odometry::ConstPtr &Kf_pose);

void updateGridMap(const geometry_msgs::PoseArray::ConstPtr &pts_and_pose);

void resetGridMap(const geometry_msgs::PoseArray::ConstPtr &pts_and_pose);

void saveMap(unsigned int id = 0);

void ptCallback(const geometry_msgs::PoseArray::ConstPtr &pts_and_pose);

void loopClosingCallback(const geometry_msgs::PoseArray::ConstPtr &all_kf_and_pts);

void parseParams(int argc, char **argv);

void printParams();

void extensionMap(const bool &width, const bool &positive);

void processMapPt(const geometry_msgs::Point &curr_pt, cv::Mat &occupied,
                  cv::Mat &visited, cv::Mat &pt_mask, int kf_pos_grid_x, int kf_pos_grid_z);

void processMapPts(const std::vector <geometry_msgs::Pose> &pts, unsigned int n_pts,
                   unsigned int start_id, int kf_pos_grid_x, int kf_pos_grid_z);

void getGridMap();

int main(int argc, char **argv) {
    ros::init(argc, argv, "Vision occupancy map");
    ros::start();

    parseParams(argc, argv);
    printParams();
    grid_max_x = cloud_max_x * scale_factor;
    grid_min_x = cloud_min_x * scale_factor;
    grid_max_z = cloud_max_z * scale_factor;
    grid_min_z = cloud_min_z * scale_factor;
    printf("grid_max: %f, %f\t grid_min: %f, %f\n", grid_max_x, grid_max_z, grid_min_x, grid_min_z);

    double grid_res_x = grid_max_x - grid_min_x, grid_res_z = grid_max_z - grid_min_z;

    h = grid_res_z;
    w = grid_res_x;
    printf("grid_size: (%d, %d)\n", h, w);
    n_kf_received = 0;

    global_occupied_counter.create(h, w, CV_32FC1);
    global_visit_counter.create(h, w, CV_32FC1);
    global_occupied_counter.setTo(cv::Scalar(0));
    global_visit_counter.setTo(cv::Scalar(0));

    grid_map_msg.data.resize(h * w);
    grid_map_msg.info.width = w;
    grid_map_msg.info.height = h;
    grid_map_msg.info.resolution = 1.0 / scale_factor;

    grid_map_int = cv::Mat(h, w, CV_8SC1, (char *) (grid_map_msg.data.data()));

    grid_map.create(h, w, CV_32FC1);
    grid_map_thresh.create(h, w, CV_8UC1);

    local_occupied_counter.create(h, w, CV_32FC1);
    local_visit_counter.create(h, w, CV_32FC1);
    local_map_pt_mask.create(h, w, CV_8UC1);

    gauss_kernel = cv::getGaussianKernel(gaussian_kernel_size, -1);

    norm_factor_x = float(grid_res_x - 1) / float(grid_max_x - grid_min_x);
    norm_factor_z = float(grid_res_z - 1) / float(grid_max_z - grid_min_z);
    printf("norm_factor_x: %f\n", norm_factor_x);
    printf("norm_factor_z: %f\n", norm_factor_z);

    ros::NodeHandle nodeHandler;


    ros::Subscriber sub_all_kf_and_pts = nodeHandler.subscribe("/all_KF_and_points", 1000, loopClosingCallback);
    message_filters::Subscriber <sensor_msgs::PointCloud2> MapPoints_sub(nodeHandler, "/KF_map_points", 1000);
    message_filters::Subscriber <nav_msgs::Odometry> KF_pose_sub(nodeHandler, "/KF_pose", 1000);

    typedef message_filters::sync_policies::ApproximateTime <sensor_msgs::PointCloud2, nav_msgs::Odometry> sync_pol;
    message_filters::Synchronizer <sync_pol> sync(sync_pol(1000), MapPoints_sub, KF_pose_sub);
    sync.registerCallback(boost::bind(&ptsKFCallback, _1, _2));

    pub_grid_map = nodeHandler.advertise<nav_msgs::OccupancyGrid>("/visual_occupancy_node/grid_map", 1000);
    if (publish_trajectory) {
        pub_traj_map = nodeHandler.advertise<sensor_msgs::Image>("/visual_occupancy_node/trajectory_grid_map", 1000);
    }
    ros::spin();
    ros::shutdown();
    cv::destroyAllWindows();
    saveMap();

    return 0;
}

void ptsKFCallback(const sensor_msgs::PointCloud2::ConstPtr &MapPoints, const nav_msgs::Odometry::ConstPtr &Kf_pose) {
    if (loop_closure_being_processed) { return; }

    std_msgs::Header h1 = MapPoints->header;
    std_msgs::Header h2 = Kf_pose->header;

    geometry_msgs::PoseArray pts_and_pose;
    pts_and_pose.header.seq = h2.seq;
    geometry_msgs::Pose temp;

    if (Tcw_form) {
        temp.position = Kf_pose->pose.pose.position;
        temp.orientation = Kf_pose->pose.pose.orientation;

        Eigen::Quaterniond Q(temp.orientation.w,
                             temp.orientation.x,
                             temp.orientation.y,
                             temp.orientation.z);


        Eigen::Matrix3d Rcw = Q.toRotationMatrix();
        Eigen::Vector3d Pcw(temp.position.x,
                            temp.position.y,
                            temp.position.z);

        Eigen::Vector3d Pwc = -Rcw.transpose() * Pcw;

        temp.position.x = Pwc.x();
        temp.position.z = Pwc.y();
        temp.position.y = Pwc.z();

        Eigen::Matrix3d Rwc = Rcw.transpose();

        Eigen::Quaterniond Qwc(Rwc);

        temp.orientation.w = Qwc.w();
        temp.orientation.x = Qwc.x();
        temp.orientation.y = Qwc.y();
        temp.orientation.z = Qwc.z();
    } else {
        temp.position.x = Kf_pose->pose.pose.position.x;
        temp.position.z = Kf_pose->pose.pose.position.y;
        temp.position.y = Kf_pose->pose.pose.position.z;

        temp.orientation.w = Kf_pose->pose.pose.orientation.w;
        temp.orientation.x = Kf_pose->pose.pose.orientation.x;
        temp.orientation.y = Kf_pose->pose.pose.orientation.y;
        temp.orientation.z = Kf_pose->pose.pose.orientation.z;
    }

    if (first_msg) {
        // Set coordinate system origin
        // grid_map_msg.info.origin.position.x = temp.position.x + cloud_min_x;
        // grid_map_msg.info.origin.position.y = temp.position.z + cloud_min_z;
        grid_map_msg.info.origin.position.x = cloud_min_x;
        grid_map_msg.info.origin.position.y = cloud_min_z;
        grid_map_msg.info.origin.position.z = temp.position.y - height_cam;

        grid_map_msg.info.origin.orientation.w = 1;
        grid_map_msg.info.origin.orientation.x = 0;
        grid_map_msg.info.origin.orientation.y = 0;
        grid_map_msg.info.origin.orientation.z = 0;
        first_msg = false;

    }

    pts_and_pose.poses.push_back(temp);

    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(*MapPoints, cloud);

    for (auto mp: cloud.points) {
        Eigen::Vector3d point(mp.x, mp.y, mp.z);
        if (point.norm() == 0)
            continue;
        if (mp.z < lower_height || mp.z > upper_height)
            continue;
        geometry_msgs::Pose tempPt;
        tempPt.position.x = mp.x;
        tempPt.position.z = mp.y;
        tempPt.position.y = mp.z;
        pts_and_pose.poses.push_back(tempPt);
    }
    if (temp.position.x > (cloud_max_x - thresh_extension)) {
        cloud_max_x = step_extension + cloud_max_x;
        extensionMap(true, true);
    }
    if (temp.position.x < (cloud_min_x + thresh_extension)) {
        cloud_min_x = -step_extension + cloud_min_x;
        extensionMap(true, false);
    }
    if (temp.position.z > (cloud_max_z - thresh_extension)) {
        cloud_max_z = step_extension + cloud_max_z;
        extensionMap(false, true);
    }
    if (temp.position.z < (cloud_min_z + thresh_extension)) {
        cloud_min_z = -step_extension + cloud_min_z;
        extensionMap(false, false);
    }
    boost::shared_ptr <geometry_msgs::PoseArray> pts_and_pose_temp = boost::make_shared<geometry_msgs::PoseArray>(
            pts_and_pose);
    updateGridMap(pts_and_pose_temp);

    grid_map_msg.info.map_load_time = ros::Time::now();
    pub_grid_map.publish(grid_map_msg);

}

void saveMap(unsigned int id) {
    printf("saving maps with id: %u\n", id);
    mkdir("results", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (id > 0) {
        cv::imwrite("results//grid_map_" + to_string(id) + ".jpg", grid_map);
        cv::imwrite("results//grid_map_thresh_" + to_string(id) + ".jpg", grid_map_thresh);

        std::vector<int> compression_params;
        compression_params.push_back(cv::IMWRITE_PXM_BINARY);
        compression_params.push_back(0);
        const std::string imageFilename = "results//grid_map_navigation" + to_string(id) + ".pgm";

        cv::imwrite(imageFilename, grid_map_thresh, compression_params);

    } else {
        cv::imwrite("results//grid_map.jpg", grid_map);
        cv::imwrite("results//grid_map_thresh.jpg", grid_map_thresh);

        std::vector<int> compression_params;
        compression_params.push_back(cv::IMWRITE_PXM_BINARY);
        compression_params.push_back(0);
        const std::string imageFilename = "results//grid_map_navigation.pgm";

        cv::imwrite(imageFilename, grid_map_thresh, compression_params);
    }

}

void ptCallback(const geometry_msgs::PoseArray::ConstPtr &pts_and_pose) {
    if (loop_closure_being_processed) { return; }

    updateGridMap(pts_and_pose);

    grid_map_msg.info.map_load_time = ros::Time::now();

    pub_grid_map.publish(grid_map_msg);
}

void loopClosingCallback(const geometry_msgs::PoseArray::ConstPtr &all_kf_and_pts) {

    loop_closure_being_processed = true;
    resetGridMap(all_kf_and_pts);
    loop_closure_being_processed = false;
}

void processMapPt(const geometry_msgs::Point &curr_pt, cv::Mat &occupied,
                  cv::Mat &visited, cv::Mat &pt_mask, int kf_pos_grid_x, int kf_pos_grid_z) {
    float pt_pos_x = curr_pt.x * scale_factor;
    float pt_pos_z = curr_pt.z * scale_factor;

    int pt_pos_grid_x = int(floor((pt_pos_x - grid_min_x) * norm_factor_x));
    int pt_pos_grid_z = int(floor((pt_pos_z - grid_min_z) * norm_factor_z));

    if (pt_pos_grid_x < 0 || pt_pos_grid_x >= w)
        return;

    if (pt_pos_grid_z < 0 || pt_pos_grid_z >= h)
        return;

    // Increment the occupancy account of the grid cell where map point is located
    ++occupied.at<int>(pt_pos_grid_z, pt_pos_grid_x);
    pt_mask.at<uchar>(pt_pos_grid_z, pt_pos_grid_x) = 255;

    // Get all grid cell that the line between keyframe and map point pass through
    int x0 = kf_pos_grid_x;
    int y0 = kf_pos_grid_z;
    int x1 = pt_pos_grid_x;
    int y1 = pt_pos_grid_z;
    bool steep = (abs(y1 - y0) > abs(x1 - x0));
    if (steep) {
        swap(x0, y0);
        swap(x1, y1);
    }
    if (x0 > x1) {
        swap(x0, x1);
        swap(y0, y1);
    }
    int dx = x1 - x0;
    int dy = abs(y1 - y0);
    double error = 0;
    double deltaerr = ((double) dy) / ((double) dx);
    int y = y0;
    int ystep = (y0 < y1) ? 1 : -1;
    for (int x = x0; x <= x1; ++x) {
        if (steep) {
            ++visited.at<int>(x, y);
        } else {
            ++visited.at<int>(y, x);
        }
        error = error + deltaerr;
        if (error >= 1) {
            y = y + ystep;
            error = error - 1.0;
        }
    }
}

void processMapPts(const std::vector <geometry_msgs::Pose> &pts, unsigned int n_pts,
                   unsigned int start_id, int kf_pos_grid_x, int kf_pos_grid_z) {
    unsigned int end_id = start_id + n_pts;
    if (use_local_counters) {
        local_map_pt_mask.setTo(0);
        local_occupied_counter.setTo(0);
        local_visit_counter.setTo(0);
        for (unsigned int pt_id = start_id; pt_id < end_id; ++pt_id) {
            processMapPt(pts[pt_id].position, local_occupied_counter, local_visit_counter,
                         local_map_pt_mask, kf_pos_grid_x, kf_pos_grid_z);
        }
        for (int row = 0; row < h; ++row) {
            for (int col = 0; col < w; ++col) {
                if (local_map_pt_mask.at<uchar>(row, col) == 0) {
                    local_occupied_counter.at<int>(row, col) = 0;
                } else {
                    local_occupied_counter.at<int>(row, col) = local_visit_counter.at<int>(row, col);
                }
            }
        }
        if (use_gaussian_counters) {
            cv::filter2D(local_occupied_counter, local_occupied_counter, CV_32F, gauss_kernel);
            cv::filter2D(local_visit_counter, local_visit_counter, CV_32F, gauss_kernel);
        }
        global_occupied_counter += local_occupied_counter;
        global_visit_counter += local_visit_counter;
    } else {
        for (unsigned int pt_id = start_id; pt_id < end_id; ++pt_id) {
            processMapPt(pts[pt_id].position, global_occupied_counter, global_visit_counter,
                         local_map_pt_mask, kf_pos_grid_x, kf_pos_grid_z);
        }
    }
}

void updateGridMap(const geometry_msgs::PoseArray::ConstPtr &pts_and_pose) {

    //printf("Received frame %u \n", pts_and_pose->header.seq);
    const geometry_msgs::Point &kf_location = pts_and_pose->poses[0].position;
    kf_pos_x = kf_location.x * scale_factor;
    kf_pos_z = kf_location.z * scale_factor;
    if (publish_trajectory) {
        kf_poses.push_back(cv::Point(kf_pos_x, kf_pos_z));
    }

    kf_pos_grid_x = int(floor((kf_pos_x - grid_min_x) * norm_factor_x));
    kf_pos_grid_z = int(floor((kf_pos_z - grid_min_z) * norm_factor_z));

    if (kf_pos_grid_x < 0 || kf_pos_grid_x >= w)
        return;

    if (kf_pos_grid_z < 0 || kf_pos_grid_z >= h)
        return;
    ++n_kf_received;
    unsigned int n_pts = pts_and_pose->poses.size() - 1;

    processMapPts(pts_and_pose->poses, n_pts, 1, kf_pos_grid_x, kf_pos_grid_z);
    getGridMap();

    // Publish trajectory image
    if (publish_trajectory) {
        cv::Mat traj = grid_map_thresh.clone();
        trj.push_back(cv::Point(kf_pos_grid_x, kf_pos_grid_z));
        for (size_t i = 0; i < trj.size(); i++) {
            cv::circle(traj, trj[i], 2, cv::Scalar(0), -1);
            if (i != 0)
                cv::line(traj, trj[i - 1], trj[i], cv::Scalar(0), 2);
        }
        ros::Time time = ros::Time::now();
        cv_ptr->encoding = "mono8";
        cv_ptr->header.stamp = time;
        cv_ptr->header.frame_id = "trajectory_grid_map";
        cv_ptr->image = traj;
        pub_traj_map.publish(cv_ptr->toImageMsg());
    }
}

void resetGridMap(const geometry_msgs::PoseArray::ConstPtr &all_kf_and_pts) {
    global_visit_counter.setTo(0);
    global_occupied_counter.setTo(0);
    // first_msg = true;
    trj.clear();

    unsigned int n_kf = all_kf_and_pts->poses[0].position.x;
    if ((unsigned int) (all_kf_and_pts->poses[0].position.y) != n_kf ||
        (unsigned int) (all_kf_and_pts->poses[0].position.z) != n_kf) {
        printf("resetGridMap :: Unexpected formatting in the keyframe count element\n");
        return;
    }
    printf("Resetting grid map with %d key frames\n", n_kf);
#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif
    unsigned int id = 0;
    for (unsigned int kf_id = 0; kf_id < n_kf; ++kf_id) {
        const geometry_msgs::Point &kf_location = all_kf_and_pts->poses[++id].position;
        //const geometry_msgs::Quaternion &kf_orientation = pts_and_pose->poses[0].orientation;
        unsigned int n_pts = all_kf_and_pts->poses[++id].position.x;
        if ((unsigned int) (all_kf_and_pts->poses[id].position.y) != n_pts ||
            (unsigned int) (all_kf_and_pts->poses[id].position.z) != n_pts) {
            printf("resetGridMap :: Unexpected formatting in the point count element for keyframe %d\n", kf_id);
            return;
        }
        // if (first_msg){
        // 	grid_map_msg.info.origin.position.x = cloud_min_x;
        // 	grid_map_msg.info.origin.position.y = cloud_min_z;
        // 	grid_map_msg.info.origin.position.z = kf_location.y - height_cam;
        // 	first_msg = false;
        // }
        float kf_pos_x = kf_location.x * scale_factor;
        float kf_pos_z = kf_location.y * scale_factor;

        int kf_pos_grid_x = int(floor((kf_pos_x - grid_min_x) * norm_factor_x));
        int kf_pos_grid_z = int(floor((kf_pos_z - grid_min_z) * norm_factor_z));

        trj.push_back(cv::Point(kf_pos_grid_x, kf_pos_grid_z));

        if (kf_pos_grid_x < 0 || kf_pos_grid_x >= w)
            continue;

        if (kf_pos_grid_z < 0 || kf_pos_grid_z >= h)
            continue;

        if (id + n_pts >= all_kf_and_pts->poses.size()) {
            printf("resetGridMap :: Unexpected end of the input array while processing keyframe %u with %u points\n",
                   kf_id, n_pts);
            return;
        }
        processMapPts(all_kf_and_pts->poses, n_pts, id + 1, kf_pos_grid_x, kf_pos_grid_z);
        id += n_pts;
    }
    getGridMap();
#ifdef COMPILEDWITHC11
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
    std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif
    double ttrack = std::chrono::duration_cast < std::chrono::duration < double > > (t2 - t1).count();
    printf("Done. Time taken: %f secs\n", ttrack);
    pub_grid_map.publish(grid_map_msg);
}

void getGridMap() {
    double unknown_region_ratio = (free_thresh + occupied_thresh) / 2.0;
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            int visits = global_visit_counter.at<int>(row, col);
            int occupieds = global_occupied_counter.at<int>(row, col);

            if (visits <= visit_thresh) {
                grid_map.at<float>(row, col) = unknown_region_ratio;
            } else {
                grid_map.at<float>(row, col) = 1.0 - float(occupieds / visits);
            }
            if (grid_map.at<float>(row, col) >= free_thresh) {
                grid_map_thresh.at<uchar>(row, col) = 255;
                grid_map_int.at<char>(row, col) = (1 - grid_map.at<float>(row, col)) * 100;
            } else if (grid_map.at<float>(row, col) < free_thresh && grid_map.at<float>(row, col) >= occupied_thresh) {
                grid_map_thresh.at<uchar>(row, col) = 128;
                grid_map_int.at<char>(row, col) = -1;
            } else {
                grid_map_thresh.at<uchar>(row, col) = 0;
                grid_map_int.at<char>(row, col) = (1 - grid_map.at<float>(row, col)) * 100;
            }

        }
    }
    if (use_boundary_detection) {
        cv::Mat canny_output;
        std::vector <std::vector<cv::Point>> contours;
        std::vector <cv::Vec4i> hierarchy;
        cv::Canny(grid_map_thresh, canny_output, canny_thresh, canny_thresh * 2, 3);
        for (int row = 0; row < h; ++row) {
            for (int col = 0; col < w; ++col) {
                if (canny_output.at<uchar>(row, col) > 0) {
                    grid_map_thresh.at<uchar>(row, col) = 0;
                    grid_map_int.at<char>(row, col) = 100;
                }
            }
        }
        cv::findContours(canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, cv::Point(0, 0));
        drawContours(grid_map_thresh, contours, -1, CV_RGB(0, 0, 0), 1, CV_AA);
    }
}

void parseParams(int argc, char **argv) {
    cv::FileStorage fs(argv[1], cv::FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "failed to open " << argv[1] << endl;
        exit(-1);
    }
    fs["Scale_factor"] >> scale_factor;
    fs["Tcw_form"] >> Tcw_form;
    fs["Cloud_max_x"] >> cloud_max_x;
    fs["Cloud_min_x"] >> cloud_min_x;
    fs["Cloud_max_z"] >> cloud_max_z;
    fs["Cloud_min_z"] >> cloud_min_z;
    fs["Free_thresh"] >> free_thresh;
    fs["Occupied_thresh"] >> occupied_thresh;
    fs["Use_local_counters"] >> use_local_counters;
    fs["Use_gaussian_counters"] >> use_gaussian_counters;
    fs["Publish_trajectory"] >> publish_trajectory;
    fs["Visit_thresh"] >> visit_thresh;
    fs["Step_extension"] >> step_extension;
    h;
    fs["Thresh_extension"] >> thresh_extension;
    fs["Gaussian_kernel_size"] >> gaussian_kernel_size;
    fs["Use_boundary_detection"] >> use_boundary_detection;
    fs["Canny_thresh"] >> canny_thresh;
    fs["Upper_height"] >> upper_height;
    fs["Lower_height"] >> lower_height;
}

void printParams() {
    printf("Using params:\n");
    printf("scale_factor: %f\n", scale_factor);
    printf("cloud_max: %f, %f\t cloud_min: %f, %f\n", cloud_max_x, cloud_max_z, cloud_min_x, cloud_min_z);
    printf("free_thresh: %f\n", free_thresh);
    printf("occupied_thresh: %f\n", occupied_thresh);
    printf("use_local_counters: %d\n", use_local_counters);
    printf("use_gaussian_counters: %d\n", use_gaussian_counters);
    printf("visit_thresh: %d\n", visit_thresh);
    printf("step of extension: %f\n", step_extension);
    printf("thresh of extension: %f\n", thresh_extension);
    printf("publish_trajectory: %d\n", publish_trajectory);
}


void extensionMap(const bool &width, const bool &positive) {
    printf("New height and width set \n");
    grid_max_x = cloud_max_x * scale_factor;
    grid_min_x = cloud_min_x * scale_factor;
    grid_max_z = cloud_max_z * scale_factor;
    grid_min_z = cloud_min_z * scale_factor;

    printf("grid_max: %f, %f\t grid_min: %f, %f\n", grid_max_x, grid_max_z, grid_min_x, grid_min_z);

    double grid_res_x = grid_max_x - grid_min_x, grid_res_z = grid_max_z - grid_min_z;

    h = grid_res_z;
    w = grid_res_x;
    printf("grid_size: (%d, %d)\n", h, w);

    norm_factor_x = float(grid_res_x - 1) / float(grid_max_x - grid_min_x);
    norm_factor_z = float(grid_res_z - 1) / float(grid_max_z - grid_min_z);

    cv::Mat new_global_occupied_counter(h, w, CV_32FC1);
    cv::Mat new_global_visit_counter(h, w, CV_32FC1);
    new_global_occupied_counter.setTo(cv::Scalar(0));
    new_global_visit_counter.setTo(cv::Scalar(0));
    // Positive Axe X and Positive Axe Z
    if (positive) {
        if (width) {
            cout << "Positive Axe X extension" << endl;
        } else {
            cout << "Positive Axe Z extension" << endl;
        }
        global_occupied_counter.copyTo(new_global_occupied_counter(
                cv::Rect(0, 0, global_occupied_counter.cols, global_occupied_counter.rows)));
        global_visit_counter.copyTo(
                new_global_visit_counter(cv::Rect(0, 0, global_visit_counter.cols, global_visit_counter.rows)));

    }
    // Negative Axe X
    if (width && !positive) {
        cout << "Negative Axe X extension" << endl;
        global_occupied_counter.copyTo(new_global_occupied_counter(
                cv::Rect(step_extension * scale_factor, 0, global_occupied_counter.cols,
                         global_occupied_counter.rows)));
        global_visit_counter.copyTo(new_global_visit_counter(
                cv::Rect(step_extension * scale_factor, 0, global_visit_counter.cols, global_visit_counter.rows)));

        grid_map_msg.info.origin.position.x -= step_extension;
        if (publish_trajectory) {
            trj.clear();
            for (cv::Point &pos: kf_poses) {
                trj.push_back(cv::Point(int(floor((pos.x - grid_min_x) * norm_factor_x)),
                                        int(floor((pos.y - grid_min_z) * norm_factor_z))));
            }
        }
    }
    // Negative Axe Z
    if (!width && !positive) {
        cout << "Negative Axe Z extension" << endl;
        global_occupied_counter.copyTo(new_global_occupied_counter(
                cv::Rect(0, step_extension * scale_factor, global_occupied_counter.cols,
                         global_occupied_counter.rows)));
        global_visit_counter.copyTo(new_global_visit_counter(
                cv::Rect(0, step_extension * scale_factor, global_visit_counter.cols, global_visit_counter.rows)));

        grid_map_msg.info.origin.position.y -= step_extension;
        if (publish_trajectory) {
            trj.clear();
            for (cv::Point &pos: kf_poses) {
                trj.push_back(cv::Point(int(floor((pos.x - grid_min_x) * norm_factor_x)),
                                        int(floor((pos.y - grid_min_z) * norm_factor_z))));
            }
        }
    }
    global_occupied_counter = new_global_occupied_counter;
    global_visit_counter = new_global_visit_counter;

    grid_map_msg.data.resize(h * w);
    grid_map_msg.info.width = w;
    grid_map_msg.info.height = h;
    grid_map_int = cv::Mat(h, w, CV_8SC1, (char *) (grid_map_msg.data.data()));

    cv::Mat new_grid_map(h, w, CV_32FC1);
    cv::Mat new_grid_map_thresh(h, w, CV_8UC1);
    grid_map = new_grid_map;
    grid_map_thresh = new_grid_map_thresh;

    local_occupied_counter.create(h, w, CV_32FC1);
    local_visit_counter.create(h, w, CV_32FC1);
    local_map_pt_mask.create(h, w, CV_8UC1);

    printf("norm_factor_x: %f\n", norm_factor_x);
    printf("norm_factor_z: %f\n", norm_factor_z);

    cout << "Extension done" << endl;
}
