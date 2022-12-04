#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include <ros/ros.h>
#include <ros/package.h>
#include <cv_bridge/cv_bridge.h>
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/PoseArray.h"
#include "nav_msgs/OccupancyGrid.h"
#include "nav_msgs/Odometry.h"
#include "sensor_msgs/PointCloud.h"
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
float scale_factor = 3;
float cloud_max_x = 10;
float cloud_min_x = -10.0;
float cloud_max_z = 16;
float cloud_min_z = -5;
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

bool make_submaps = false;
bool use_semantic_filter = false;

float grid_max_x, grid_min_x,grid_max_z, grid_min_z;
cv::Mat global_occupied_counter, global_visit_counter;
cv::Mat local_occupied_counter, local_visit_counter;
cv::Mat local_map_pt_mask;
cv::Mat grid_map, grid_map_int, grid_map_thresh;
cv::Mat semantic_map;
cv::Mat gauss_kernel;
cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
float norm_factor_x, norm_factor_z;
int h, w;
unsigned int n_kf_received;
bool loop_closure_being_processed = false;
ros::Publisher pub_grid_map;
ros::Publisher pub_traj_map;
ros::Publisher pub_semantic_map;
nav_msgs::OccupancyGrid grid_map_msg;
bool first_msg = true;
bool Tcw_form;
float kf_pos_x, kf_pos_z;
int kf_pos_grid_x, kf_pos_grid_z;
std::vector<cv::Point> trj;
std::vector<cv::Point> kf_poses;
std::map<std::vector<uchar>, cv::Mat> submaps;
std::vector<std::vector<uchar>> sem_filter_colors;

using namespace std;
void ptsKFCallback(const sensor_msgs::PointCloud::ConstPtr& MapPoints, const nav_msgs::Odometry::ConstPtr& Kf_pose, const sensor_msgs::Image::ConstPtr& Sem_mask);
void updateGridMap(const geometry_msgs::PoseArray::ConstPtr& pts_and_pose);
void resetGridMap(const geometry_msgs::PoseArray::ConstPtr& pts_and_pose);
void saveMap(unsigned int id = 0);
void ptCallback(const geometry_msgs::PoseArray::ConstPtr& pts_and_pose);
void loopClosingCallback(const geometry_msgs::PoseArray::ConstPtr& all_kf_and_pts);
void parseParams(int argc, char **argv);
void printParams();
void extensionMap(const bool& width, const bool& positive);
void processMapPt(const geometry_msgs::Point &curr_pt, cv::Mat &occupied,
	cv::Mat &visited, cv::Mat &pt_mask, int kf_pos_grid_x, int kf_pos_grid_z, unsigned int id);
void processMapPts(const std::vector<geometry_msgs::Pose> &pts, unsigned int n_pts,
	unsigned int start_id, int kf_pos_grid_x, int kf_pos_grid_z);
void getGridMap();
void initialSubmaps(int argc, char **argv);
void readSemFilter(int argc, char **argv);

//TODO .yaml reading
float fx_zed=524.73925804;
float cx_zed=637.481812;
float fy_zed=524.73952;
float cy_zed=347.481812;
int width = 1280;
int height = 720;

struct semanticMP{
    Eigen::Vector3d point;
    uchar red;
    uchar green;
    uchar blue;
};

int main(int argc, char **argv){
	ros::init(argc, argv, "Vision occupancy map");
	ros::start();

	parseParams(argc, argv);
	printParams();

	grid_max_x = cloud_max_x*scale_factor;
	grid_min_x = cloud_min_x*scale_factor;
	grid_max_z = cloud_max_z*scale_factor;
	grid_min_z = cloud_min_z*scale_factor;
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

	grid_map_msg.data.resize(h*w);
	grid_map_msg.info.width = w;
	grid_map_msg.info.height = h;
	grid_map_msg.info.resolution = 1.0/scale_factor;

	grid_map_int = cv::Mat(h, w, CV_8SC1, (char*)(grid_map_msg.data.data()));

	grid_map.create(h, w, CV_32FC1);
	grid_map_thresh.create(h, w, CV_8UC1);
	
    semantic_map.create(h, w, CV_8UC3);
    semantic_map.setTo(cv::Scalar(64, 64, 64));
	
	local_occupied_counter.create(h, w, CV_32FC1);
	local_visit_counter.create(h, w, CV_32FC1);
	local_map_pt_mask.create(h, w, CV_8UC1);

	gauss_kernel = cv::getGaussianKernel(gaussian_kernel_size, -1);

	norm_factor_x = float(grid_res_x - 1) / float(grid_max_x - grid_min_x);
	norm_factor_z = float(grid_res_z - 1) / float(grid_max_z - grid_min_z);
	printf("norm_factor_x: %f\n", norm_factor_x);
	printf("norm_factor_z: %f\n", norm_factor_z);

	ros::NodeHandle nodeHandler;

	ros::Subscriber sub_all_kf_and_pts = nodeHandler.subscribe("all_kf_and_pts", 1000, loopClosingCallback);

	message_filters::Subscriber<sensor_msgs::PointCloud> MapPoints_sub(nodeHandler, "/KF_map_points", 1000);
    message_filters::Subscriber<nav_msgs::Odometry> KF_pose_sub(nodeHandler, "KF_pose", 1000);
    message_filters::Subscriber<sensor_msgs::Image> Semantic_sub(nodeHandler, "/semantic_image", 1000);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud, nav_msgs::Odometry, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(1000),  MapPoints_sub, KF_pose_sub, Semantic_sub);
    sync.registerCallback(boost::bind(&ptsKFCallback,_1,_2, _3));

	pub_grid_map = nodeHandler.advertise<nav_msgs::OccupancyGrid>("/visual_occupancy_node/grid_map", 1000);
	pub_semantic_map = nodeHandler.advertise<sensor_msgs::Image>("/visual_occupancy_node/semantic_occupancy_map", 1000);
	if (publish_trajectory){
		pub_traj_map = nodeHandler.advertise<sensor_msgs::Image>("/visual_occupancy_node/trajectory_grid_map", 1000);
	}
	
	if (make_submaps){
		initialSubmaps(argc, argv);
	}
	if (use_semantic_filter){
		readSemFilter(argc, argv);
	}

	ros::spin();
	ros::shutdown();
	cv::destroyAllWindows();
	saveMap();
	
	return 0;
}

void ptsKFCallback(const sensor_msgs::PointCloud::ConstPtr& MapPoints, const nav_msgs::Odometry::ConstPtr& Kf_pose, const sensor_msgs::Image::ConstPtr& Sem_mask){
	if (loop_closure_being_processed){ return; }
	
    std_msgs::Header h1 = MapPoints->header;
	std_msgs::Header h2 = Kf_pose->header;
	cv::Mat bgr_sem_image = cv_bridge::toCvShare(Sem_mask, "rgb8")->image;
    cv::Mat sem_image;
    cv::cvtColor(bgr_sem_image, sem_image, cv::COLOR_BGR2RGB);

	geometry_msgs::PoseArray pts_and_pose;
	pts_and_pose.header.seq = h2.seq;
	geometry_msgs::Pose temp;
	geometry_msgs::Pose temp_sem;
    Eigen::Matrix3d Rwc;
    Eigen::Vector3d Pwc;
	if (Tcw_form){ 
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

		Pwc = - Rcw.transpose() * Pcw;

		temp.position.x = Pwc.x();
		temp.position.z = Pwc.y();
		temp.position.y = Pwc.z();
	
		Rwc = Rcw.transpose();

		Eigen::Quaterniond Qwc(Rwc);
	
		temp.orientation.w = Qwc.w();
		temp.orientation.x = Qwc.x();
		temp.orientation.y = Qwc.y();
		temp.orientation.z = Qwc.z();
	}
	else {
		
		temp.position.x = Kf_pose->pose.pose.position.x;
		temp.position.z = Kf_pose->pose.pose.position.y;
		temp.position.y = Kf_pose->pose.pose.position.z;

		temp.orientation.w = Kf_pose->pose.pose.orientation.w;
		temp.orientation.x = Kf_pose->pose.pose.orientation.x;
		temp.orientation.y = Kf_pose->pose.pose.orientation.y;
		temp.orientation.z = Kf_pose->pose.pose.orientation.z;

		Eigen::Quaterniond Q(temp.orientation.w,
						 temp.orientation.x,
						 temp.orientation.y,
						 temp.orientation.z);
	
		Rwc = Q.toRotationMatrix();
        Pwc[0] = temp.position.x;
		Pwc[1] = temp.position.y;
        Pwc[2] = temp.position.z;
	}
	
	if (first_msg){
		// Set coordinate system origin
		grid_map_msg.info.origin.position.x = temp.position.x + cloud_min_x;
		grid_map_msg.info.origin.position.y = temp.position.z + cloud_min_z;
		grid_map_msg.info.origin.position.z = 0;
		
		grid_map_msg.info.origin.orientation.w = 1;
		grid_map_msg.info.origin.orientation.x = 0;
		grid_map_msg.info.origin.orientation.y = 0;
		grid_map_msg.info.origin.orientation.z = 0;
		first_msg =false;

	}
	
	pts_and_pose.poses.push_back(temp);
	vector <semanticMP> sem_MPs;

	for(auto mp:MapPoints->points){
		Eigen::Vector3d point(mp.x,mp.y,mp.z);
		if(point.norm()==0)
			continue;
		if(!(lower_height == 0 && upper_height == 0)){
			if(mp.z < lower_height || mp.z > upper_height)
				continue;
		}
		
        // find corresping pixel in semantic image  
		Eigen::Vector3d  Mp_Pcw = Rwc.transpose() * (point - Pwc);
        
        float X = Mp_Pcw[0];
        float Y = Mp_Pcw[1];
        float Z = Mp_Pcw[2];

        int u=int(((fx_zed*X)/Z)+cx_zed);
        int v=int(((fy_zed*Y)/Z)+cy_zed);

        if (u > 0 && u <= width && v > 0 && v <= height){
            // get pixel
            cv::Vec3b color = sem_image.at<cv::Vec3b>(v - 1, u - 1);
			if (use_semantic_filter){	
				vector<uchar> ucolor;
				ucolor.push_back(color[2]);
				ucolor.push_back(color[1]);
				ucolor.push_back(color[0]);
				bool found = false;
				for (const auto& i : sem_filter_colors){
					if (i == ucolor){
						found = true;
						sem_MPs.push_back({point, color[0], color[1], color[2]});
					}
				}
				if (!found){
					continue;
				}
			}		
            else {
				sem_MPs.push_back({point, color[0], color[1], color[2]});
			}
        } else {
			continue;
		}
		
		geometry_msgs::Pose tempPt;
		tempPt.position.x = mp.x;
		tempPt.position.z = mp.y;
		tempPt.position.y = mp.z;
		pts_and_pose.poses.push_back(tempPt);
	}

	if (temp.position.x > (cloud_max_x - thresh_extension)){	
		cloud_max_x = step_extension + cloud_max_x;
		extensionMap(true, true);
	}
	if (temp.position.x < (cloud_min_x + thresh_extension)){
		cloud_min_x = -step_extension + cloud_min_x;
		extensionMap(true, false);
	}	
	if (temp.position.z > (cloud_max_z - thresh_extension)){	
		cloud_max_z = step_extension + cloud_max_z;
		extensionMap(false, true);
	}
	if (temp.position.z < (cloud_min_z + thresh_extension)){		
		cloud_min_z = -step_extension + cloud_min_z;
		extensionMap(false, false);
	}	

	boost::shared_ptr<geometry_msgs::PoseArray> pts_and_pose_temp = boost::make_shared<geometry_msgs::PoseArray>(pts_and_pose);	
	updateGridMap(pts_and_pose_temp);
	 
    for (semanticMP sem_MP : sem_MPs){
        float pt_pos_x = sem_MP.point[0] * scale_factor;
	    float pt_pos_z = sem_MP.point[1] * scale_factor; // switch Y axe to Z axe

	    int pt_pos_grid_x = int(floor((pt_pos_x - grid_min_x) * norm_factor_x));
	    int pt_pos_grid_z = int(floor((pt_pos_z - grid_min_z) * norm_factor_z));

	    if (pt_pos_grid_x < 0 || pt_pos_grid_x >= w)
		    return;

	    if (pt_pos_grid_z < 0 || pt_pos_grid_z >= h)
		    return; 
		
		if (global_visit_counter.at<int>(pt_pos_grid_z, pt_pos_grid_x) > visit_thresh){
			// semantic occupancy grid map
			if (grid_map_thresh.at<uchar>(pt_pos_grid_z, pt_pos_grid_x) != 255){
			cv::circle(semantic_map,cv::Point(pt_pos_grid_x, pt_pos_grid_z),3,cv::Scalar(sem_MP.blue, sem_MP.green, sem_MP.red),-1);
			}
			// submaps
			if (make_submaps){
				vector<uchar> color;
				color.push_back(sem_MP.blue);
				color.push_back(sem_MP.green);
				color.push_back(sem_MP.red);
				if (submaps.find(color) != submaps.end()){
					cv::circle(submaps[color],cv::Point(pt_pos_grid_x, pt_pos_grid_z),1,cv::Scalar(sem_MP.blue, sem_MP.green, sem_MP.red),-1);
				}
			}
		}
    }
    ros::Time time = ros::Time::now();
    cv_ptr->encoding = "rgb8";
    cv_ptr->header.stamp = time;
    cv_ptr->header.frame_id = "semantic_grid_map";
    cv_ptr->image = semantic_map;
    pub_semantic_map.publish(cv_ptr->toImageMsg());

	grid_map_msg.info.map_load_time = ros::Time::now();
	pub_grid_map.publish(grid_map_msg);
}

void saveMap(unsigned int id) {
	printf("saving maps with id: %u\n", id);
	mkdir("results", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	mkdir("submaps", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if (id > 0) {
		cv::imwrite("results//grid_map_" + to_string(id) + ".jpg", grid_map);
		cv::imwrite("results//grid_map_thresh_" + to_string(id) + ".jpg", grid_map_thresh);
		cv::Mat rgb_map;
		cv::cvtColor(semantic_map, rgb_map, cv::COLOR_BGR2RGB);
		cv::imwrite("results//semantic_map_.jpg", rgb_map);
		
		std::vector<int> compression_params; 
		compression_params.push_back(cv::IMWRITE_PXM_BINARY);
		compression_params.push_back(0); 
		const std::string imageFilename = "results//grid_map_navigation" + to_string(id) + ".pgm"; 

		cv::imwrite(imageFilename, grid_map_thresh, compression_params); 

	}
	else {
		cv::imwrite("results//grid_map.jpg", grid_map);
		cv::imwrite("results//grid_map_thresh.jpg", grid_map_thresh);
		cv::Mat rgb_map;
		cv::cvtColor(semantic_map, rgb_map, cv::COLOR_BGR2RGB);
		cv::imwrite("results//semantic_map_.jpg", rgb_map);

		std::vector<int> compression_params; 
		compression_params.push_back(cv::IMWRITE_PXM_BINARY);
		compression_params.push_back(0); 
		const std::string imageFilename = "results//grid_map_navigation.pgm"; 

		cv::imwrite(imageFilename, grid_map_thresh, compression_params); 
		
		for (auto const& submap : submaps){
			cv::Scalar color(submap.first[0], submap.first[1], submap.first[2]);
			int first = color[0];
			int second = color[1];
			int third = color[2];
			cv::Mat rgb_map;
			cv::cvtColor(submap.second, rgb_map, cv::COLOR_BGR2RGB);
			cv::filter2D(rgb_map, rgb_map, CV_32F, gauss_kernel);
			cv::imwrite("submaps//submap_" + to_string(first) + "_" + to_string(second)  + "_" + to_string(third) + ".jpg", rgb_map);
			cout << "map saved with " + to_string(first) + "_" + to_string(second) + "_" + to_string(third) + " name" << endl;
			
		}
	}

}
void ptCallback(const geometry_msgs::PoseArray::ConstPtr& pts_and_pose){
	if (loop_closure_being_processed){ return; }

	updateGridMap(pts_and_pose);
	grid_map_msg.info.map_load_time = ros::Time::now();
	pub_grid_map.publish(grid_map_msg);
}

void loopClosingCallback(const geometry_msgs::PoseArray::ConstPtr& all_kf_and_pts){
	loop_closure_being_processed = true;
	resetGridMap(all_kf_and_pts);
	loop_closure_being_processed = false;
}

void processMapPt(const geometry_msgs::Point &curr_pt, cv::Mat &occupied, 
	cv::Mat &visited, cv::Mat &pt_mask, int kf_pos_grid_x, int kf_pos_grid_z, unsigned int pt_id) {
	float pt_pos_x = curr_pt.x*scale_factor;
	float pt_pos_z = curr_pt.z*scale_factor;

	int pt_pos_grid_x = int(floor((pt_pos_x - grid_min_x) * norm_factor_x));
	int pt_pos_grid_z = int(floor((pt_pos_z - grid_min_z) * norm_factor_z));

	if (pt_pos_grid_x < 0 || pt_pos_grid_x >= w)
		return;

	if (pt_pos_grid_z < 0 || pt_pos_grid_z >= h)
		return;
	++occupied.at<int>(pt_pos_grid_z, pt_pos_grid_x);
	pt_mask.at<uchar>(pt_pos_grid_z, pt_pos_grid_x) = 255;
	// Get all grid cell that the line between keyframe and map point pass through
	int x0 = kf_pos_grid_x;
	int y0 = kf_pos_grid_z;
	int x1 = pt_pos_grid_x;
	int y1 = pt_pos_grid_z;
	bool steep = (abs(y1 - y0) > abs(x1 - x0));
	if (steep){
		swap(x0, y0);
		swap(x1, y1);
	}
	if (x0 > x1){
		swap(x0, x1);
		swap(y0, y1);
	}
	int dx = x1 - x0;
	int dy = abs(y1 - y0);
	double error = 0;
	double deltaerr = ((double)dy) / ((double)dx);
	int y = y0;
	int ystep = (y0 < y1) ? 1 : -1;
	for (int x = x0; x <= x1; ++x){
		if (steep) {
			++visited.at<int>(x, y);
		}
		else {
			++visited.at<int>(y, x);
		}
		error = error + deltaerr;
		if (error >= 0.5){
			y = y + ystep;
			error = error - 1.0;
		}
	}
}

void initialSubmaps(int argc, char **argv){
	ifstream in(argv[2]);
	string line;
	if (in){
		while (getline(in, line)){
			cv::Mat new_map(h, w, CV_8UC3);
			new_map.setTo(cv::Scalar(127,127,127));
			stringstream s(line);
			std::vector<uchar> color;
			int red;
			int green;
			int blue;
			s >> red;
			s >> green;
			s >> blue;
			uchar uc_red = (uchar) red;
			uchar uc_green = (uchar) green;
			uchar uc_blue = (uchar) blue;
			color.push_back(uc_red);
			color.push_back(uc_green);
			color.push_back(uc_blue);
			submaps[color] = new_map;
		}	
	}	
}

void readSemFilter(int argc, char **argv){
	ifstream in(argv[3]);
	string line;
	if (in){
		while (getline(in, line)){
			stringstream s(line);
			std::vector<uchar> color;
			int red;
			int green;
			int blue;
			s >> red;
			s >> green;
			s >> blue;
			uchar uc_red = (uchar) red;
			uchar uc_green = (uchar) green;
			uchar uc_blue = (uchar) blue;
			color.push_back(uc_red);
			color.push_back(uc_green);
			color.push_back(uc_blue);
			sem_filter_colors.push_back(color);
		}
	}
}

void processMapPts(const std::vector<geometry_msgs::Pose> &pts, unsigned int n_pts,
	unsigned int start_id, int kf_pos_grid_x, int kf_pos_grid_z) {
	unsigned int end_id = start_id + n_pts;
	if (use_local_counters) {
		local_map_pt_mask.setTo(0);
		local_occupied_counter.setTo(0);
		local_visit_counter.setTo(0);
		for (unsigned int pt_id = start_id; pt_id < end_id; ++pt_id){
			processMapPt(pts[pt_id].position, local_occupied_counter, local_visit_counter,
				local_map_pt_mask, kf_pos_grid_x, kf_pos_grid_z, pt_id - start_id);
		}
		for (int row = 0; row < h; ++row){
			for (int col = 0; col < w; ++col){
				if (local_map_pt_mask.at<uchar>(row, col) == 0) {
					local_occupied_counter.at<int>(row, col) = 0;
				}
				else {
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

	}
	else {
		for (unsigned int pt_id = start_id; pt_id < end_id; ++pt_id){
			processMapPt(pts[pt_id].position, global_occupied_counter, global_visit_counter,
				local_map_pt_mask, kf_pos_grid_x, kf_pos_grid_z, pt_id - start_id);
		}
	}
}

void updateGridMap(const geometry_msgs::PoseArray::ConstPtr& pts_and_pose){
	const geometry_msgs::Point &kf_location = pts_and_pose->poses[0].position;

	kf_pos_x = kf_location.x*scale_factor;
	kf_pos_z = kf_location.z*scale_factor;
	kf_poses.push_back(cv::Point(kf_pos_x, kf_pos_z));

	kf_pos_grid_x = int(floor((kf_pos_x - grid_min_x) * norm_factor_x));
	kf_pos_grid_z = int(floor((kf_pos_z - grid_min_z) * norm_factor_z));

	if (kf_pos_grid_x < 0 || kf_pos_grid_x >= w)
		return;

	if (kf_pos_grid_z < 0 || kf_pos_grid_z >= h)
		return;
	++n_kf_received;
	unsigned int n_pts = pts_and_pose->poses.size() - 1;
	// printf("Processing key frame %u and %u points\n",n_kf_received, n_pts);
	processMapPts(pts_and_pose->poses, n_pts, 1, kf_pos_grid_x, kf_pos_grid_z);
	getGridMap();
	
	// Publish trajectory image
	if (publish_trajectory){
		cv::Mat traj = grid_map_thresh.clone();
		trj.push_back(cv::Point(kf_pos_grid_x,kf_pos_grid_z));
		for(size_t i=0;i<trj.size();i++){
			cv::circle(traj,trj[i],2,cv::Scalar(0),-1);
			if(i!=0)
				cv::line(traj, trj[i-1], trj[i],cv::Scalar(0), 2);
		}
	
		ros::Time time = ros::Time::now();
        cv_ptr->encoding = "mono8";
        cv_ptr->header.stamp = time;
        cv_ptr->header.frame_id = "trajectory_grid_map";
        cv_ptr->image = traj;
        pub_traj_map.publish(cv_ptr->toImageMsg());
	}
	
	
}

void resetGridMap(const geometry_msgs::PoseArray::ConstPtr& all_kf_and_pts){
	global_visit_counter.setTo(0);
	global_occupied_counter.setTo(0);

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
	for (unsigned int kf_id = 0; kf_id < n_kf; ++kf_id){
		const geometry_msgs::Point &kf_location = all_kf_and_pts->poses[++id].position;
		//const geometry_msgs::Quaternion &kf_orientation = pts_and_pose->poses[0].orientation;
		unsigned int n_pts = all_kf_and_pts->poses[++id].position.x;
		if ((unsigned int)(all_kf_and_pts->poses[id].position.y) != n_pts ||
			(unsigned int)(all_kf_and_pts->poses[id].position.z) != n_pts) {
			printf("resetGridMap :: Unexpected formatting in the point count element for keyframe %d\n", kf_id);
			return;
		}
		float kf_pos_x = kf_location.x*scale_factor;
		float kf_pos_z = kf_location.z*scale_factor;

		int kf_pos_grid_x = int(floor((kf_pos_x - grid_min_x) * norm_factor_x));
		int kf_pos_grid_z = int(floor((kf_pos_z - grid_min_z) * norm_factor_z));

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
	double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
	printf("Done. Time taken: %f secs\n", ttrack);
	pub_grid_map.publish(grid_map_msg);
}

void getGridMap() {
	double 	unknown_region_ratio = (free_thresh + occupied_thresh) / 2.0;
	for (int row = 0; row < h; ++row){
		for (int col = 0; col < w; ++col){
			int visits = global_visit_counter.at<int>(row, col);
			int occupieds = global_occupied_counter.at<int>(row, col);

			if (visits <= visit_thresh){
				grid_map.at<float>(row, col) = unknown_region_ratio;
			}
			else {
				grid_map.at<float>(row, col) = 1.0 - float(occupieds / visits);
			}
			if (grid_map.at<float>(row, col) >= free_thresh) {
				grid_map_thresh.at<uchar>(row, col) = 255;
				grid_map_int.at<char>(row, col) = (1 - grid_map.at<float>(row, col)) * 100;
				semantic_map.at<cv::Vec3b>(row, col)[0] =  255;
                semantic_map.at<cv::Vec3b>(row, col)[1] =  255;
                semantic_map.at<cv::Vec3b>(row, col)[2] =  255;
			}
			else if (grid_map.at<float>(row, col) < free_thresh && grid_map.at<float>(row, col) >= occupied_thresh) {
				grid_map_thresh.at<uchar>(row, col) = 128;
				grid_map_int.at<char>(row, col) = -1;
			}
			else {
				grid_map_thresh.at<uchar>(row, col) = 0;
				grid_map_int.at<char>(row, col) = (1 - grid_map.at<float>(row, col)) * 100;
			}
			
		}
	}
	if (use_boundary_detection) {
		cv::Mat canny_output;
		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::Canny(grid_map_thresh, canny_output, canny_thresh, canny_thresh * 2, 3);
		for (int row = 0; row < h; ++row){
			for (int col = 0; col < w; ++col){
				if (canny_output.at<uchar>(row, col)>0) {
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
	if (!fs.isOpened())
    {
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
	fs["Step_extension"] >> step_extension;h;
	fs["Thresh_extension"] >> thresh_extension;
	fs["Gaussian_kernel_size"] >> gaussian_kernel_size;
	fs["Use_boundary_detection"] >> use_boundary_detection;
	fs["Canny_thresh"] >> canny_thresh;
	fs["Upper_height"] >> upper_height;
    fs["Lower_height"] >> lower_height;
	fs["Make_submaps"] >> make_submaps;
	fs["Use_semantic_filter"] >> use_semantic_filter;
}

void printParams() {
	printf("Using params:\n");
	printf("scale_factor: %f\n", scale_factor);
	printf("cloud_max: %f, %f\t cloud_min: %f, %f\n", cloud_max_x, cloud_max_z, cloud_min_x, cloud_min_z);
	printf("free_thresh: %f\n", free_thresh);
	printf("occupied_thresh: %f\n", occupied_thresh);
	printf("use_local_counters: %d\n", use_local_counters);
	printf("use_gaussian_counters: %d\n", use_gaussian_counters);
	printf("make_submaps: %d\n", make_submaps);
	printf("use_semantic_filter: %d\n", use_semantic_filter);
	printf("visit_thresh: %d\n", visit_thresh);
	printf("step of extension: %f\n", step_extension);
	printf("thresh of extension: %f\n", thresh_extension);
	printf("publish_trajectory: %d\n", publish_trajectory);
}	


void extensionMap(const bool& width, const bool& positive){	
	printf("New height and width set \n");
	grid_max_x = cloud_max_x*scale_factor;
	grid_min_x = cloud_min_x*scale_factor;
	grid_max_z = cloud_max_z*scale_factor;
	grid_min_z = cloud_min_z*scale_factor;

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
	
    cv::Mat new_semantic_map(h, w, CV_8UC3);
    new_semantic_map.setTo(cv::Scalar(64, 64, 64));
	
	// Positive Axe X and Positive Axe Z
	if (positive){
		if (width){
			cout << "Positive Axe X extension" << endl;
		} else {
			cout << "Positive Axe Z extension" << endl;
		}
		
		global_occupied_counter.copyTo(new_global_occupied_counter(cv::Rect(0, 0, global_occupied_counter.cols, global_occupied_counter.rows)));
		global_visit_counter.copyTo(new_global_visit_counter(cv::Rect(0, 0, global_visit_counter.cols, global_visit_counter.rows)));
        semantic_map.copyTo(new_semantic_map(cv::Rect(0, 0,semantic_map.cols, semantic_map.rows)));
		if (make_submaps){
			for (auto&  submap : submaps){
				cv::Mat new_submap(h, w, CV_8UC3);
				new_submap.setTo(cv::Scalar(128, 128, 128));
				submap.second.copyTo(new_submap(cv::Rect(0,0, submap.second.cols, submap.second.rows)));
				submap.second = new_submap;
			}
		}
	}
	// Negative Axe X
	if (width && !positive){
		cout << "Negative Axe X extension" << endl;
		global_occupied_counter.copyTo(new_global_occupied_counter(cv::Rect(step_extension * scale_factor, 0, global_occupied_counter.cols, global_occupied_counter.rows)));
		global_visit_counter.copyTo(new_global_visit_counter(cv::Rect(step_extension * scale_factor, 0, global_visit_counter.cols, global_visit_counter.rows)));
		semantic_map.copyTo(new_semantic_map(cv::Rect(step_extension * scale_factor, 0,semantic_map.cols, semantic_map.rows)));
		if (make_submaps){
			for (auto&  submap : submaps){
				cv::Mat new_submap(h, w, CV_8UC3);
				new_submap.setTo(cv::Scalar(128, 128, 128));
				submap.second.copyTo(new_submap(cv::Rect(step_extension * scale_factor,0, submap.second.cols, submap.second.rows)));
				submap.second = new_submap;
			}
		}
        grid_map_msg.info.origin.position.x -=  step_extension;
		if (publish_trajectory){
			trj.clear();
			for (cv::Point& pos : kf_poses){
				trj.push_back(cv::Point(int(floor((pos.x - grid_min_x) * norm_factor_x)), int(floor((pos.y - grid_min_z) * norm_factor_z))));
			}
		}
	}
	// Negative Axe Z
	if (!width && !positive){
		cout << "Negative Axe Z extension" << endl;
		global_occupied_counter.copyTo(new_global_occupied_counter(cv::Rect(0, step_extension * scale_factor, global_occupied_counter.cols, global_occupied_counter.rows)));
		global_visit_counter.copyTo(new_global_visit_counter(cv::Rect(0, step_extension * scale_factor, global_visit_counter.cols, global_visit_counter.rows)));
		semantic_map.copyTo(new_semantic_map(cv::Rect(0, step_extension * scale_factor, semantic_map.cols, semantic_map.rows)));
        if (make_submaps){
			for (auto&  submap : submaps){
				cv::Mat new_submap(h, w, CV_8UC3);
				new_submap.setTo(cv::Scalar(128, 128, 128));
				submap.second.copyTo(new_submap(cv::Rect(0,step_extension * scale_factor, submap.second.cols, submap.second.rows)));
				submap.second = new_submap;
			}
		}
		grid_map_msg.info.origin.position.y -=  step_extension;
		if (publish_trajectory){
			trj.clear();
			for (cv::Point& pos : kf_poses){
				trj.push_back(cv::Point(int(floor((pos.x - grid_min_x) * norm_factor_x)), int(floor((pos.y - grid_min_z) * norm_factor_z))));
			}
		}
	}
	global_occupied_counter = new_global_occupied_counter;
	global_visit_counter = new_global_visit_counter;
    semantic_map = new_semantic_map;
	grid_map_msg.data.resize(h*w);
	grid_map_msg.info.width = w;
	grid_map_msg.info.height = h;
	grid_map_int = cv::Mat(h, w, CV_8SC1, (char*)(grid_map_msg.data.data()));
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
