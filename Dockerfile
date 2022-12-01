FROM ros:melodic

ENV CATKIN_WS=/root/catkin_ws
ENV N_PROC = 2

RUN apt-get update && apt-get install -y \
    cmake \
    python-catkin-tools \    
    ros-melodic-image-geometry \
    ros-melodic-pcl-ros \
    ros-melodic-image-proc \
    ros-melodic-tf-conversions \
    ros-melodic-cv-bridge \
    ros-melodic-tf2-geometry-msgs  

# OpenCV 
RUN apt-get install -y software-properties-common
RUN add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"

RUN apt-get update && apt-get install -y \
    build-essential \
    qt5-default libvtk6-dev \
    zlib1g-dev libjpeg-dev libwebp-dev libpng-dev libtiff5-dev libjasper1 libjasper-dev libopenexr-dev libgdal-dev \
    libtbb-dev libeigen3-dev \
    python-dev python-tk python-numpy python3-dev python3-tk python3-numpy \
    ant default-jdk \
    doxygen \
    unzip wget 
    
RUN wget https://github.com/opencv/opencv/archive/3.2.0.zip -O OpenCV320.zip 
RUN unzip OpenCV320.zip 
RUN cd opencv-3.2.0 &&\
    mkdir build &&\
    cd build &&\
    cmake -DWITH_QT=ON -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_TBB=ON -DWITH_GDAL=ON -DWITH_XINE=ON -DBUILD_EXAMPLES=ON -DENABLE_PRECOMPILED_HEADERS=OFF .. &&\
    make -j4 &&\
    make install &&\
    ldconfig

# Setup catkin workspace
RUN mkdir -p $CATKIN_WS/src/visual_occupancy_map/ && \
    cd ${CATKIN_WS} && \
    catkin init && \
    catkin config \
    --extend /opt/ros/melodic \
    --cmake-args \
    -DCMAKE_BUILD_TYPE=Release && \
    catkin config --merge-devel

COPY . $CATKIN_WS/src/visual_occupancy_map/

RUN rosdep update

WORKDIR ${CATKIN_WS}

RUN cd ${CATKIN_WS} && \ 
    catkin build -j2 && \
    sed -i '/exec "$@"/i \
    source "/root/catkin_ws/devel/setup.bash"' /ros_entrypoint.sh

RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc && \
    echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc

EXPOSE 11311







