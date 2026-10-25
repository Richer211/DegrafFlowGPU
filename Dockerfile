FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# 安装构建依赖 + OpenCV 依赖
RUN apt-get update && apt-get install -y \
    build-essential cmake git unzip pkg-config \
    libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev \
    libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev \
    libjsoncpp-dev \
    binutils \
    python3-dev python3-numpy wget \
    && apt-get clean

# 构建 OpenCV + contrib 模块
WORKDIR /opt
RUN git clone --branch 4.2.0 https://github.com/opencv/opencv.git && \
    git clone --branch 4.2.0 https://github.com/opencv/opencv_contrib.git

RUN mkdir -p /opt/opencv/build && cd /opt/opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# 设置工作目录为项目挂载目录
WORKDIR /app

# 最后用 bash 启动容器（由你自己手动构建）
CMD ["/bin/bash"]
