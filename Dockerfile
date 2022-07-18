FROM debian:stable-slim
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update -y && apt-get install -y build-essential \
  pkg-config \
  gcc \
  cmake \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev \
  libv4l-dev \
  libxvidcore-dev \
  libx264-dev \
  libjpeg-dev \
  libpng-dev \
  libtiff-dev \
  libatlas-base-dev \
  libtbb2 \
  libtbb-dev \
  wget \
  curl \
  clang \
  unzip

RUN wget -O opencv.zip "https://github.com/opencv/opencv/archive/4.5.5.zip" \
    && unzip "opencv.zip" \
    && mkdir -p build && cd build

RUN cmake -DBUILD_opencv_java=OFF \
    -DWITH_QT=OFF -DWITH_GTK=OFF \
    -DBUILD_opencv_python=OFF \
    -DWITH_CUDA=OFF \
    -DWITH_OPENGL=OFF \
    -DWITH_OPENCL=ON \
    -DWITH_IPP=ON \
    -DWITH_TBB=ON \
    -DWITH_EIGEN=ON \
    -DWITH_V4L=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DOPENCV_GENERATE_PKGCONFIG=YES \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    "../opencv-4.5.5"

#RUN cmake --build . -j 10
RUN make install -j 10

RUN pkg-config --cflags opencv4 && pkg-config --libs opencv4

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"
WORKDIR /app
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN rm -rf ../opencv-4.5.5 && rm ../opencv.zip

COPY . .

RUN cargo build --release

#EXPOSE 5000
CMD ["./target/release/yolov5-api"]