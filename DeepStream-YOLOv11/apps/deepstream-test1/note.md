
apt install -y libcairo2-dev pkg-config python3-dev

pip3 install pyds-1.1.8-py3-none-linux_aarch64.whl

pip3 install cuda-python

apt install -y \
    libavcodec58 \
    libmpg123-0 \
    libmpeg2encpp-2.1-0 \
    libmpeg2-4 \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly
    
apt update && apt install v4l-utils

apt install graphviz

## RUN
```bash
# Run xhost - Run outside docker container
xhost +SI:localuser:root
xhost +SI:localuser:$(whoami)

# Install xrandr - Run inside docker container
apt-get install x11-xserver-utils

# Set display - Run inside docker container
export DISPLAY=:1

# query
xrandr --query

# Final: run app
```
v4l2-ctl -d /dev/video0 --list-formats-ext
gst-launch-1.0 v4l2src device=/dev/video4 ! videoconvert ! nv3dsink
gst-launch-1.0 v4l2src device=/dev/video4 ! "video/x-raw, width=640, height=360, framerate=10/1" ! videoconvert ! nv3dsink
gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! fakesink
gst-launch-1.0 v4l2src device=/dev/video0 ! "video/x-raw, width=640, height=480, framerate=30/1" ! nvvidconv ! "video/x-raw(memory:NVMM), format=NV12" ! nv3dsink
python3 deepstream_test_1_usb.py /dev/video0
GST_DEBUG=3 python3 deepstream_test_1.py /opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264


docker run     -it --rm     --runtime nvidia -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/video0 --device /dev/video2  --device /dev/video4   -v ./:/deepstream     -w /deepstream     --name deepstream-101     -p 8554:8554     deepstream-yolo:latest


docker run     -it --rm     --runtime nvidia -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/snd    -v ./:/deepstream     -w /deepstream     --name deepstream-101     -p 8554:8554     deepstream-yolo:latest



docker run     -it --rm     --runtime nvidia -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/video0  -v ./:/deepstream     -w /deepstream     --name deepstream-101     -p 8554:8554     deepstream-yolo:latest