# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ubuntu/OurTest/super_point

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/OurTest/super_point/build

# Include any dependencies generated for this target.
include CMakeFiles/demo.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/demo.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/demo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/demo.dir/flags.make

CMakeFiles/demo.dir/src/superpoint.cpp.o: CMakeFiles/demo.dir/flags.make
CMakeFiles/demo.dir/src/superpoint.cpp.o: ../src/superpoint.cpp
CMakeFiles/demo.dir/src/superpoint.cpp.o: CMakeFiles/demo.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/OurTest/super_point/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/demo.dir/src/superpoint.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/demo.dir/src/superpoint.cpp.o -MF CMakeFiles/demo.dir/src/superpoint.cpp.o.d -o CMakeFiles/demo.dir/src/superpoint.cpp.o -c /home/ubuntu/OurTest/super_point/src/superpoint.cpp

CMakeFiles/demo.dir/src/superpoint.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo.dir/src/superpoint.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/OurTest/super_point/src/superpoint.cpp > CMakeFiles/demo.dir/src/superpoint.cpp.i

CMakeFiles/demo.dir/src/superpoint.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo.dir/src/superpoint.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/OurTest/super_point/src/superpoint.cpp -o CMakeFiles/demo.dir/src/superpoint.cpp.s

CMakeFiles/demo.dir/src/demo_superpoint.cpp.o: CMakeFiles/demo.dir/flags.make
CMakeFiles/demo.dir/src/demo_superpoint.cpp.o: ../src/demo_superpoint.cpp
CMakeFiles/demo.dir/src/demo_superpoint.cpp.o: CMakeFiles/demo.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/OurTest/super_point/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/demo.dir/src/demo_superpoint.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/demo.dir/src/demo_superpoint.cpp.o -MF CMakeFiles/demo.dir/src/demo_superpoint.cpp.o.d -o CMakeFiles/demo.dir/src/demo_superpoint.cpp.o -c /home/ubuntu/OurTest/super_point/src/demo_superpoint.cpp

CMakeFiles/demo.dir/src/demo_superpoint.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo.dir/src/demo_superpoint.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/OurTest/super_point/src/demo_superpoint.cpp > CMakeFiles/demo.dir/src/demo_superpoint.cpp.i

CMakeFiles/demo.dir/src/demo_superpoint.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo.dir/src/demo_superpoint.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/OurTest/super_point/src/demo_superpoint.cpp -o CMakeFiles/demo.dir/src/demo_superpoint.cpp.s

# Object files for target demo
demo_OBJECTS = \
"CMakeFiles/demo.dir/src/superpoint.cpp.o" \
"CMakeFiles/demo.dir/src/demo_superpoint.cpp.o"

# External object files for target demo
demo_EXTERNAL_OBJECTS =

demo: CMakeFiles/demo.dir/src/superpoint.cpp.o
demo: CMakeFiles/demo.dir/src/demo_superpoint.cpp.o
demo: CMakeFiles/demo.dir/build.make
demo: /root/opencv_build/opencv/build/lib/libopencv_gapi.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_stitching.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_alphamat.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_aruco.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_barcode.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_bgsegm.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_bioinspired.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_ccalib.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_dnn_objdetect.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_dnn_superres.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_dpm.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_face.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_fuzzy.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_hdf.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_hfs.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_img_hash.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_intensity_transform.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_line_descriptor.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_mcc.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_quality.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_rapid.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_reg.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_rgbd.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_saliency.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_sfm.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_stereo.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_structured_light.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_superres.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_surface_matching.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_tracking.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_videostab.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_viz.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_wechat_qrcode.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_xfeatures2d.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_xobjdetect.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_xphoto.so.4.6.0
demo: /usr/lib/libvitis_ai_library-dpu_task.so.3.5.0
demo: /usr/lib/libvitis_ai_library-math.so.3.5.0
demo: /root/opencv_build/opencv/build/lib/libopencv_shape.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_datasets.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_plot.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_text.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_ml.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_phase_unwrapping.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_optflow.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_ximgproc.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_objdetect.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_photo.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_highgui.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_video.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_calib3d.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_dnn.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_features2d.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_flann.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_videoio.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_imgcodecs.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_imgproc.so.4.6.0
demo: /root/opencv_build/opencv/build/lib/libopencv_core.so.4.6.0
demo: /usr/lib/libvitis_ai_library-model_config.so.3.5.0
demo: /usr/lib/aarch64-linux-gnu/libprotobuf.so
demo: /usr/lib/libvart-runner.so.3.5.0
demo: /usr/lib/libvart-util.so.3.5.0
demo: /usr/lib/libxir.so.3.5.0
demo: /usr/lib/libunilog.so.3.5.0
demo: /root/glog_build/lib/libglog.so.0.5.0
demo: /usr/lib/aarch64-linux-gnu/libgflags.so.2.2.2
demo: CMakeFiles/demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/OurTest/super_point/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable demo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/demo.dir/build: demo
.PHONY : CMakeFiles/demo.dir/build

CMakeFiles/demo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/demo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/demo.dir/clean

CMakeFiles/demo.dir/depend:
	cd /home/ubuntu/OurTest/super_point/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/OurTest/super_point /home/ubuntu/OurTest/super_point /home/ubuntu/OurTest/super_point/build /home/ubuntu/OurTest/super_point/build /home/ubuntu/OurTest/super_point/build/CMakeFiles/demo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/demo.dir/depend

