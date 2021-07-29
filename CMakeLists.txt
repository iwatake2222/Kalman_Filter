cmake_minimum_required(VERSION 3.0)

# Create project
set(ProjectName "main")
project(${ProjectName})

# Create executable file
add_executable(${ProjectName}
    main.cpp
    simple_matrix.h
    kalman_filter.h
)

# For OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(${ProjectName} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${ProjectName} ${OpenCV_LIBS})


