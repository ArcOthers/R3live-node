/* 
This code is the implementation of our paper "R3LIVE: A Robust, Real-time, RGB-colored, 
LiDAR-Inertial-Visual tightly-coupled state Estimation and mapping package".

Author: Jiarong Lin   < ziv.lin.ljr@gmail.com >

If you use any code of this repo in your academic research, please cite at least
one of our papers:
[1] Lin, Jiarong, and Fu Zhang. "R3LIVE: A Robust, Real-time, RGB-colored, 
    LiDAR-Inertial-Visual tightly-coupled state Estimation and mapping package." 
[2] Xu, Wei, et al. "Fast-lio2: Fast direct lidar-inertial odometry."
[3] Lin, Jiarong, et al. "R2LIVE: A Robust, Real-time, LiDAR-Inertial-Visual
     tightly-coupled state Estimator and mapping." 
[4] Xu, Wei, and Fu Zhang. "Fast-lio: A fast, robust lidar-inertial odometry 
    package by tightly-coupled iterated kalman filter."
[5] Cai, Yixi, Wei Xu, and Fu Zhang. "ikd-Tree: An Incremental KD Tree for 
    Robotic Applications."
[6] Lin, Jiarong, and Fu Zhang. "Loam-livox: A fast, robust, high-precision 
    LiDAR odometry and mapping package for LiDARs of small FoV."

For commercial use, please contact me < ziv.lin.ljr@gmail.com > and
Dr. Fu Zhang < fuzhang@hku.hk >.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
 3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from this
    software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.
*/
#include "r3live.hpp"
// #include "photometric_error.hpp"
#include "tools_mem_used.h"
#include "tools_logger.hpp"

Common_tools::Cost_time_logger              g_cost_time_logger;
std::shared_ptr< Common_tools::ThreadPool > m_thread_pool_ptr;  //智能指针，指向线程池
double                                      g_vio_frame_cost_time = 0;
double                                      g_lio_frame_cost_time = 0;
int                                         g_flag_if_first_rec_img = 1;
#define DEBUG_PHOTOMETRIC 0
#define USING_CERES 0
void dump_lio_state_to_log( FILE *fp )   //打印状态
{
    if ( fp != nullptr && g_camera_lidar_queue.m_if_dump_log )//如果启用日志记录：
    {
        Eigen::Vector3d rot_angle = Sophus::SO3d( Eigen::Quaterniond( g_lio_state.rot_end ) ).log();
        Eigen::Vector3d rot_ext_i2c_angle = Sophus::SO3d( Eigen::Quaterniond( g_lio_state.rot_ext_i2c ) ).log();
        fprintf( fp, "%lf ", g_lio_state.last_update_time - g_camera_lidar_queue.m_first_imu_time ); // Time   [0]
        fprintf( fp, "%lf %lf %lf ", rot_angle( 0 ), rot_angle( 1 ), rot_angle( 2 ) );               // Angle  [1-3]
        fprintf( fp, "%lf %lf %lf ", g_lio_state.pos_end( 0 ), g_lio_state.pos_end( 1 ),
                 g_lio_state.pos_end( 2 ) );          // Pos    [4-6]
        fprintf( fp, "%lf %lf %lf ", 0.0, 0.0, 0.0 ); // omega  [7-9]
        fprintf( fp, "%lf %lf %lf ", g_lio_state.vel_end( 0 ), g_lio_state.vel_end( 1 ),
                 g_lio_state.vel_end( 2 ) );          // Vel    [10-12]
        fprintf( fp, "%lf %lf %lf ", 0.0, 0.0, 0.0 ); // Acc    [13-15]
        fprintf( fp, "%lf %lf %lf ", g_lio_state.bias_g( 0 ), g_lio_state.bias_g( 1 ),
                 g_lio_state.bias_g( 2 ) ); // Bias_g [16-18]
        fprintf( fp, "%lf %lf %lf ", g_lio_state.bias_a( 0 ), g_lio_state.bias_a( 1 ),
                 g_lio_state.bias_a( 2 ) ); // Bias_a [19-21]
        fprintf( fp, "%lf %lf %lf ", g_lio_state.gravity( 0 ), g_lio_state.gravity( 1 ),
                 g_lio_state.gravity( 2 ) ); // Gravity[22-24]
        fprintf( fp, "%lf %lf %lf ", rot_ext_i2c_angle( 0 ), rot_ext_i2c_angle( 1 ),
                 rot_ext_i2c_angle( 2 ) ); // Rot_ext_i2c[25-27]
        fprintf( fp, "%lf %lf %lf ", g_lio_state.pos_ext_i2c( 0 ), g_lio_state.pos_ext_i2c( 1 ),
                 g_lio_state.pos_ext_i2c( 2 ) ); // pos_ext_i2c [28-30]
        fprintf( fp, "%lf %lf %lf %lf ", g_lio_state.cam_intrinsic( 0 ), g_lio_state.cam_intrinsic( 1 ), g_lio_state.cam_intrinsic( 2 ),
                 g_lio_state.cam_intrinsic( 3 ) );     // Camera Intrinsic [31-34]
        fprintf( fp, "%lf ", g_lio_state.td_ext_i2c ); // Camera Intrinsic [35]
        // cout <<  g_lio_state.cov.diagonal().transpose() << endl;
        // cout <<  g_lio_state.cov.block(0,0,3,3) << endl;
        for ( int idx = 0; idx < DIM_OF_STATES; idx++ ) // Cov    [36-64]
        {
            fprintf( fp, "%.9f ", sqrt( g_lio_state.cov( idx, idx ) ) );
        }
        fprintf( fp, "%lf %lf ", g_lio_frame_cost_time, g_vio_frame_cost_time ); // costime [65-66]
        fprintf( fp, "\r\n" );
        fflush( fp );
    }
}

//向字符串中添加一定数量的空格，使得字符串的长度达到指定的字符位置,为了对齐
double g_last_stamped_mem_mb = 0;
std::string append_space_to_bits( std::string & in_str, int bits )
{
    while( in_str.length() < bits )
    {
        in_str.append(" ");
    }
    return in_str;
}

//用于打印一个控制台仪表板，显示有关系统状态的各种信息（系统时间、LiDAR帧、相机帧、地图点数和内存使用）
void R3LIVE::print_dash_board()
{
#if DEBUG_PHOTOMETRIC
    return;
#endif
    int mem_used_mb = ( int ) ( Common_tools::get_RSS_Mb() ); //获取当前内存使用情况（以MB为单位）。
    // clang-format off
    if( (mem_used_mb - g_last_stamped_mem_mb < 1024 ) && g_last_stamped_mem_mb != 0 )//如果内存使用变化不大
    {
        cout  << ANSI_DELETE_CURRENT_LINE << ANSI_DELETE_LAST_LINE ;
    }
    else
    {
        cout << "\r\n" << endl;
        cout << ANSI_COLOR_WHITE_BOLD << "======================= R3LIVE Dashboard ======================" << ANSI_COLOR_RESET << endl;
        g_last_stamped_mem_mb = mem_used_mb ;//重新打印仪表盘 ，并记录内存使用量
    }
    std::string out_str_line_1, out_str_line_2;
    out_str_line_1 = std::string(        "| System-time | LiDAR-frame | Camera-frame |  Pts in maps | Memory used (Mb) |") ;
    //                                    1             16            30             45             60     
    // clang-format on
    out_str_line_2.reserve( 1e3 ); //reserve(1e3) 为 out_str_line_2 字符串分配了1000个字符的内存空间
    out_str_line_2.append( "|   " ).append( Common_tools::get_current_time_str() ); //系统时间
    append_space_to_bits( out_str_line_2, 14 ); //14个字符的位置，对齐输出
    out_str_line_2.append( "|    " ).append( std::to_string( g_LiDAR_frame_index ) ); //LiDAR帧
    append_space_to_bits( out_str_line_2, 28 );
    out_str_line_2.append( "|    " ).append( std::to_string( g_camera_frame_idx ) ); //相机帧
    append_space_to_bits( out_str_line_2, 43 );
    out_str_line_2.append( "| " ).append( std::to_string( m_map_rgb_pts.m_rgb_pts_vec.size() ) ); //地图点数
    append_space_to_bits( out_str_line_2, 58 );
    out_str_line_2.append( "|    " ).append( std::to_string( mem_used_mb ) ); //内存使用
    //使用ANSI颜色代码设置仪表板各部分的颜色:
    out_str_line_2.insert( 58, ANSI_COLOR_YELLOW, 7 );
    out_str_line_2.insert( 43, ANSI_COLOR_BLUE, 7 );
    out_str_line_2.insert( 28, ANSI_COLOR_GREEN, 7 );
    out_str_line_2.insert( 14, ANSI_COLOR_RED, 7 );
    out_str_line_2.insert( 0, ANSI_COLOR_WHITE, 7 );

    out_str_line_1.insert( 58, ANSI_COLOR_YELLOW_BOLD, 7 );
    out_str_line_1.insert( 43, ANSI_COLOR_BLUE_BOLD, 7 );
    out_str_line_1.insert( 28, ANSI_COLOR_GREEN_BOLD, 7 );
    out_str_line_1.insert( 14, ANSI_COLOR_RED_BOLD, 7 );
    out_str_line_1.insert( 0, ANSI_COLOR_WHITE_BOLD, 7 );
    //输出仪表板
    cout << out_str_line_1 << endl;
    cout << out_str_line_2 << ANSI_COLOR_RESET << "          ";
    ANSI_SCREEN_FLUSH;
}

//设置了状态变量的初始协方差矩阵
void R3LIVE::set_initial_state_cov( StatesGroup &state )
{
    // Set cov
    scope_color( ANSI_COLOR_RED_BOLD );
    state.cov = state.cov.setIdentity() * INIT_COV; //将 state.cov 矩阵初始化为单位矩阵，并乘以一个常数 INIT_COV 0.0001
    // state.cov.block(18, 18, 6 , 6 ) = state.cov.block(18, 18, 6 , 6 ) .setIdentity() * 0.1;
    // state.cov.block(24, 24, 5 , 5 ) = state.cov.block(24, 24, 5 , 5 ).setIdentity() * 0.001;
    state.cov.block( 0, 0, 3, 3 ) = mat_3_3::Identity() * 1e-5;   // R    设置了矩阵的前 3x3 块（即旋转矩阵的协方差）为单位矩阵乘以 1e-5，表示旋转矩阵的初始不确定性非常小
    state.cov.block( 3, 3, 3, 3 ) = mat_3_3::Identity() * 1e-5;   // T     设置了从第三行第三列开始的 3X3 的单位矩阵   -》平移
    state.cov.block( 6, 6, 3, 3 ) = mat_3_3::Identity() * 1e-5;   // vel    速度
    state.cov.block( 9, 9, 3, 3 ) = mat_3_3::Identity() * 1e-3;   // bias_g   陀螺仪偏置
    state.cov.block( 12, 12, 3, 3 ) = mat_3_3::Identity() * 1e-1; // bias_a  加速度计偏置
    state.cov.block( 15, 15, 3, 3 ) = mat_3_3::Identity() * 1e-5; // Gravity  重力
    state.cov( 24, 24 ) = 0.00001;   //设置矩阵第 24 行第 24 列的元素为 0.00001
    state.cov.block( 18, 18, 6, 6 ) = state.cov.block( 18, 18, 6, 6 ).setIdentity() *  1e-3; // Extrinsic between camera and IMU.设置从第 18 行第 18 列起的 6x6 块为单位矩阵乘以 1e-3，表示相机和 IMU 之间的外部标定的初始不确定性。
    state.cov.block( 25, 25, 4, 4 ) = state.cov.block( 25, 25, 4, 4 ).setIdentity() *  1e-3; // Camera intrinsic. 设置从第 25 行第 25 列起的 4x4 块为单位矩阵乘以 1e-3，表示相机内参的初始不确定性。
}

//用于这个函数service_VIO_update()，来产生一张图像，告诉用户交互操作
cv::Mat R3LIVE::generate_control_panel_img()
{
    int     line_y = 40;
    int     padding_x = 10;
    int     padding_y = line_y * 0.7;
    cv::Mat res_image = cv::Mat( line_y * 3 + 1 * padding_y, 960, CV_8UC3, cv::Scalar::all( 0 ) );//定义Mat大小，RGB
    char    temp_char[ 128 ];
    sprintf( temp_char, "Click this windows to enable the keyboard controls." ); //将那句话写入temp_char
    cv::putText( res_image, std::string( temp_char ), cv::Point( padding_x, line_y * 0 + padding_y ), cv::FONT_HERSHEY_COMPLEX, 1,
                 cv::Scalar( 0, 255, 255 ), 2, 8, 0 ); //将文本绘制到图上，“点击此窗口以启用键盘控制”
    sprintf( temp_char, "Press 'S' or 's' key to save current map" );
    cv::putText( res_image, std::string( temp_char ), cv::Point( padding_x, line_y * 1 + padding_y ), cv::FONT_HERSHEY_COMPLEX, 1,
                 cv::Scalar( 255, 255, 255 ), 2, 8, 0 );//同上，将文本绘制到图上  “按下 'S' 或 's' 键以保存当前地图”
    sprintf( temp_char, "Press 'space' key to pause the mapping process" );
    cv::putText( res_image, std::string( temp_char ), cv::Point( padding_x, line_y * 2 + padding_y ), cv::FONT_HERSHEY_COMPLEX, 1,
                 cv::Scalar( 255, 255, 255 ), 2, 8, 0 );//同上，将文本绘制到图上     “按下 'space' 键以暂停映射过程”
    return res_image;
}

//初始化相机的参数
void R3LIVE::set_initial_camera_parameter( StatesGroup &state, double *intrinsic_data, double *camera_dist_data, double *imu_camera_ext_R,
                                           double *imu_camera_ext_t, double cam_k_scale )
{
    scope_color( ANSI_COLOR_YELLOW_BOLD );
    // g_cam_K << 863.4241 / cam_k_scale, 0, 625.6808 / cam_k_scale,
    //     0, 863.4171 / cam_k_scale, 518.3392 / cam_k_scale,
    //     0, 0, 1;

    //从参数中获取内参和缩放因子来设置内参矩阵 g_cam_K
    g_cam_K << intrinsic_data[ 0 ] / cam_k_scale, intrinsic_data[ 1 ], intrinsic_data[ 2 ] / cam_k_scale, intrinsic_data[ 3 ],
        intrinsic_data[ 4 ] / cam_k_scale, intrinsic_data[ 5 ] / cam_k_scale, intrinsic_data[ 6 ], intrinsic_data[ 7 ], intrinsic_data[ 8 ];
    //畸变系数  g_cam_dist
    // camera_dist_data: 这是一个指向 double 数据的指针，假设它指向一个包含至少 5 个 double 元素的内存区域
    //igen::Matrix<double, 5, 1>: 这是一个 5 行 1 列的矩阵类型（即列向量）
    //Eigen::Map<Eigen::Matrix<double, 5, 1>>: 这是 Eigen 提供的映射类，它将 camera_dist_data 视作一个 5x1 的矩阵而不进行数据复制。
    g_cam_dist = Eigen::Map< Eigen::Matrix< double, 5, 1 > >( camera_dist_data );
    //设置IMU到相机的坐标变换外参（旋转和平移）
    state.rot_ext_i2c = Eigen::Map< Eigen::Matrix< double, 3, 3, Eigen::RowMajor > >( imu_camera_ext_R );
    state.pos_ext_i2c = Eigen::Map< Eigen::Matrix< double, 3, 1 > >( imu_camera_ext_t );
    // state.pos_ext_i2c.setZero();

    // Lidar to camera parameters.锁定互斥量 m_mutex_lio_process，以确保线程安全地保存 IMU 到相机的旋转矩阵和位置向量的初始值
    m_mutex_lio_process.lock();
    m_inital_rot_ext_i2c = state.rot_ext_i2c;  //传递数据
    m_inital_pos_ext_i2c = state.pos_ext_i2c;
    //将相机内存矩阵中数据传给系统状态变量
    state.cam_intrinsic( 0 ) = g_cam_K( 0, 0 );
    state.cam_intrinsic( 1 ) = g_cam_K( 1, 1 );
    state.cam_intrinsic( 2 ) = g_cam_K( 0, 2 );
    state.cam_intrinsic( 3 ) = g_cam_K( 1, 2 );
    //调用函数 set_initial_state_cov 来设置状态的初始协方差，这通常用于优化算法中的不确定性估计
    set_initial_state_cov( state );
    m_mutex_lio_process.unlock();
}

//在图像上绘制处理时间并发布出来
void R3LIVE::publish_track_img( cv::Mat &img, double frame_cost_time = -1 )
{
    cv_bridge::CvImage out_msg; //将 OpenCV 图像(cv::Mat) 转换为 ROS 图像消息的桥梁
    out_msg.header.stamp = ros::Time::now();               // Same timestamp and tf frame as input image  设置为当前时间
    out_msg.encoding = sensor_msgs::image_encodings::BGR8; // Or whatever表示图像的编码格式是 BGR8
    cv::Mat pub_image = img.clone(); //克隆图像
    //计算处理时间并在图像上绘制文本
    if ( frame_cost_time > 0 )
    {
        char fps_char[ 100 ];
        sprintf( fps_char, "Per-frame cost time: %.2f ms", frame_cost_time );//显示处理时间
        // sprintf(fps_char, "%.2f ms", frame_cost_time);

        if ( pub_image.cols <= 640 )//对于宽度小于或等于 640 像素的图像，文本字体大小为 1。
        {
            cv::putText( pub_image, std::string( fps_char ), cv::Point( 30, 30 ), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar( 255, 255, 255 ), 2, 8,
                         0 ); // 640 * 480
        }
        else if ( pub_image.cols > 640 )//对于宽度大于 640 像素的图像，文本字体大小为 2
        {
            cv::putText( pub_image, std::string( fps_char ), cv::Point( 30, 50 ), cv::FONT_HERSHEY_COMPLEX, 2, cv::Scalar( 255, 255, 255 ), 2, 8,
                         0 ); // 1280 * 1080
        }
    }
    out_msg.image = pub_image; // Your cv::Mat
    pub_track_img.publish( out_msg ); //发布图像消息到 ROS 中的 pub_track_img 话题
}

//发布原始图像
void R3LIVE::publish_raw_img( cv::Mat &img )
{
    cv_bridge::CvImage out_msg; //OpenCV 图像(cv::Mat) 转换为 ROS 图像消息
    out_msg.header.stamp = ros::Time::now();               // Same timestamp and tf frame as input image
    out_msg.encoding = sensor_msgs::image_encodings::BGR8; // Or whatever
    out_msg.image = img;                                   // Your cv::Mat
    pub_raw_img.publish( out_msg );
}

int        sub_image_typed = 0; // 0: TBD 1: sub_raw, 2: sub_comp
std::mutex mutex_image_callback;   //定义了一把图像数据的互斥锁

std::deque< sensor_msgs::CompressedImageConstPtr > g_received_compressed_img_msg; //存储压缩图像指针的队列
std::deque< sensor_msgs::ImageConstPtr >           g_received_img_msg;
std::shared_ptr< std::thread >                     g_thr_process_image;

//通过 m_thread_pool_ptr->commit_task 线程池的方式完成了图像buffer的压入与处理。在这个函数的末尾，会调用 process_image 函数
void R3LIVE::service_process_img_buffer()
{
    while ( 1 ) //不停循环处理传过来的图像数据
    {
        // To avoid uncompress so much image buffer, reducing the use of memory.
        //如果m_queue_image_with_pose队列内的数据 > 4，表示这些数据还没被处理，暂时挂起预处理线程（丢一些数据）
        if ( m_queue_image_with_pose.size() > 4 )
        {
            while ( m_queue_image_with_pose.size() > 4 )   //知道队列中数据小于四个，跳出循环
            {
                ros::spinOnce();  //ros::spinOnce() 主要用于在主循环中定期处理 ROS 的回调，以确保你的节点能够响应消息和事件，而不会在处理这些回调时被阻塞。
                std::this_thread::sleep_for( std::chrono::milliseconds( 2 ) );  //使当前线程暂停 2 毫秒，这样可以减少 CPU 占用
                std::this_thread::yield();  //它会请求操作系统将当前线程的执行权交给其他线程
            }
        }
        cv::Mat image_get;
        double  img_rec_time;

        // sub_image_typed == 2，表示接收的是压缩图像格式
        if ( sub_image_typed == 2 )  //当接受了压缩图像：
        {
            // 如果队列中没有数据，暂停当前线程1s，以减少CPU的使用
            while ( g_received_compressed_img_msg.size() == 0 )
            {
                ros::spinOnce();
                std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
                std::this_thread::yield();
            }
            //有数据了： 从队列的前端获取一个压缩图像消息msg
            sensor_msgs::CompressedImageConstPtr msg = g_received_compressed_img_msg.front();
            try   //尝试运行，报错则运行catch
            {
                // 将压缩图像消息转换为cv::Mat类型的图像数据
                cv_bridge::CvImagePtr cv_ptr_compressed = cv_bridge::toCvCopy( msg, sensor_msgs::image_encodings::BGR8 );
                // 存储获取的时间和图像
                img_rec_time = msg->header.stamp.toSec(); //获取当前图像的时间戳-》img_rec_time
                image_get = cv_ptr_compressed->image; //获取图像
                // 释放内存
                cv_ptr_compressed->image.release();
            }
            catch ( cv_bridge::Exception &e )   //捕捉 cv_bridge::toCvCopy 函数可能抛出的 cv_bridge::Exception 异常。
            {
                printf( "Could not convert from '%s' to 'bgr8' !!! ", msg->format.c_str() );
            }
            mutex_image_callback.lock(); //锁上，保护与unlock()之间的代码，其他资源无法操作该数据
            g_received_compressed_img_msg.pop_front();  //弹出最前面的一个压缩图像数据
            mutex_image_callback.unlock();  //解锁
        }
        else  //如果不是压缩数据：
        {
            // 如果队列中没有数据，暂停当前线程1s，以减少CPU的使用
            while ( g_received_img_msg.size() == 0 ) 
            {
                ros::spinOnce();
                std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
                std::this_thread::yield();
            }
            // 与前面的流程类似
            sensor_msgs::ImageConstPtr msg = g_received_img_msg.front(); //获取最前面的一个图像
            image_get = cv_bridge::toCvCopy( msg, sensor_msgs::image_encodings::BGR8 )->image.clone(); //将该图像转为opencv格式
            img_rec_time = msg->header.stamp.toSec(); //获取时间戳
            mutex_image_callback.lock();
            g_received_img_msg.pop_front(); //同样是要锁上并弹出最前面的图像
            mutex_image_callback.unlock();
        }
        process_image( image_get, img_rec_time ); //将这一帧数据处理完
    }
}

//通过一个线程获取图像信息
void R3LIVE::image_comp_callback( const sensor_msgs::CompressedImageConstPtr &msg )
{
    std::unique_lock< std::mutex > lock2( mutex_image_callback );//使用互斥量（mutex）来保护共享资源，确保在处理图像时不发生竞争条件。这使得在多线程环境中，这个回调函数是安全的。
    //使用unique_lock来管理 mutex锁    lock是unique_lock的实例化，
    //mutex_image_callback是一个std::mutex类型的变量，它代表了我们要保护的一把锁。   

    if ( sub_image_typed == 1 )  //
    {
        return; // Avoid subscribe the same image twice.
    }
    sub_image_typed = 2;
    g_received_compressed_img_msg.push_back( msg );
    // 如果是第一次收到图片，则启动一个线程，用来处理image_comp_callback回调中接收的压缩图片，内部其实在循环调用process_image()函数
    if ( g_flag_if_first_rec_img )
    {
        g_flag_if_first_rec_img = 0;
        // 通过线程池k方法调用service_process_img_buffer函数来处理图像
         // 内部其实在循环调用process_image()函数
        m_thread_pool_ptr->commit_task( &R3LIVE::service_process_img_buffer, this ); //图像处理线程器，，，，
    }
    return;
}

// ANCHOR - image_callback   接收和处理来自ROS的话题的图像消息
void R3LIVE::image_callback( const sensor_msgs::ImageConstPtr &msg )
{
    std::unique_lock< std::mutex > lock( mutex_image_callback );//使用互斥量（mutex）来保护共享资源，确保在处理图像时不发生竞争条件。这使得在多线程环境中，这个回调函数是安全的。
    //使用unique_lock来管理 mutex    lock是unique_lock的实例化，
    //mutex_image_callback是一个std::mutex类型的变量，它代表了我们要保护的资源或代码段的锁。   

    if ( sub_image_typed == 2 )  //检查是否处理过
    {
        return; // Avoid subscribe the same image twice.
    }
    sub_image_typed = 1; //设置为1，其他函数就不处理了

    if ( g_flag_if_first_rec_img )  //检测是否是第一次接受图像
    {
        g_flag_if_first_rec_img = 0;   //标记为已接收
        m_thread_pool_ptr->commit_task( &R3LIVE::service_process_img_buffer, this );//图像处理线程器，，，，提交任务到线程池，去执行service_process_img_buffer函数，this指向当前实例化的对象
    }
    // 将图像消息转opencv格式
    cv::Mat temp_img = cv_bridge::toCvCopy( msg, sensor_msgs::image_encodings::BGR8 )->image.clone();
    //利用 cv_bridge::toCvCopy将 msg 的图像消息转为opencv格式，
    // 调用image成员，会返回一个Mat图像
    //.clone()是深拷贝

    // 图像预处理，然后保存到m_queue_image_with_pose队列，其中msg->header.stamp.toSec()为以秒为单位的时间戳
    process_image( temp_img, msg->header.stamp.toSec());
}

double last_accept_time = 0;
int    buffer_max_frame = 0;   //最大存放帧数
int    total_frame_count = 0;  //处理的图像帧数


//1.检测时间戳，初始化参数
//2.启动 service_pub_rgb_maps 与 service_VIO_update 线程
//3.去畸变与图像处理
void   R3LIVE::process_image( cv::Mat &temp_img, double msg_time )  //一帧数据的预处理！
{
    cv::Mat img_get;
    // 检测图像rows是否正常
    if ( temp_img.rows == 0 )  //rows为图像的行数，为零则不正常
    {
        cout << "Process image error, image rows =0 " << endl;
        return;
    }
    // 检查时间戳是否正常
    if ( msg_time < last_accept_time ) //last_accept_time开始为0
    {
        cout << "Error, image time revert!!" << endl;
        return;
    }
    // 控制图像处理的频率，防止频率过高
    if ( ( msg_time - last_accept_time ) < ( 1.0 / m_control_image_freq ) * 0.9 ) //m_control_image_freq设定的图像处理的频率，时间间隔如果小于要求间隔，则退出，防止太频繁
    {
        return;
    }
    last_accept_time = msg_time; //记录这一帧时间，用于下一帧计算
    // 如果是第一次运行
    if ( m_camera_start_ros_tim < 0 )  //第一次运行：
    {
        m_camera_start_ros_tim = msg_time; //修改第一次运行标记
        m_vio_scale_factor = m_vio_image_width * m_image_downsample_ratio / temp_img.cols; // 320 * 24      图像宽度*1/图像列数，计算一个缩放因子，用于将图像数据缩放到适合 VIO 处理的尺寸
        // load_vio_parameters();  加载vio参数
        //调用 set_initial_camera_parameter 函数，传入多个参数以初始化相机设置 ,,,及系统参数协方差矩阵
        set_initial_camera_parameter( g_lio_state, m_camera_intrinsic.data(), m_camera_dist_coeffs.data(), m_camera_ext_R.data(),
                                      m_camera_ext_t.data(), m_vio_scale_factor );
        cv::eigen2cv( g_cam_K, intrinsic ); //将 Eigen 矩阵 g_cam_K 转换为 OpenCV 矩阵 intrinsic
        cv::eigen2cv( g_cam_dist, dist_coeffs ); //将 Eigen 矩阵 g_cam_dist 转换为 OpenCV 矩阵 dist_coeffs
        // 初始化畸变  使用 OpenCV 的 initUndistortRectifyMap 函数来初始化去畸变和校正映射
        //m_ud_map1 和 m_ud_map2：输出的去畸变和校正映射图像
        initUndistortRectifyMap( intrinsic, dist_coeffs, cv::Mat(), intrinsic, cv::Size( m_vio_image_width / m_vio_scale_factor, m_vio_image_heigh / m_vio_scale_factor ),
                                 CV_16SC2, m_ud_map1, m_ud_map2 );
        // 启动两个线程
        //将 &R3LIVE::service_pub_rgb_maps 作为任务提交给 m_thread_pool_ptr这个线程池子
        m_thread_pool_ptr->commit_task( &R3LIVE::service_pub_rgb_maps, this);//第一次运行的时候也会触发rgb map的发布线程，将RGB点云地图拆分成了子点云（1000）发布
        m_thread_pool_ptr->commit_task( &R3LIVE::service_VIO_update, this); //第一次运行的时候会触发主线程，用于直接调用VIO计算ESIKF的操作
        // 初始化数据记录器
        m_mvs_recorder.init( g_cam_K, m_vio_image_width / m_vio_scale_factor, &m_map_rgb_pts );//传入内参，全局地图
        m_mvs_recorder.set_working_dir( m_map_output_dir );
    }
    //图像下采样：
    if ( m_image_downsample_ratio != 1.0 )
    {
        cv::resize( temp_img, img_get, cv::Size( m_vio_image_width / m_vio_scale_factor, m_vio_image_heigh / m_vio_scale_factor ) );
    }
    else
    {
        img_get = temp_img; // clone ?
    }
    std::shared_ptr< Image_frame > img_pose = std::make_shared< Image_frame >( g_cam_K );//设置该帧图的信息变量img_pose
    // 是否发布原始img
    if ( m_if_pub_raw_img )
    {
        img_pose->m_raw_img = img_get; //将img_get作为原始图像
    }
    // 以img_get为输入，进行去畸变，输出到img_pose->m_img
    cv::remap( img_get, img_pose->m_img, m_ud_map1, m_ud_map2, cv::INTER_LINEAR ); //将输入图像 img_get 的像素重新映射到目标图像 img_pose->m_img 上
    // cv::imshow("sub Img", img_pose->m_img);
    img_pose->m_timestamp = msg_time;  //将时间戳赋给img_pose->m_timestamp
    img_pose->init_cubic_interpolation();  // 转灰度图
    img_pose->image_equalize();              // 直方图均衡化，改善图像效果
    m_camera_data_mutex.lock();
    m_queue_image_with_pose.push_back( img_pose );    // 保存到队列
    m_camera_data_mutex.unlock();
    total_frame_count++;

    // 调整buffer数量，不断加大buffer_max_frame大小
    if ( m_queue_image_with_pose.size() > buffer_max_frame )
    {
        buffer_max_frame = m_queue_image_with_pose.size();
    }

    // cout << "Image queue size = " << m_queue_image_with_pose.size() << endl;
}

//没有用到，，从 ROS 参数服务器加载视觉惯性里程计 (VIO) 所需的相机参数，并将这些参数保存到内部数据结构中
void R3LIVE::load_vio_parameters()
{

    std::vector< double > camera_intrinsic_data, camera_dist_coeffs_data, camera_ext_R_data, camera_ext_t_data;//相机参数
    m_ros_node_handle.getParam( "r3live_vio/image_width", m_vio_image_width );//从ROS服务器获取图像宽高
    m_ros_node_handle.getParam( "r3live_vio/image_height", m_vio_image_heigh );
    m_ros_node_handle.getParam( "r3live_vio/camera_intrinsic", camera_intrinsic_data );  //从ROS参数服务器接收相机内参到camera_intrinsic_data
    m_ros_node_handle.getParam( "r3live_vio/camera_dist_coeffs", camera_dist_coeffs_data ); //相机畸变系数
    m_ros_node_handle.getParam( "r3live_vio/camera_ext_R", camera_ext_R_data ); //相机外参旋转矩阵
    m_ros_node_handle.getParam( "r3live_vio/camera_ext_t", camera_ext_t_data ); //相机外参平移向量

    CV_Assert( ( m_vio_image_width != 0 && m_vio_image_heigh != 0 ) ); //确保图像宽度和高度不为零。
    //如果相机参数出现问题：则进行等待
    if ( ( camera_intrinsic_data.size() != 9 ) || ( camera_dist_coeffs_data.size() != 5 ) || ( camera_ext_R_data.size() != 9 ) ||
         ( camera_ext_t_data.size() != 3 ) )
    {

        cout << ANSI_COLOR_RED_BOLD << "Load VIO parameter fail!!!, please check!!!" << endl;
        printf( "Load camera data size = %d, %d, %d, %d\n", ( int ) camera_intrinsic_data.size(), camera_dist_coeffs_data.size(),
                camera_ext_R_data.size(), camera_ext_t_data.size() );
        cout << ANSI_COLOR_RESET << endl;
        std::this_thread::sleep_for( std::chrono::seconds( 3000000 ) );
    }
    //将参数数据转换为 Eigen 矩阵 :
    m_camera_intrinsic = Eigen::Map< Eigen::Matrix< double, 3, 3, Eigen::RowMajor > >( camera_intrinsic_data.data() ); //将内存数据直接给到Map中
    m_camera_dist_coeffs = Eigen::Map< Eigen::Matrix< double, 5, 1 > >( camera_dist_coeffs_data.data() );
    m_camera_ext_R = Eigen::Map< Eigen::Matrix< double, 3, 3, Eigen::RowMajor > >( camera_ext_R_data.data() );
    m_camera_ext_t = Eigen::Map< Eigen::Matrix< double, 3, 1 > >( camera_ext_t_data.data() );
    //打印输出
    cout << "[Ros_parameter]: r3live_vio/Camera Intrinsic: " << endl;
    cout << m_camera_intrinsic << endl;
    cout << "[Ros_parameter]: r3live_vio/Camera distcoeff: " << m_camera_dist_coeffs.transpose() << endl;
    cout << "[Ros_parameter]: r3live_vio/Camera extrinsic R: " << endl;
    cout << m_camera_ext_R << endl;
    cout << "[Ros_parameter]: r3live_vio/Camera extrinsic T: " << m_camera_ext_t.transpose() << endl;
    std::this_thread::sleep_for( std::chrono::seconds( 1 ) );
}

//根据IMU的位姿，设置一帧图像的位姿和内参
void R3LIVE::set_image_pose( std::shared_ptr< Image_frame > &image_pose, const StatesGroup &state )
{
    mat_3_3 rot_mat = state.rot_end;    //g_lio_state;  系统状态的旋转和平移，即IMU到世界坐标系的旋转
    vec_3   t_vec = state.pos_end;          //g_lio_state;  系统状态的旋转和平移，即IMU到世界坐标系的平移
    // 相机在世界坐标系的位置？？？
    vec_3   pose_t = rot_mat * state.pos_ext_i2c + t_vec;   // IMU到世界坐标系的旋转 * IMU到相机的平移 + IMU到世界坐标系的平移== 相机在世界坐标系的平移
     // 相机在世界坐标系的姿态
    mat_3_3 R_w2c = rot_mat * state.rot_ext_i2c;   //  IMU到世界坐标系的旋转 * IMU到相机的旋转 == 相机在世界坐标系的旋转

    image_pose->set_pose( eigen_q( R_w2c ), pose_t );  //使用eigen_q将R_w2c转为四元数，并设置这一帧图像的位姿
    //提取相机内参给到这一帧：
    image_pose->fx = state.cam_intrinsic( 0 ); 
    image_pose->fy = state.cam_intrinsic( 1 );
    image_pose->cx = state.cam_intrinsic( 2 );
    image_pose->cy = state.cam_intrinsic( 3 );
    //设置这一帧相机内参
    image_pose->m_cam_K << image_pose->fx, 0, image_pose->cx, 0, image_pose->fy, image_pose->cy, 0, 0, 1;
    scope_color( ANSI_COLOR_CYAN_BOLD );
    // cout << "Set Image Pose frm [" << image_pose->m_frame_idx << "], pose: " << eigen_q(rot_mat).coeffs().transpose()
    // << " | " << t_vec.transpose()
    // << " | " << eigen_q(rot_mat).angularDistance( eigen_q::Identity()) *57.3 << endl;
    // image_pose->inverse_pose();
}

//发布相机的位姿信息和相机路径信息
void R3LIVE::publish_camera_odom( std::shared_ptr< Image_frame > &image, double msg_time )
{
    eigen_q            odom_q = image->m_pose_w2c_q;//获取这一帧的位姿信息
    vec_3              odom_t = image->m_pose_w2c_t;
    nav_msgs::Odometry camera_odom;  //相机里程计
    camera_odom.header.frame_id = "world";
    camera_odom.child_frame_id = "/aft_mapped";
    camera_odom.header.stamp = ros::Time::now(); // ros::Time().fromSec(last_timestamp_lidar);设置时间戳
    camera_odom.pose.pose.orientation.x = odom_q.x(); //设置位姿的四元数
    camera_odom.pose.pose.orientation.y = odom_q.y();
    camera_odom.pose.pose.orientation.z = odom_q.z();
    camera_odom.pose.pose.orientation.w = odom_q.w();
    camera_odom.pose.pose.position.x = odom_t( 0 );  //位置
    camera_odom.pose.pose.position.y = odom_t( 1 );
    camera_odom.pose.pose.position.z = odom_t( 2 );
    pub_odom_cam.publish( camera_odom );  //发布相机里程计消息

    //PoseStamped 通常会被用来记录历史位姿，并形成一个路径（camera_path）
    //表示带有时间戳和坐标系信息的位姿消息
    geometry_msgs::PoseStamped msg_pose;
    msg_pose.header.stamp = ros::Time().fromSec( msg_time );
    msg_pose.header.frame_id = "world";
    msg_pose.pose.orientation.x = odom_q.x();
    msg_pose.pose.orientation.y = odom_q.y();
    msg_pose.pose.orientation.z = odom_q.z();
    msg_pose.pose.orientation.w = odom_q.w();
    msg_pose.pose.position.x = odom_t( 0 );
    msg_pose.pose.position.y = odom_t( 1 );
    msg_pose.pose.position.z = odom_t( 2 );
    //camera_path 被用来记录相机的位置和姿态历史，并在每次调用时将新的位姿信息添加到路径中
    camera_path.header.frame_id = "world";
    camera_path.poses.push_back( msg_pose ); //将 PoseStamped 消息添加到 camera_path.poses 中，形成路径的一个新点
    pub_path_cam.publish( camera_path );//发布
}

//没有用到
void R3LIVE::publish_track_pts( Rgbmap_tracker &tracker )
{
    pcl::PointXYZRGB                    temp_point;
    pcl::PointCloud< pcl::PointXYZRGB > pointcloud_for_pub;

    for ( auto it : tracker.m_map_rgb_pts_in_current_frame_pos )
    {
        vec_3      pt = ( ( RGB_pts * ) it.first )->get_pos();
        cv::Scalar color = ( ( RGB_pts * ) it.first )->m_dbg_color;
        temp_point.x = pt( 0 );
        temp_point.y = pt( 1 );
        temp_point.z = pt( 2 );
        temp_point.r = color( 2 );
        temp_point.g = color( 1 );
        temp_point.b = color( 0 );
        pointcloud_for_pub.points.push_back( temp_point );
    }
    sensor_msgs::PointCloud2 ros_pc_msg;
    pcl::toROSMsg( pointcloud_for_pub, ros_pc_msg );
    ros_pc_msg.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
    ros_pc_msg.header.frame_id = "world";       // world; camera_init
    m_pub_visual_tracked_3d_pts.publish( ros_pc_msg );
}

// ANCHOR - VIO preintegration    在 VIO 系统中处理 IMU 数据，通过预积分计算更新状态
bool R3LIVE::vio_preintegration( StatesGroup &state_in, StatesGroup &state_out, double current_frame_time )
{
    state_out = state_in;  //输出状态先设为输入状态
    // 检查当前帧的时间是否小于等于上一次更新的时间，图像时间必须是新的，没有处理过
    if ( current_frame_time <= state_in.last_update_time ) 
    {
        // cout << ANSI_COLOR_RED_BOLD << "Error current_frame_time <= state_in.last_update_time | " <<
        // current_frame_time - state_in.last_update_time << ANSI_COLOR_RESET << endl;
        return false;
    }
    mtx_buffer.lock();
    std::deque< sensor_msgs::Imu::ConstPtr > vio_imu_queue;   //放置IMU数据的队列
    // 遍历imu_buffer_vio容器中的元素，将其放入到vio_imu_queue
    for ( auto it = imu_buffer_vio.begin(); it != imu_buffer_vio.end(); it++ )
    {
        vio_imu_queue.push_back( *it ); //将其添加到vio_imu_queue
        // 如果时间戳大于当前帧的时间，则跳出循环，意思就是：到最新的这一帧图像就停止，即只处理以前的数据
        if ( ( *it )->header.stamp.toSec() > current_frame_time )
        {
            break;
        }
    }
    // 当imu_buffer_vio容器不为空时执行循环，即有预积分数据时：只保留前0.2秒的数据，，太老的数据认为没有关系
    while ( !imu_buffer_vio.empty() )
    {
        // 获取imu_buffer_vio容器中第一个元素的时间戳
        double imu_time = imu_buffer_vio.front()->header.stamp.toSec();
        // imu和current_frame_time的时间差
        if ( imu_time < current_frame_time - 0.2 ) //只保留前0.2秒的数据，太老的数据认为没有关系
        {
            // 将该元素从容器中移除
            imu_buffer_vio.pop_front();
        }
        else
        {
            break;
        }
    }
    // cout << "Current VIO_imu buffer size = " << imu_buffer_vio.size() << endl;
    //进行IMU预积分
    state_out = m_imu_process->imu_preintegration( state_out, vio_imu_queue, current_frame_time - vio_imu_queue.back()->header.stamp.toSec() );
    eigen_q q_diff( state_out.rot_end.transpose() * state_in.rot_end ); //计算积分前后两个状态的旋转差异
    // cout << "Pos diff = " << (state_out.pos_end - state_in.pos_end).transpose() << endl;
    // cout << "Euler diff = " << q_diff.angularDistance(eigen_q::Identity()) * 57.3 << endl;
    mtx_buffer.unlock();
    // 更新时间信息
    state_out.last_update_time = current_frame_time; //将当前图像时间赋给输出的系统状态
    return true;
}

// ANCHOR - huber_loss  根据损失求huber核函数   计算 Huber 损失函数的缩放因子
//Huber 损失函数是一种用于处理回归问题的损失函数，它结合了均方误差和绝对误差的优点，以提高对异常值的鲁棒性
//根据给定的重投影误差（reprojection_error）和离群点阈值（outlier_threshold）来计算这个缩放因子
double get_huber_loss_scale( double reprojection_error, double outlier_threshold = 1.0 )
{
    // http://ceres-solver.org/nnls_modeling.html#lossfunction
    double scale = 1.0;
    if ( reprojection_error / outlier_threshold < 1.0 )  //表示重投影误差相对较小，低于阈值。此时，返回的 scale 设置为 1.0。
    {
        scale = 1.0;
    }
    else
    {
        scale = ( 2 * sqrt( reprojection_error ) / sqrt( outlier_threshold ) - 1.0 ) / reprojection_error;  //这个公式将 Huber 损失函数的计算与阈值进行标定，调整对大误差的处理方式
    }
    return scale;
}

// ANCHOR - VIO_esikf    进行帧到帧VIO   根据重投影误差 =》更新优化系统状态
const int minimum_iteration_pts = 10;  //最小的能够优化的点数
bool      R3LIVE::vio_esikf( StatesGroup &state_in, Rgbmap_tracker &op_track ) //输入：相机状态、光流跟踪
{
    Common_tools::Timer tim;  //测量时间
    tim.tic();   //开始计时
    scope_color( ANSI_COLOR_BLUE_BOLD );       //设置控制台输出的文本颜色为蓝色粗体
    StatesGroup state_iter = state_in;   //新建一个状态变量
    if ( !m_if_estimate_intrinsic ) // When disable the online intrinsic calibration.  当不能进行相机内参标定时：
    {
        state_iter.cam_intrinsic << g_cam_K( 0, 0 ), g_cam_K( 1, 1 ), g_cam_K( 0, 2 ), g_cam_K( 1, 2 ); //直接将相机自身参数作为内参
    }

    if ( !m_if_estimate_i2c_extrinsic )  //当不使用标定的IMU到相机之间的外参标定时：
    {
        state_iter.pos_ext_i2c = m_inital_pos_ext_i2c;    //将原始数据设置为外参数
        state_iter.rot_ext_i2c = m_inital_rot_ext_i2c;       //设将原始数据设置为外参数
    }

    Eigen::Matrix< double, -1, -1 >                       H_mat;  //不固定大小的矩阵
    Eigen::Matrix< double, -1, 1 >                        meas_vec;   //定义了一个动态大小的列向量 meas_vec
    Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES > G, H_T_H, I_STATE;  //定义了三个大小为 DIM_OF_STATES x DIM_OF_STATES 的矩阵
    Eigen::Matrix< double, DIM_OF_STATES, 1 >             solution;  //存储结果
    Eigen::Matrix< double, -1, -1 >                       K, KH;   //不固定大小的矩阵
    Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES > K_1;   //定义了一个大小为 DIM_OF_STATES x DIM_OF_STATES 的矩阵 K_1

    Eigen::SparseMatrix< double > H_mat_spa, H_T_H_spa, K_spa, KH_spa, vec_spa, I_STATE_spa;  //定义几个稀疏矩阵
    I_STATE.setIdentity();  //单位矩阵
    I_STATE_spa = I_STATE.sparseView();  //上面的单位阵变成稀疏矩阵
    double fx, fy, cx, cy, time_td;  //用于存储相机的内参和其他数据

    int                   total_pt_size = op_track.m_map_rgb_pts_in_current_frame_pos.size();//获取当前帧跟踪到的地图点中 RGB 点的数量
    //创建两个 std::vector，分别用于存储上一次和当前的重投影误差
    std::vector< double > last_reprojection_error_vec( total_pt_size ), current_reprojection_error_vec( total_pt_size );
    //少于10个点不能优化：
    if ( total_pt_size < minimum_iteration_pts )
    {
        state_in = state_iter;
        return false;
    }
    H_mat.resize( total_pt_size * 2, DIM_OF_STATES );//行数：点数*2    列数：29
    meas_vec.resize( total_pt_size * 2, 1 );   //点数* 2 行和 1 列
    double last_repro_err = 3e8;  //设置上次重投影误差初值
    int    avail_pt_count = 0;  //有效跟踪点的数量
    double last_avr_repro_err = 0;  //用于跟踪上一次的平均重投影误差

    double acc_reprojection_error = 0;  //累积重投影误差
    double img_res_scale = 1.0;  //图像分辨率缩放因子
    //优化两次：
    for ( int iter_count = 0; iter_count < esikf_iter_times; iter_count++ )
    {

        // cout << "========== Iter " << iter_count << " =========" << endl;
        mat_3_3 R_imu = state_iter.rot_end;   //即IMU到世界坐标系的旋转
        vec_3   t_imu = state_iter.pos_end;    //即IMU到世界坐标系的平移
        vec_3   t_c2w = R_imu * state_iter.pos_ext_i2c + t_imu;   //计算相机在世界坐标系中的位置
        mat_3_3 R_c2w = R_imu * state_iter.rot_ext_i2c; // 计算相机到世界坐标系的旋转矩阵
        //提取相机内参
        fx = state_iter.cam_intrinsic( 0 );
        fy = state_iter.cam_intrinsic( 1 );
        cx = state_iter.cam_intrinsic( 2 );
        cy = state_iter.cam_intrinsic( 3 );
        time_td = state_iter.td_ext_i2c_delta;//时间差

        vec_3   t_w2c = -R_c2w.transpose() * t_c2w;  //反过来，世界到相机
        mat_3_3 R_w2c = R_c2w.transpose();//反过来，世界到相机
        int     pt_idx = -1;  //点的索引
        acc_reprojection_error = 0;  //累积重投影误差
        vec_3               pt_3d_w, pt_3d_cam;   //世界坐标系和相机坐标系中的 3D 点
        vec_2               pt_img_measure, pt_img_proj, pt_img_vel;  //图像坐标系中的点，分别表示测量点、投影点和速度点
        eigen_mat_d< 2, 3 > mat_pre;  // 2x3 和 3x3 的矩阵
        eigen_mat_d< 3, 3 > mat_A, mat_B, mat_C, mat_D, pt_hat;
        H_mat.setZero(); //全设置为0
        solution.setZero();
        meas_vec.setZero();
        avail_pt_count = 0;  //记录有效点的数量
        //遍历当前这一帧所有追踪到的点
        for ( auto it = op_track.m_map_rgb_pts_in_last_frame_pos.begin(); it != op_track.m_map_rgb_pts_in_last_frame_pos.end(); it++ )
        {
            pt_3d_w = ( ( RGB_pts * ) it->first )->get_pos();  //获取该点在三维世界坐标系中的位置 
            pt_img_vel = ( ( RGB_pts * ) it->first )->m_img_vel;  //获取该点的速度
            pt_img_measure = vec_2( it->second.x, it->second.y );  // 该跟踪地图点   通过光流法 在这一帧图像中的位置，
            pt_3d_cam = R_w2c * pt_3d_w + t_w2c;   //将地图点从世界坐标系变换到当前帧相机坐标系
            // 1. 根据相机内参将相机坐标系转换到像素坐标系，，，，
            // 2. 计算相机-imu时间偏差导致的增量
            //最终得到 ：地图点在当前帧图像的投影像素坐标 pt_img_proj
            pt_img_proj = vec_2( fx * pt_3d_cam( 0 ) / pt_3d_cam( 2 ) + cx, fy * pt_3d_cam( 1 ) / pt_3d_cam( 2 ) + cy ) + time_td * pt_img_vel;
            //计算重投影误差，这里计算的是误差的大小，不用于迭代求解，只是用来确定huber核函数
            double repro_err = ( pt_img_proj - pt_img_measure ).norm();  //地图点映射到这一帧的xy点  -  地图点映射到上一帧经过光流到这一帧，.norm()就是求两点距离
            double huber_loss_scale = get_huber_loss_scale( repro_err );  //根据误差大小来确定损失核函数
            pt_idx++;  //点索引+1，开始第一个点是0
            acc_reprojection_error += repro_err;  //累加所有点的重投影误差
            // if (iter_count == 0 || ((repro_err - last_reprojection_error_vec[pt_idx]) < 1.5))
            //将重投影误差记录在last_reprojection_error_vec中
            if ( iter_count == 0 || ( ( repro_err - last_avr_repro_err * 5.0 ) < 0 ) )   //第一次迭代优化 或者 重投影误差比平均重投影误差五倍还小
            {
                last_reprojection_error_vec[ pt_idx ] = repro_err;
            }
            else
            {
                last_reprojection_error_vec[ pt_idx ] = repro_err;
            }
            avail_pt_count++;  //点的数量加1
            // Appendix E of r2live_Supplementary_material.
            // https://github.com/hku-mars/r2live/blob/master/supply/r2live_Supplementary_material.pdf
            // 像素对相机点求雅可比，
            //该矩阵用于将相机坐标系中的 3D 点映射到图像坐标系中
            mat_pre << fx / pt_3d_cam( 2 ), 0, -fx * pt_3d_cam( 0 ) / pt_3d_cam( 2 ), 0, fy / pt_3d_cam( 2 ), -fy * pt_3d_cam( 1 ) / pt_3d_cam( 2 );
            //将世界坐标系下的三维点转换到IMU中
            pt_hat = Sophus::SO3d::hat( ( R_imu.transpose() * ( pt_3d_w - t_imu ) ) );  //IMU系下地图点 p(IMU)^ = R(IMU <-- W) * ( p(W) - p(IMU) )
            //计算雅可比矩阵:
            mat_A = state_iter.rot_ext_i2c.transpose() * pt_hat;  //三维点在IMU-》相机
            mat_B = -state_iter.rot_ext_i2c.transpose() * ( R_imu.transpose() );   //相机-》世界
            mat_C = Sophus::SO3d::hat( pt_3d_cam );   //世界 -》相机
            mat_D = -state_iter.rot_ext_i2c.transpose();  //相机 -》IMU
            meas_vec.block( pt_idx * 2, 0, 2, 1 ) = ( pt_img_proj - pt_img_measure ) * huber_loss_scale / img_res_scale;  //观测向量填充

            //对各个参数进行求雅可比：外参、速度、内参
            H_mat.block( pt_idx * 2, 0, 2, 3 ) = mat_pre * mat_A * huber_loss_scale;  // H, 1-2行，前3列, 对R(IMU)雅可比
            H_mat.block( pt_idx * 2, 3, 2, 3 ) = mat_pre * mat_B * huber_loss_scale;  // H, 1-2行，4-6列，对P(IMU)雅可比
            if ( DIM_OF_STATES > 24 )
            {
                // Estimate time td.   对时间差的导数
                H_mat.block( pt_idx * 2, 24, 2, 1 ) = pt_img_vel * huber_loss_scale;  // H，1-2行， 25-26列，对像素速度雅可比
                // H_mat(pt_idx * 2, 24) = pt_img_vel(0) * huber_loss_scale;
                // H_mat(pt_idx * 2 + 1, 24) = pt_img_vel(1) * huber_loss_scale;
            }
            if ( m_if_estimate_i2c_extrinsic )
            {
                H_mat.block( pt_idx * 2, 18, 2, 3 ) = mat_pre * mat_C * huber_loss_scale;  //H ,1-2行，19-21列，对外参R(IMU<--C)雅可比
                H_mat.block( pt_idx * 2, 21, 2, 3 ) = mat_pre * mat_D * huber_loss_scale;  //H ,1-2行，22-24列，对外参t(IMU<--C)雅可比
            }

            if ( m_if_estimate_intrinsic )
            {
                H_mat( pt_idx * 2, 25 ) = pt_3d_cam( 0 ) / pt_3d_cam( 2 ) * huber_loss_scale;  //H,1行，26列，对内参fx雅可比
                H_mat( pt_idx * 2 + 1, 26 ) = pt_3d_cam( 1 ) / pt_3d_cam( 2 ) * huber_loss_scale;  //H,2行，27列，对内参fy雅可比
                H_mat( pt_idx * 2, 27 ) = 1 * huber_loss_scale;  //H,1行，28列，对内参cx雅可比
                H_mat( pt_idx * 2 + 1, 28 ) = 1 * huber_loss_scale;  //H,2行，29列，对内参cy雅可比
            }
        }
        H_mat = H_mat / img_res_scale;  //不变
        acc_reprojection_error /= total_pt_size;   //求每个点平均的重投影误差

        last_avr_repro_err = acc_reprojection_error;  //记录本次的平均重投影误差
        if ( avail_pt_count < minimum_iteration_pts )  //如果点数小于最小点数
        {
            break;
        }

        H_mat_spa = H_mat.sparseView();  //转为稀疏矩阵   节省内存和计算时间
        Eigen::SparseMatrix< double > Hsub_T_temp_mat = H_mat_spa.transpose();  //转置：Hsub_T_temp_mat
        vec_spa = ( state_iter - state_in ).sparseView();  //系统状态变化
        H_T_H_spa = Hsub_T_temp_mat * H_mat_spa; //雅可比 乘 雅可比转置
        // Notice that we have combine some matrix using () in order to boost the matrix multiplication.
        Eigen::SparseMatrix< double > temp_inv_mat =
            ( ( H_T_H_spa.toDense() + eigen_mat< -1, -1 >( state_in.cov * m_cam_measurement_weight ).inverse() ).inverse() ).sparseView();
        KH_spa = temp_inv_mat * ( Hsub_T_temp_mat * H_mat_spa );//卡尔曼增益矩阵
        //用于计算更新后的状态估计。具体来说，它结合了预测值和新的测量数据
        solution = ( temp_inv_mat * ( Hsub_T_temp_mat * ( ( -1 * meas_vec.sparseView() ) ) ) - ( I_STATE_spa - KH_spa ) * vec_spa ).toDense();  //用于计算更新后的状态估计

        state_iter = state_iter + solution;//更新状态
        //本次平均重投影误差  接近于 上一次平均重投影误差   =》 结束优化
        if ( fabs( acc_reprojection_error - last_repro_err ) < 0.01 )
        {
            break;
        }
        last_repro_err = acc_reprojection_error;  //记录本次平均重投影误差
    }
    //几次优化完了：
    if ( avail_pt_count >= minimum_iteration_pts )
    {
        state_iter.cov = ( ( I_STATE_spa - KH_spa ) * state_iter.cov.sparseView() ).toDense();  //更新协方差矩阵
    }

    state_iter.td_ext_i2c += state_iter.td_ext_i2c_delta;  //????????
    state_iter.td_ext_i2c_delta = 0;  //IMU到相机的外参
    state_in = state_iter;  //更新输入的状态 =》优化后的状态
    return true;
}

//第2步优化：帧到地图VIO   在上一步更新的状态和方差基础上，利用跟踪点的光度误差再进行ESIKF状态更新
//关键求：求观测量对误差状态向量的各个偏导数（H矩阵）
bool R3LIVE::vio_photometric( StatesGroup &state_in, Rgbmap_tracker &op_track, std::shared_ptr< Image_frame > &image )
{
    Common_tools::Timer tim;  //计时器
    tim.tic(); //开始计时
    StatesGroup state_iter = state_in;  //复制系统状态
    if (!m_if_estimate_intrinsic)     // When disable the online intrinsic calibration.
    {
        state_iter.cam_intrinsic << g_cam_K(0, 0), g_cam_K(1, 1), g_cam_K(0, 2), g_cam_K(1, 2);  //相机内参 =》state_iter.cam_intrinsic
    }
    if (!m_if_estimate_i2c_extrinsic) // When disable the online extrinsic calibration.
    {
        state_iter.pos_ext_i2c = m_inital_pos_ext_i2c;  //IMU到相机外参的平移
        state_iter.rot_ext_i2c = m_inital_rot_ext_i2c;  //IMU到相机外参的旋转
    }
    Eigen::Matrix< double, -1, -1 >                       H_mat, R_mat_inv;
    Eigen::Matrix< double, -1, 1 >                        meas_vec;
    Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES > G, H_T_H, I_STATE;
    Eigen::Matrix< double, DIM_OF_STATES, 1 >             solution;
    Eigen::Matrix< double, -1, -1 >                       K, KH;
    Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES > K_1;
    Eigen::SparseMatrix< double >                         H_mat_spa, H_T_H_spa, R_mat_inv_spa, K_spa, KH_spa, vec_spa, I_STATE_spa;
    I_STATE.setIdentity();
    I_STATE_spa = I_STATE.sparseView();
    double fx, fy, cx, cy, time_td;

    int                   total_pt_size = op_track.m_map_rgb_pts_in_current_frame_pos.size();  //获取跟踪点的数量
    std::vector< double > last_reprojection_error_vec( total_pt_size ), current_reprojection_error_vec( total_pt_size ); //存储上一次和当前重投影误差的向量
    if ( total_pt_size < minimum_iteration_pts )  //点数不足则退出优化
    {
        state_in = state_iter;
        return false;
    }

    int err_size = 3;
    H_mat.resize( total_pt_size * err_size, DIM_OF_STATES );//设置雅可比矩阵大小
    meas_vec.resize( total_pt_size * err_size, 1 );  //观测向量
    R_mat_inv.resize( total_pt_size * err_size, total_pt_size * err_size );  //表示协方差矩阵的逆

    double last_repro_err = 3e8;//初始化重投影误差
    int    avail_pt_count = 0;
    double last_avr_repro_err = 0;  //上一次平均重投影误差
    int    if_esikf = 1; //是否启用esikf

    double acc_photometric_error = 0; //用于累计光度误差
#if DEBUG_PHOTOMETRIC
    printf("==== [Image frame %d] ====\r\n", g_camera_frame_idx);
#endif
    for ( int iter_count = 0; iter_count < 2; iter_count++ )//迭代两次
    {
        mat_3_3 R_imu = state_iter.rot_end;//获取IMU（惯性测量单元）的旋转矩阵和位置向量。
        vec_3   t_imu = state_iter.pos_end;
        vec_3   t_c2w = R_imu * state_iter.pos_ext_i2c + t_imu; //计算相机到世界坐标系的平移向量
        mat_3_3 R_c2w = R_imu * state_iter.rot_ext_i2c; // world to camera frame计算相机坐标系到世界坐标系的旋转矩阵 
        //相机内参:
        fx = state_iter.cam_intrinsic( 0 );
        fy = state_iter.cam_intrinsic( 1 );
        cx = state_iter.cam_intrinsic( 2 );
        cy = state_iter.cam_intrinsic( 3 );
        time_td = state_iter.td_ext_i2c_delta; //相机的时间偏移量。

        vec_3   t_w2c = -R_c2w.transpose() * t_c2w; //计算世界坐标系到相机坐标系的平移向量 t_w2c
        mat_3_3 R_w2c = R_c2w.transpose();//计算世界坐标系到相机坐标系的旋转矩阵 
        int     pt_idx = -1;  //点的索引
        acc_photometric_error = 0; //初始化累计光度误差
        vec_3               pt_3d_w, pt_3d_cam;
        vec_2               pt_img_measure, pt_img_proj, pt_img_vel;
        eigen_mat_d< 2, 3 > mat_pre;
        eigen_mat_d< 3, 2 > mat_photometric;
        eigen_mat_d< 3, 3 > mat_d_pho_d_img;
        eigen_mat_d< 3, 3 > mat_A, mat_B, mat_C, mat_D, pt_hat;
        R_mat_inv.setZero();
        H_mat.setZero();
        solution.setZero();
        meas_vec.setZero();
        avail_pt_count = 0;
        int iter_layer = 0;
        tim.tic( "Build_cost" );//开始计时："Build_cost"
        //存储RGB点与其对应图像位置的映射
        ///遍历所有地图点，计算雅可比矩阵和观测向量
        for ( auto it = op_track.m_map_rgb_pts_in_last_frame_pos.begin(); it != op_track.m_map_rgb_pts_in_last_frame_pos.end(); it++ )
        {
            if ( ( ( RGB_pts * ) it->first )->m_N_rgb < 3 )//没有rgb跳过
            {
                continue;
            }
            pt_idx++;//点索引加1
            pt_3d_w = ( ( RGB_pts * ) it->first )->get_pos();  //获取三维点
            // 对齐相机曝光时间和imu的时间差
            pt_img_vel = ( ( RGB_pts * ) it->first )->m_img_vel;//获取图像点的速度。
            pt_img_measure = vec_2( it->second.x, it->second.y );  //获取图像二维点
            // 将地图点转到相机坐标系下后投影到相机平面
            pt_3d_cam = R_w2c * pt_3d_w + t_w2c; //将地图点从世界坐标系转换到相机坐标系
            pt_img_proj = vec_2( fx * pt_3d_cam( 0 ) / pt_3d_cam( 2 ) + cx, fy * pt_3d_cam( 1 ) / pt_3d_cam( 2 ) + cy ) + time_td * pt_img_vel;//将点投影到像素坐标

            // 每个图像帧的跟踪点在地图中存有一个索引，因此可获取上次融合后的地图点rgb值和协方差
            vec_3   pt_rgb = ( ( RGB_pts * ) it->first )->get_rgb();  //获取三维点的RGB值
            mat_3_3 pt_rgb_info = mat_3_3::Zero();
            mat_3_3 pt_rgb_cov = ( ( RGB_pts * ) it->first )->get_rgb_cov();  //地图点rgb对角协方差矩阵
            // 作为观测量，协方差矩阵在更新公式中出现在逆的位置
            for ( int i = 0; i < 3; i++ )
            {
                pt_rgb_info( i, i ) = 1.0 / pt_rgb_cov( i, i ) ; //计算协方差矩阵的逆矩阵的对角线元素。
                R_mat_inv( pt_idx * err_size + i, pt_idx * err_size + i ) = pt_rgb_info( i, i ); //设置为逆协方差矩阵的元素
                // R_mat_inv( pt_idx * err_size + i, pt_idx * err_size + i ) =  1.0;
            }
            vec_3  obs_rgb_dx, obs_rgb_dy;
            // 获取这一帧中跟踪点的亚像素级rgb值
            //投影点从影像插值获取rgb，计算x、y方向上的rgb差值
            vec_3  obs_rgb = image->get_rgb( pt_img_proj( 0 ), pt_img_proj( 1 ), 0, &obs_rgb_dx, &obs_rgb_dy );
            vec_3  photometric_err_vec = ( obs_rgb - pt_rgb );  //rgb误差: 图像对应点rgb值 - 三维地图点rgb值
            // huber loss可以显著降低离群点对优化的影响
            double huber_loss_scale = get_huber_loss_scale( photometric_err_vec.norm() );//计算Huber损失的缩放因子
            photometric_err_vec *= huber_loss_scale; //将Huber损失缩放因子应用于RGB误差向量
            double photometric_err = photometric_err_vec.transpose() * pt_rgb_info * photometric_err_vec;//计算影像误差:          e^T * info_rgb(map) * e

            acc_photometric_error += photometric_err; //将计算出的影像误差累加到总的光度误差

            last_reprojection_error_vec[ pt_idx ] = photometric_err;//记录每个点的光度误差

            mat_photometric.setZero();
            mat_photometric.col( 0 ) = obs_rgb_dx; //将x和y方向的RGB差值分别赋给mat_photometric矩阵的列。
            mat_photometric.col( 1 ) = obs_rgb_dy;

            avail_pt_count++; //增加可用点计数
            // 内参投影对地图点的雅可比矩阵
            // 像素坐标对相机坐标求导    ,像素坐标对相机坐标的雅可比矩阵 
            mat_pre << fx / pt_3d_cam( 2 ), 0, -fx * pt_3d_cam( 0 ) / pt_3d_cam( 2 ), 0, fy / pt_3d_cam( 2 ), -fy * pt_3d_cam( 1 ) / pt_3d_cam( 2 );
            mat_d_pho_d_img = mat_photometric * mat_pre; //误差对图像坐标的雅可比矩阵
            //jacobian与vio_esikf相同    计算IMU坐标系下地图点的雅可比矩阵
            pt_hat = Sophus::SO3d::hat( ( R_imu.transpose() * ( pt_3d_w - t_imu ) ) );    //IMU系下地图点 p(IMU)^ = R(IMU <-- W) * ( p(W) - p(IMU) )

            /// 观测量对误差的雅可比矩阵的推导参照r2live附录E
            /// 对姿态R的雅可比
            mat_A = state_iter.rot_ext_i2c.transpose() * pt_hat;//3 * 3， R（C <-- IMU) * p(imu)^
            /// 对位移t的雅可比
            mat_B = -state_iter.rot_ext_i2c.transpose() * ( R_imu.transpose() );// - R(C <--IMU) * R(IMU <-- W)
            /// 对外参旋转量的雅可比
            mat_C = Sophus::SO3d::hat( pt_3d_cam );// p(C)^
            /// 对外参位移量的雅可比
            mat_D = -state_iter.rot_ext_i2c.transpose();// - R(C <-- IMU)
            meas_vec.block( pt_idx * 3, 0, 3, 1 ) = photometric_err_vec ; //观测向量填充
            /// 链式求导的结果
            H_mat.block( pt_idx * 3, 0, 3, 3 ) = mat_d_pho_d_img * mat_A * huber_loss_scale; //分别填充了对相机旋转和位移的雅可比矩阵。
            H_mat.block( pt_idx * 3, 3, 3, 3 ) = mat_d_pho_d_img * mat_B * huber_loss_scale;
            if ( 1 )
            {
                if ( m_if_estimate_i2c_extrinsic )
                {
                    H_mat.block( pt_idx * 3, 18, 3, 3 ) = mat_d_pho_d_img * mat_C * huber_loss_scale;  //外参旋转和位移的雅可比
                    H_mat.block( pt_idx * 3, 21, 3, 3 ) = mat_d_pho_d_img * mat_D * huber_loss_scale;
                }
            }
        }
        R_mat_inv_spa = R_mat_inv.sparseView(); //稀疏
       
        last_avr_repro_err = acc_photometric_error; //保存了上一帧的平均重投影误差
        if ( avail_pt_count < minimum_iteration_pts )//如果当前可用的点数(avail_pt_count) 小于最小迭代点数(minimum_iteration_pts)，则退出循环
        {
            break;
        }
        // Esikf
        tim.tic( "Iter" ); //("Iter") 开始计时
        if ( if_esikf )  //使用esikf
        {

            /// sparseView使矩阵无需申请过大内存
            /// 经典的计算增益K的过程
            H_mat_spa = H_mat.sparseView();//稀疏
            Eigen::SparseMatrix< double > Hsub_T_temp_mat = H_mat_spa.transpose(); //转置矩阵
            vec_spa = ( state_iter - state_in ).sparseView();//系统状态变化
            H_T_H_spa = Hsub_T_temp_mat * R_mat_inv_spa * H_mat_spa; //计算了稀疏雅可比矩阵转置与逆矩阵的乘积
            /// 这一帧与地图重叠部分越多视觉测量权重越大
            Eigen::SparseMatrix< double > temp_inv_mat =
                ( H_T_H_spa.toDense() + ( state_in.cov * m_cam_measurement_weight ).inverse() ).inverse().sparseView();
            // ( H_T_H_spa.toDense() + ( state_in.cov ).inverse() ).inverse().sparseView();
            Eigen::SparseMatrix< double > Ht_R_inv = ( Hsub_T_temp_mat * R_mat_inv_spa );
            KH_spa = temp_inv_mat * Ht_R_inv * H_mat_spa; //计算了增益矩阵 K
            solution = ( temp_inv_mat * ( Ht_R_inv * ( ( -1 * meas_vec.sparseView() ) ) ) - ( I_STATE_spa - KH_spa ) * vec_spa ).toDense(); //计算了状态的更新量
        }
        state_iter = state_iter + solution; //更新状态
#if DEBUG_PHOTOMETRIC
        cout << "Average photometric error: " <<  acc_photometric_error / total_pt_size << endl;
        cout << "Solved solution: "<< solution.transpose() << endl;
#else
        if ( ( acc_photometric_error / total_pt_size ) < 10 ) // By experience.如果每点的光度误差低于 10，则退出循环
        {
            break;
        }
#endif
        if ( fabs( acc_photometric_error - last_repro_err ) < 0.01 )//这个条件判断当前误差与上次误差的变化是否很小
        {
            break;
        }
        last_repro_err = acc_photometric_error;//记录本次迭代的误差
    }
    if ( if_esikf && avail_pt_count >= minimum_iteration_pts )//如果使用 ESIKF（扩展平滑信息滤波器）且可用的点数大于或等于最小迭代点数，更新状态协方差矩阵
    {
        state_iter.cov = ( ( I_STATE_spa - KH_spa ) * state_iter.cov.sparseView() ).toDense();
    }
    state_iter.td_ext_i2c += state_iter.td_ext_i2c_delta;
    state_iter.td_ext_i2c_delta = 0;
    state_in = state_iter;//更新状态
    return true;
}

//不断发布新的全局RGB地图
void R3LIVE::service_pub_rgb_maps()
{
    int last_publish_map_idx = -3e8;  //设置一个初始值为负数的大整数，表示最后一次发布的地图索引
    int sleep_time_aft_pub = 10; //设置每次发布后要等待的时间
    int number_of_pts_per_topic = 1000;  //每个话题要发布的点云点数的初始值
    if ( number_of_pts_per_topic < 0 )
    {
        return;
    }
    while ( 1 )  //不断发布新的全局RGB地图：
    {
        ros::spinOnce();  //在循环体内，处理完所有的回调函数，在循环中响应ROS的消息的回调函数
        std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );  //暂停该线程10毫秒
        pcl::PointCloud< pcl::PointXYZRGB > pc_rgb;  //定义XZY颜色的点云数据
        sensor_msgs::PointCloud2            ros_pc_msg;  //定义ROS的点云数据
        int pts_size = m_map_rgb_pts.m_rgb_pts_vec.size();  //从RGB点云地图中获取    vector保存的地图rgb点（指针形式）的数量
        pc_rgb.resize( number_of_pts_per_topic ); //调整大小
        // for (int i = pts_size - 1; i > 0; i--)
        int pub_idx_size = 0;  //发布点的数量
        int cur_topic_idx = 0;  //当前话题的序列
        if ( last_publish_map_idx == m_map_rgb_pts.m_last_updated_frame_idx )  //如果这两个值相等，说明当前帧的数据已经处理过，不需要再次处理
        {
            continue;
        }
        last_publish_map_idx = m_map_rgb_pts.m_last_updated_frame_idx;  //赋值给他最新一帧的序列号

        for ( int i = 0; i < pts_size; i++ )  //挨个点处理：
        {
            if ( m_map_rgb_pts.m_rgb_pts_vec[ i ]->m_N_rgb < 1 ) //如果该点没有RGB则跳出循环
            {
                continue;
            }
            //将m_map_rgb_pts中的信息传给pc_rgb
            pc_rgb.points[ pub_idx_size ].x = m_map_rgb_pts.m_rgb_pts_vec[ i ]->m_pos[ 0 ];
            pc_rgb.points[ pub_idx_size ].y = m_map_rgb_pts.m_rgb_pts_vec[ i ]->m_pos[ 1 ];
            pc_rgb.points[ pub_idx_size ].z = m_map_rgb_pts.m_rgb_pts_vec[ i ]->m_pos[ 2 ];
            pc_rgb.points[ pub_idx_size ].r = m_map_rgb_pts.m_rgb_pts_vec[ i ]->m_rgb[ 2 ];
            pc_rgb.points[ pub_idx_size ].g = m_map_rgb_pts.m_rgb_pts_vec[ i ]->m_rgb[ 1 ];
            pc_rgb.points[ pub_idx_size ].b = m_map_rgb_pts.m_rgb_pts_vec[ i ]->m_rgb[ 0 ];
            // pc_rgb.points[i].intensity = m_map_rgb_pts.m_rgb_pts_vec[i]->m_obs_dis;
            pub_idx_size++;  //计数pc_rgb
            if ( pub_idx_size == number_of_pts_per_topic )  //得到的点的数量达到了1000个的时候：
            {
                pub_idx_size = 0;  //计数器1000个点后归零
                pcl::toROSMsg( pc_rgb, ros_pc_msg );  //将其复制给ros_pc_msg，将这1000个点发出去
                ros_pc_msg.header.frame_id = "world";   //  设置frame_id
                ros_pc_msg.header.stamp = ros::Time::now(); //设置时间
                if ( m_pub_rgb_render_pointcloud_ptr_vec[ cur_topic_idx ] == nullptr )  //如果这一个指针是空的，即没有保存过
                {
                    m_pub_rgb_render_pointcloud_ptr_vec[ cur_topic_idx ] =     //创建一个指向ROS发布器的智能指针，并实例化，将其存储在m_pub_rgb_render_pointcloud_ptr_vec
                        std::make_shared< ros::Publisher >(m_ros_node_handle.advertise< sensor_msgs::PointCloud2 >(
                            std::string( "/RGB_map_" ).append( std::to_string( cur_topic_idx ) ), 100 ) );
                    //其中advertise会创建发布< sensor_msgs::PointCloud2 >类型的发布器，，，std::string( "/RGB_map_" ).append( std::to_string( cur_topic_idx ) )是发布的话题名称
                }
                m_pub_rgb_render_pointcloud_ptr_vec[ cur_topic_idx ]->publish( ros_pc_msg );  //使用刚创建的发布器发布点云数据
                std::this_thread::sleep_for( std::chrono::microseconds( sleep_time_aft_pub ) );  //该线程等待10毫秒
                ros::spinOnce();  //
                cur_topic_idx++;
            }
        }
        //将最后不到1000个点发布出去，和上面相同：
        pc_rgb.resize( pub_idx_size ); 
        pcl::toROSMsg( pc_rgb, ros_pc_msg );
        ros_pc_msg.header.frame_id = "world";       
        ros_pc_msg.header.stamp = ros::Time::now(); 
        if ( m_pub_rgb_render_pointcloud_ptr_vec[ cur_topic_idx ] == nullptr )
        {
            m_pub_rgb_render_pointcloud_ptr_vec[ cur_topic_idx ] =
                std::make_shared< ros::Publisher >( m_ros_node_handle.advertise< sensor_msgs::PointCloud2 >(
                    std::string( "/RGB_map_" ).append( std::to_string( cur_topic_idx ) ), 100 ) );
        }
        std::this_thread::sleep_for( std::chrono::microseconds( sleep_time_aft_pub ) );
        ros::spinOnce();
        m_pub_rgb_render_pointcloud_ptr_vec[ cur_topic_idx ]->publish( ros_pc_msg );
        cur_topic_idx++;
        if ( cur_topic_idx >= 45 ) // Maximum pointcloud topics = 45.
        {
            number_of_pts_per_topic *= 1.5;  //如果点数太多，则提高每次发布的点数
            sleep_time_aft_pub *= 1.5;   //提高延时
        }
    }
}

//没有用到
void R3LIVE::publish_render_pts( ros::Publisher &pts_pub, Global_map &m_map_rgb_pts )
{
    pcl::PointCloud< pcl::PointXYZRGB > pc_rgb;
    sensor_msgs::PointCloud2            ros_pc_msg;
    pc_rgb.reserve( 1e7 );
    m_map_rgb_pts.m_mutex_m_box_recent_hitted->lock();
    std::unordered_set< std::shared_ptr< RGB_Voxel > > boxes_recent_hitted = m_map_rgb_pts.m_voxels_recent_visited;
    m_map_rgb_pts.m_mutex_m_box_recent_hitted->unlock();

    for ( Voxel_set_iterator it = boxes_recent_hitted.begin(); it != boxes_recent_hitted.end(); it++ )
    {
        for ( int pt_idx = 0; pt_idx < ( *it )->m_pts_in_grid.size(); pt_idx++ )
        {
            pcl::PointXYZRGB           pt;
            std::shared_ptr< RGB_pts > rgb_pt = ( *it )->m_pts_in_grid[ pt_idx ];
            pt.x = rgb_pt->m_pos[ 0 ];
            pt.y = rgb_pt->m_pos[ 1 ];
            pt.z = rgb_pt->m_pos[ 2 ];
            pt.r = rgb_pt->m_rgb[ 2 ];
            pt.g = rgb_pt->m_rgb[ 1 ];
            pt.b = rgb_pt->m_rgb[ 0 ];
            if ( rgb_pt->m_N_rgb > m_pub_pt_minimum_views )
            {
                pc_rgb.points.push_back( pt );
            }
        }
    }
    pcl::toROSMsg( pc_rgb, ros_pc_msg );
    ros_pc_msg.header.frame_id = "world";       // world; camera_init
    ros_pc_msg.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
    pts_pub.publish( ros_pc_msg );
}

//处理键盘输入保存并显示点云
char R3LIVE::cv_keyboard_callback()
{
    char c = cv_wait_key( 1 ); //获取输入
    // return c;
    if ( c == 's' || c == 'S' )
    {
        scope_color( ANSI_COLOR_GREEN_BOLD );
        cout << "I capture the keyboard input!!!" << endl;
        m_mvs_recorder.export_to_mvs( m_map_rgb_pts );  //将当前的 RGB 点云数据导出到 MVS 文件中
        // m_map_rgb_pts.save_and_display_pointcloud( m_map_output_dir, std::string("/rgb_pt"), std::max(m_pub_pt_minimum_views, 5) );
        m_map_rgb_pts.save_and_display_pointcloud( m_map_output_dir, std::string("/rgb_pt"), m_pub_pt_minimum_views  );  //保存并显示点云数据
    }
    return c;
}

// ANCHOR -  service_VIO_update ESIKF优化计算
void R3LIVE::service_VIO_update()
{
    // Init cv windows for debug
    //初始化光流跟踪
    op_track.set_intrinsic( g_cam_K, g_cam_dist * 0, cv::Size( m_vio_image_width / m_vio_scale_factor, m_vio_image_heigh / m_vio_scale_factor ) ); 
    op_track.m_maximum_vio_tracked_pts = m_maximum_vio_tracked_pts;//最大光流追踪点数量
    m_map_rgb_pts.m_minimum_depth_for_projection = m_tracker_minimum_depth; //从全局RGB地图,用于投影时的最小深度和最大深度
    m_map_rgb_pts.m_maximum_depth_for_projection = m_tracker_maximum_depth;//从全局RGB地图,用于投影时的最小深度和最大深度
    cv::imshow( "Control panel", generate_control_panel_img().clone() );  //显示名为 "Control panel" 的窗口，并用 generate_control_panel_img() 函数生成的图像填充窗口。clone() 方法用于确保图像的副本被显示。
    Common_tools::Timer tim;  //创建一个计时器对象
    cv::Mat             img_get;
    while ( ros::ok() )   //一直循环：
    {
        cv_keyboard_callback();//根据输入S保存现实点云
        // 检查是否收到第一帧激光雷达扫描，没收到，则循环等待
        while ( g_camera_lidar_queue.m_if_have_lidar_data == 0 )
        {
            ros::spinOnce();
            std::this_thread::sleep_for( std::chrono::milliseconds( THREAD_SLEEP_TIM ) );
            std::this_thread::yield();
            continue;
        }
        // 检查是否收到预处理后的图像,没有则循环等待
        if ( m_queue_image_with_pose.size() == 0 )
        {
            ros::spinOnce();
            std::this_thread::sleep_for( std::chrono::milliseconds( THREAD_SLEEP_TIM ) );
            std::this_thread::yield();
            continue;
        }
        m_camera_data_mutex.lock();
        // 如果m_queue_image_with_pose队列内的缓存数据大于buffer,则将最旧的图像帧用于跟踪,然后pop掉
        while ( m_queue_image_with_pose.size() > m_maximum_image_buffer )
        {
            cout << ANSI_COLOR_BLUE_BOLD << "=== Pop image! current queue size = " << m_queue_image_with_pose.size() << " ===" << ANSI_COLOR_RESET
                 << endl;  //输出m_queue_image_with_pose的大小
            op_track.track_img( m_queue_image_with_pose.front(), -20 );//跟踪特征点,最前面的一张图
            m_queue_image_with_pose.pop_front();//弹出已经处理的最前面的一张图
        }

        std::shared_ptr< Image_frame > img_pose = m_queue_image_with_pose.front();  //拿到处理过的第一帧图像
        double                             message_time = img_pose->m_timestamp;  //获取这一帧的时间戳
        m_queue_image_with_pose.pop_front();  //弹出这一帧
        m_camera_data_mutex.unlock();
        g_camera_lidar_queue.m_last_visual_time = img_pose->m_timestamp + g_lio_state.td_ext_i2c;  //得出最新观测时间，当前的时间+时间补偿

        img_pose->set_frame_idx( g_camera_frame_idx );  //开始为0，为每一个图像编号
        tim.tic( "Frame" );  //调用tim中的tic来起动计时

        if ( g_camera_frame_idx == 0 )  //处理第一帧：
        {
            std::vector< cv::Point2f >                pts_2d_vec;       // 选中的地图点反投影到图像上的坐标 ，存储二维点
            std::vector< std::shared_ptr< RGB_pts > > rgb_pts_vec;      // 选中的地图点
            // while ( ( m_map_rgb_pts.is_busy() ) || ( ( m_map_rgb_pts.m_rgb_pts_vec.size() <= 100 ) ) )
            
            // 等待地图中的颜色点大于100个，此时LIO模块会调用Global_map::append_points_to_global_map，不断向全局地图添加点
            while ( ( ( m_map_rgb_pts.m_rgb_pts_vec.size() <= 100 ) ) )  
            {
                ros::spinOnce();
                std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
            }
            // 根据IMU的位姿，设置一帧图像的位姿和内参
            // 对于第一帧，这里假设是静止的运动状态，直接用系统状态设置位姿，认为还没有动
            set_image_pose( img_pose, g_lio_state ); // For first frame pose, we suppose that the motion is static.
            // 调用全局地图模块，根据相机状态，选择一些点，将三维点投影到图像上，
            m_map_rgb_pts.selection_points_for_projection( img_pose, &rgb_pts_vec, &pts_2d_vec, m_track_windows_size / m_vio_scale_factor );
            // 初始化跟踪模块
            op_track.init( img_pose, rgb_pts_vec, pts_2d_vec );  //初始化光流，，输入：一帧图像，跟踪点，投影点
            g_camera_frame_idx++;   //图片索引+1
            continue;
        }

        //接着通过对比相机和lidar队列头的时间戳，如果lidar的时间戳更早则等待lio线程把更早的激光处理完。
        //接着进行预积分部分，通过IMU结合图像的形式完成位置的预积分
        g_camera_frame_idx++; //图片索引+1
        tim.tic( "Wait" );   //启动Wait计时器
        // if_camera_can_process(): 当雷达有数据，并且lidar buffer中最旧的雷达数据时间 > 当前正在处理的图像时间戳，则返回true，即可以处理图像了
        while ( g_camera_lidar_queue.if_camera_can_process() == false )  //不可处理图像
        {
            // 否则，在这里循环等待处理雷达数据
            ros::spinOnce();
            std::this_thread::sleep_for( std::chrono::milliseconds( THREAD_SLEEP_TIM ) );
            std::this_thread::yield();
            cv_keyboard_callback();  //键盘输入保存点云
        }
        g_cost_time_logger.record( tim, "Wait" );  //记录时间，即等待时间
        m_mutex_lio_process.lock();  //保护线程
        tim.tic( "Frame" );    //Frame 计时开始
        tim.tic( "Track_img" );    //Track_img 计时开始
        StatesGroup state_out;  //积分后得到的系统状态
        //最终，m_cam_measurement_weight 将被设置为介于 0.001 和 0.01 之间的一个值
        m_cam_measurement_weight = std::max( 0.001, std::min( 5.0 / m_number_of_new_visited_voxel, 0.01 ) ); 
        //从LIO预积分到当前帧时刻， VIO 系统中处理 IMU 数据，通过预积分计算更新状态
        if ( vio_preintegration( g_lio_state, state_out, img_pose->m_timestamp + g_lio_state.td_ext_i2c ) == false ) //不能预积分时：锁打开，跳过这一次循环：
        {
            m_mutex_lio_process.unlock();
            continue;
        }
        //将从上一LIO预积分的结果，设定为当前帧对应的位姿
        set_image_pose( img_pose, state_out );

        // 调用 op_track.track_img( img_pose, -20 ) 跟踪特征点，将张一帧图像中的点跟踪到这一帧图像！
        //这里是跟踪图像上的点，将上一帧的点通过光流法 =》这一帧
        //将追踪到的点放到=》m_map_rgb_pts_in_last_frame_pos
        op_track.track_img( img_pose, -20 )


        //这里的track_img注意与 LK_optical_flow_kernel::track_image区分
        // 光流跟踪，同时去除outliers

        g_cost_time_logger.record( tim, "Track_img" ); //Track_img 计时结束
        // cout << "Track_img cost " << tim.toc( "Track_img" ) << endl;
        tim.tic( "Ransac" );  //Ransac计时开始
        set_image_pose( img_pose, state_out );//最后将更新的img_pose作为输出，完成相机pose和内参的校准  ？？？？？

        // ANCHOR -  remove point using PnP.
        if ( op_track.remove_outlier_using_ransac_pnp( img_pose ) == 0 )  //使用 RANSAC 和 PnP 算法去除异常的跟踪点
        {
            cout << ANSI_COLOR_RED_BOLD << "****** Remove_outlier_using_ransac_pnp error*****" << ANSI_COLOR_RESET << endl;
        }
        g_cost_time_logger.record( tim, "Ransac" );  //Ransac计时结束
        tim.tic( "Vio_f2f" );   // Vio_f2f 计时开始
        bool res_esikf = true, res_photometric = true;
        wait_render_thread_finish();
        //输入光流追踪和预测后的系统状态，进行ESIKF
        res_esikf = vio_esikf( state_out, op_track );   //根据重投影误差 = 》更新优化系统状态
        g_cost_time_logger.record( tim, "Vio_f2f" );   //记录 "Vio_f2f"计时结束
        tim.tic( "Vio_f2m" );  //"Vio_f2m"计时开始
        //更新帧到地图   ESIKF
        res_photometric = vio_photometric( state_out, op_track, img_pose );
        g_cost_time_logger.record( tim, "Vio_f2m" );  //记录 "Vio_f2m"计时结束 
        g_lio_state = state_out;//更新系统状态
        //生成并显示一个格式化的仪表板，提供实时的系统状态信息
        print_dash_board();
        //用优化后的位姿来设定当前帧的位姿
        set_image_pose( img_pose, state_out );

        if ( 1 )
        {
            tim.tic( "Render" );//"Render"开始计时        渲染
            // m_map_rgb_pts.render_pts_in_voxels(img_pose, m_last_added_rgb_pts_vec);
            if ( 1 ) // Using multiple threads for rendering
            {
                m_map_rgb_pts.m_if_get_all_pts_in_boxes_using_mp = 0;//使用单线程处理数据
                // m_map_rgb_pts.render_pts_in_voxels_mp(img_pose, &m_map_rgb_pts.m_rgb_pts_in_recent_visited_voxels,
                // img_pose->m_timestamp);
                
                //线程5：主要作用是将点云地图的active点投影到图像上，用对应的像素对地图点rgb的均值和方差用贝叶斯迭代进行更新。
                //调用函数模版std::make_shared，来管理内存,   创建一个指针m_render_thread
                //函数模版的参数为std::shared_future< void >
                //函数模版的 形式参数表 为 
                //m_thread_pool_ptr->commit_task(render_pts_in_voxels_mp, img_pose, & m_map_rgb_pts.m_voxels_recent_visited, img_pose->m_timestamp )
                m_render_thread = std::make_shared< std::shared_future< void > >( m_thread_pool_ptr->commit_task(
                    render_pts_in_voxels_mp, img_pose, &m_map_rgb_pts.m_voxels_recent_visited, img_pose->m_timestamp ) );
            }       //并行渲染体素数据
            else
            {
                m_map_rgb_pts.m_if_get_all_pts_in_boxes_using_mp = 0;
                // m_map_rgb_pts.render_pts_in_voxels( img_pose, m_map_rgb_pts.m_rgb_pts_in_recent_visited_voxels,
                // img_pose->m_timestamp );
            }
            m_map_rgb_pts.m_last_updated_frame_idx = img_pose->m_frame_idx;//将这一帧序列给到m_last_updated_frame_idx
            g_cost_time_logger.record( tim, "Render" );  //结束渲染计时

            tim.tic( "Mvs_record" );//开始计时："Mvs_record"
            if ( m_if_record_mvs )  //?
            {
                // m_mvs_recorder.insert_image_and_pts( img_pose, m_map_rgb_pts.m_voxels_recent_visited );
                m_mvs_recorder.insert_image_and_pts( img_pose, m_map_rgb_pts.m_pts_last_hitted );//数据记录器记录这一帧图像的信息和全局地图
            }
            g_cost_time_logger.record( tim, "Mvs_record" );  
        }
        // ANCHOR - render point cloud
        //打印输出系统各种状态：
        dump_lio_state_to_log( m_lio_state_fp );
        m_mutex_lio_process.unlock();//线程解锁
        // cout << "Solve image pose cost " << tim.toc("Solve_pose") << endl;
        //根据这帧图像的数据，更新图像投影参数
        m_map_rgb_pts.update_pose_for_projection( img_pose, -0.4 );
        op_track.update_and_append_track_pts( img_pose, m_map_rgb_pts, m_track_windows_size / m_vio_scale_factor, 1000000 );//更新这一帧的跟踪点
        g_cost_time_logger.record( tim, "Frame" );  //记录处理这一帧图像所消耗的时间
        double frame_cost = tim.toc( "Frame" ); //获取标记为 "Frame" 的时间段所消耗的时间
        g_image_vec.push_back( img_pose );//保存图像
        frame_cost_time_vec.push_back( frame_cost );//保存时间
        if ( g_image_vec.size() > 10 )  //只保留近10帧的数据：
        {
            g_image_vec.pop_front();
            frame_cost_time_vec.pop_front();
        }
        tim.tic( "Pub" );   //计时开始
        double display_cost_time = std::accumulate( frame_cost_time_vec.begin(), frame_cost_time_vec.end(), 0.0 ) / frame_cost_time_vec.size(); //计算所有帧处理时间的平均值
        g_vio_frame_cost_time = display_cost_time;
        // publish_render_pts( m_pub_render_rgb_pts, m_map_rgb_pts );
        //发布相机位姿和相机路径信息
        publish_camera_odom( img_pose, message_time );
        // publish_track_img( op_track.m_debug_track_img, display_cost_time );
        //在图像上绘制处理时间并发布出来
        publish_track_img( img_pose->m_raw_img, display_cost_time );

        if ( m_if_pub_raw_img ) //发布原始图像
        {
            publish_raw_img( img_pose->m_raw_img );
        }
        //刷新日志
        if ( g_camera_lidar_queue.m_if_dump_log )
        {
            g_cost_time_logger.flush();
        }
        // cout << "Publish cost time " << tim.toc("Pub") << endl;
    }
}
