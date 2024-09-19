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
std::shared_ptr< Common_tools::ThreadPool > m_thread_pool_ptr;  //����ָ�룬ָ���̳߳�
double                                      g_vio_frame_cost_time = 0;
double                                      g_lio_frame_cost_time = 0;
int                                         g_flag_if_first_rec_img = 1;
#define DEBUG_PHOTOMETRIC 0
#define USING_CERES 0
void dump_lio_state_to_log( FILE *fp )   //��ӡ״̬
{
    if ( fp != nullptr && g_camera_lidar_queue.m_if_dump_log )//���������־��¼��
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

//���ַ��������һ�������Ŀո�ʹ���ַ����ĳ��ȴﵽָ�����ַ�λ��,Ϊ�˶���
double g_last_stamped_mem_mb = 0;
std::string append_space_to_bits( std::string & in_str, int bits )
{
    while( in_str.length() < bits )
    {
        in_str.append(" ");
    }
    return in_str;
}

//���ڴ�ӡһ������̨�Ǳ�壬��ʾ�й�ϵͳ״̬�ĸ�����Ϣ��ϵͳʱ�䡢LiDAR֡�����֡����ͼ�������ڴ�ʹ�ã�
void R3LIVE::print_dash_board()
{
#if DEBUG_PHOTOMETRIC
    return;
#endif
    int mem_used_mb = ( int ) ( Common_tools::get_RSS_Mb() ); //��ȡ��ǰ�ڴ�ʹ���������MBΪ��λ����
    // clang-format off
    if( (mem_used_mb - g_last_stamped_mem_mb < 1024 ) && g_last_stamped_mem_mb != 0 )//����ڴ�ʹ�ñ仯����
    {
        cout  << ANSI_DELETE_CURRENT_LINE << ANSI_DELETE_LAST_LINE ;
    }
    else
    {
        cout << "\r\n" << endl;
        cout << ANSI_COLOR_WHITE_BOLD << "======================= R3LIVE Dashboard ======================" << ANSI_COLOR_RESET << endl;
        g_last_stamped_mem_mb = mem_used_mb ;//���´�ӡ�Ǳ��� ������¼�ڴ�ʹ����
    }
    std::string out_str_line_1, out_str_line_2;
    out_str_line_1 = std::string(        "| System-time | LiDAR-frame | Camera-frame |  Pts in maps | Memory used (Mb) |") ;
    //                                    1             16            30             45             60     
    // clang-format on
    out_str_line_2.reserve( 1e3 ); //reserve(1e3) Ϊ out_str_line_2 �ַ���������1000���ַ����ڴ�ռ�
    out_str_line_2.append( "|   " ).append( Common_tools::get_current_time_str() ); //ϵͳʱ��
    append_space_to_bits( out_str_line_2, 14 ); //14���ַ���λ�ã��������
    out_str_line_2.append( "|    " ).append( std::to_string( g_LiDAR_frame_index ) ); //LiDAR֡
    append_space_to_bits( out_str_line_2, 28 );
    out_str_line_2.append( "|    " ).append( std::to_string( g_camera_frame_idx ) ); //���֡
    append_space_to_bits( out_str_line_2, 43 );
    out_str_line_2.append( "| " ).append( std::to_string( m_map_rgb_pts.m_rgb_pts_vec.size() ) ); //��ͼ����
    append_space_to_bits( out_str_line_2, 58 );
    out_str_line_2.append( "|    " ).append( std::to_string( mem_used_mb ) ); //�ڴ�ʹ��
    //ʹ��ANSI��ɫ���������Ǳ������ֵ���ɫ:
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
    //����Ǳ��
    cout << out_str_line_1 << endl;
    cout << out_str_line_2 << ANSI_COLOR_RESET << "          ";
    ANSI_SCREEN_FLUSH;
}

//������״̬�����ĳ�ʼЭ�������
void R3LIVE::set_initial_state_cov( StatesGroup &state )
{
    // Set cov
    scope_color( ANSI_COLOR_RED_BOLD );
    state.cov = state.cov.setIdentity() * INIT_COV; //�� state.cov �����ʼ��Ϊ��λ���󣬲�����һ������ INIT_COV 0.0001
    // state.cov.block(18, 18, 6 , 6 ) = state.cov.block(18, 18, 6 , 6 ) .setIdentity() * 0.1;
    // state.cov.block(24, 24, 5 , 5 ) = state.cov.block(24, 24, 5 , 5 ).setIdentity() * 0.001;
    state.cov.block( 0, 0, 3, 3 ) = mat_3_3::Identity() * 1e-5;   // R    �����˾����ǰ 3x3 �飨����ת�����Э���Ϊ��λ������� 1e-5����ʾ��ת����ĳ�ʼ��ȷ���Էǳ�С
    state.cov.block( 3, 3, 3, 3 ) = mat_3_3::Identity() * 1e-5;   // T     �����˴ӵ����е����п�ʼ�� 3X3 �ĵ�λ����   -��ƽ��
    state.cov.block( 6, 6, 3, 3 ) = mat_3_3::Identity() * 1e-5;   // vel    �ٶ�
    state.cov.block( 9, 9, 3, 3 ) = mat_3_3::Identity() * 1e-3;   // bias_g   ������ƫ��
    state.cov.block( 12, 12, 3, 3 ) = mat_3_3::Identity() * 1e-1; // bias_a  ���ٶȼ�ƫ��
    state.cov.block( 15, 15, 3, 3 ) = mat_3_3::Identity() * 1e-5; // Gravity  ����
    state.cov( 24, 24 ) = 0.00001;   //���þ���� 24 �е� 24 �е�Ԫ��Ϊ 0.00001
    state.cov.block( 18, 18, 6, 6 ) = state.cov.block( 18, 18, 6, 6 ).setIdentity() *  1e-3; // Extrinsic between camera and IMU.���ôӵ� 18 �е� 18 ����� 6x6 ��Ϊ��λ������� 1e-3����ʾ����� IMU ֮����ⲿ�궨�ĳ�ʼ��ȷ���ԡ�
    state.cov.block( 25, 25, 4, 4 ) = state.cov.block( 25, 25, 4, 4 ).setIdentity() *  1e-3; // Camera intrinsic. ���ôӵ� 25 �е� 25 ����� 4x4 ��Ϊ��λ������� 1e-3����ʾ����ڲεĳ�ʼ��ȷ���ԡ�
}

//�����������service_VIO_update()��������һ��ͼ�񣬸����û���������
cv::Mat R3LIVE::generate_control_panel_img()
{
    int     line_y = 40;
    int     padding_x = 10;
    int     padding_y = line_y * 0.7;
    cv::Mat res_image = cv::Mat( line_y * 3 + 1 * padding_y, 960, CV_8UC3, cv::Scalar::all( 0 ) );//����Mat��С��RGB
    char    temp_char[ 128 ];
    sprintf( temp_char, "Click this windows to enable the keyboard controls." ); //���Ǿ仰д��temp_char
    cv::putText( res_image, std::string( temp_char ), cv::Point( padding_x, line_y * 0 + padding_y ), cv::FONT_HERSHEY_COMPLEX, 1,
                 cv::Scalar( 0, 255, 255 ), 2, 8, 0 ); //���ı����Ƶ�ͼ�ϣ�������˴��������ü��̿��ơ�
    sprintf( temp_char, "Press 'S' or 's' key to save current map" );
    cv::putText( res_image, std::string( temp_char ), cv::Point( padding_x, line_y * 1 + padding_y ), cv::FONT_HERSHEY_COMPLEX, 1,
                 cv::Scalar( 255, 255, 255 ), 2, 8, 0 );//ͬ�ϣ����ı����Ƶ�ͼ��  ������ 'S' �� 's' ���Ա��浱ǰ��ͼ��
    sprintf( temp_char, "Press 'space' key to pause the mapping process" );
    cv::putText( res_image, std::string( temp_char ), cv::Point( padding_x, line_y * 2 + padding_y ), cv::FONT_HERSHEY_COMPLEX, 1,
                 cv::Scalar( 255, 255, 255 ), 2, 8, 0 );//ͬ�ϣ����ı����Ƶ�ͼ��     ������ 'space' ������ͣӳ����̡�
    return res_image;
}

//��ʼ������Ĳ���
void R3LIVE::set_initial_camera_parameter( StatesGroup &state, double *intrinsic_data, double *camera_dist_data, double *imu_camera_ext_R,
                                           double *imu_camera_ext_t, double cam_k_scale )
{
    scope_color( ANSI_COLOR_YELLOW_BOLD );
    // g_cam_K << 863.4241 / cam_k_scale, 0, 625.6808 / cam_k_scale,
    //     0, 863.4171 / cam_k_scale, 518.3392 / cam_k_scale,
    //     0, 0, 1;

    //�Ӳ����л�ȡ�ڲκ����������������ڲξ��� g_cam_K
    g_cam_K << intrinsic_data[ 0 ] / cam_k_scale, intrinsic_data[ 1 ], intrinsic_data[ 2 ] / cam_k_scale, intrinsic_data[ 3 ],
        intrinsic_data[ 4 ] / cam_k_scale, intrinsic_data[ 5 ] / cam_k_scale, intrinsic_data[ 6 ], intrinsic_data[ 7 ], intrinsic_data[ 8 ];
    //����ϵ��  g_cam_dist
    // camera_dist_data: ����һ��ָ�� double ���ݵ�ָ�룬������ָ��һ���������� 5 �� double Ԫ�ص��ڴ�����
    //igen::Matrix<double, 5, 1>: ����һ�� 5 �� 1 �еľ������ͣ�����������
    //Eigen::Map<Eigen::Matrix<double, 5, 1>>: ���� Eigen �ṩ��ӳ���࣬���� camera_dist_data ����һ�� 5x1 �ľ�������������ݸ��ơ�
    g_cam_dist = Eigen::Map< Eigen::Matrix< double, 5, 1 > >( camera_dist_data );
    //����IMU�����������任��Σ���ת��ƽ�ƣ�
    state.rot_ext_i2c = Eigen::Map< Eigen::Matrix< double, 3, 3, Eigen::RowMajor > >( imu_camera_ext_R );
    state.pos_ext_i2c = Eigen::Map< Eigen::Matrix< double, 3, 1 > >( imu_camera_ext_t );
    // state.pos_ext_i2c.setZero();

    // Lidar to camera parameters.���������� m_mutex_lio_process����ȷ���̰߳�ȫ�ر��� IMU ���������ת�����λ�������ĳ�ʼֵ
    m_mutex_lio_process.lock();
    m_inital_rot_ext_i2c = state.rot_ext_i2c;  //��������
    m_inital_pos_ext_i2c = state.pos_ext_i2c;
    //������ڴ���������ݴ���ϵͳ״̬����
    state.cam_intrinsic( 0 ) = g_cam_K( 0, 0 );
    state.cam_intrinsic( 1 ) = g_cam_K( 1, 1 );
    state.cam_intrinsic( 2 ) = g_cam_K( 0, 2 );
    state.cam_intrinsic( 3 ) = g_cam_K( 1, 2 );
    //���ú��� set_initial_state_cov ������״̬�ĳ�ʼЭ�����ͨ�������Ż��㷨�еĲ�ȷ���Թ���
    set_initial_state_cov( state );
    m_mutex_lio_process.unlock();
}

//��ͼ���ϻ��ƴ���ʱ�䲢��������
void R3LIVE::publish_track_img( cv::Mat &img, double frame_cost_time = -1 )
{
    cv_bridge::CvImage out_msg; //�� OpenCV ͼ��(cv::Mat) ת��Ϊ ROS ͼ����Ϣ������
    out_msg.header.stamp = ros::Time::now();               // Same timestamp and tf frame as input image  ����Ϊ��ǰʱ��
    out_msg.encoding = sensor_msgs::image_encodings::BGR8; // Or whatever��ʾͼ��ı����ʽ�� BGR8
    cv::Mat pub_image = img.clone(); //��¡ͼ��
    //���㴦��ʱ�䲢��ͼ���ϻ����ı�
    if ( frame_cost_time > 0 )
    {
        char fps_char[ 100 ];
        sprintf( fps_char, "Per-frame cost time: %.2f ms", frame_cost_time );//��ʾ����ʱ��
        // sprintf(fps_char, "%.2f ms", frame_cost_time);

        if ( pub_image.cols <= 640 )//���ڿ��С�ڻ���� 640 ���ص�ͼ���ı������СΪ 1��
        {
            cv::putText( pub_image, std::string( fps_char ), cv::Point( 30, 30 ), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar( 255, 255, 255 ), 2, 8,
                         0 ); // 640 * 480
        }
        else if ( pub_image.cols > 640 )//���ڿ�ȴ��� 640 ���ص�ͼ���ı������СΪ 2
        {
            cv::putText( pub_image, std::string( fps_char ), cv::Point( 30, 50 ), cv::FONT_HERSHEY_COMPLEX, 2, cv::Scalar( 255, 255, 255 ), 2, 8,
                         0 ); // 1280 * 1080
        }
    }
    out_msg.image = pub_image; // Your cv::Mat
    pub_track_img.publish( out_msg ); //����ͼ����Ϣ�� ROS �е� pub_track_img ����
}

//����ԭʼͼ��
void R3LIVE::publish_raw_img( cv::Mat &img )
{
    cv_bridge::CvImage out_msg; //OpenCV ͼ��(cv::Mat) ת��Ϊ ROS ͼ����Ϣ
    out_msg.header.stamp = ros::Time::now();               // Same timestamp and tf frame as input image
    out_msg.encoding = sensor_msgs::image_encodings::BGR8; // Or whatever
    out_msg.image = img;                                   // Your cv::Mat
    pub_raw_img.publish( out_msg );
}

int        sub_image_typed = 0; // 0: TBD 1: sub_raw, 2: sub_comp
std::mutex mutex_image_callback;   //������һ��ͼ�����ݵĻ�����

std::deque< sensor_msgs::CompressedImageConstPtr > g_received_compressed_img_msg; //�洢ѹ��ͼ��ָ��Ķ���
std::deque< sensor_msgs::ImageConstPtr >           g_received_img_msg;
std::shared_ptr< std::thread >                     g_thr_process_image;

//ͨ�� m_thread_pool_ptr->commit_task �̳߳صķ�ʽ�����ͼ��buffer��ѹ���봦�������������ĩβ������� process_image ����
void R3LIVE::service_process_img_buffer()
{
    while ( 1 ) //��ͣѭ������������ͼ������
    {
        // To avoid uncompress so much image buffer, reducing the use of memory.
        //���m_queue_image_with_pose�����ڵ����� > 4����ʾ��Щ���ݻ�û��������ʱ����Ԥ�����̣߳���һЩ���ݣ�
        if ( m_queue_image_with_pose.size() > 4 )
        {
            while ( m_queue_image_with_pose.size() > 4 )   //֪������������С���ĸ�������ѭ��
            {
                ros::spinOnce();  //ros::spinOnce() ��Ҫ��������ѭ���ж��ڴ��� ROS �Ļص�����ȷ����Ľڵ��ܹ���Ӧ��Ϣ���¼����������ڴ�����Щ�ص�ʱ��������
                std::this_thread::sleep_for( std::chrono::milliseconds( 2 ) );  //ʹ��ǰ�߳���ͣ 2 ���룬�������Լ��� CPU ռ��
                std::this_thread::yield();  //�����������ϵͳ����ǰ�̵߳�ִ��Ȩ���������߳�
            }
        }
        cv::Mat image_get;
        double  img_rec_time;

        // sub_image_typed == 2����ʾ���յ���ѹ��ͼ���ʽ
        if ( sub_image_typed == 2 )  //��������ѹ��ͼ��
        {
            // ���������û�����ݣ���ͣ��ǰ�߳�1s���Լ���CPU��ʹ��
            while ( g_received_compressed_img_msg.size() == 0 )
            {
                ros::spinOnce();
                std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
                std::this_thread::yield();
            }
            //�������ˣ� �Ӷ��е�ǰ�˻�ȡһ��ѹ��ͼ����Ϣmsg
            sensor_msgs::CompressedImageConstPtr msg = g_received_compressed_img_msg.front();
            try   //�������У�����������catch
            {
                // ��ѹ��ͼ����Ϣת��Ϊcv::Mat���͵�ͼ������
                cv_bridge::CvImagePtr cv_ptr_compressed = cv_bridge::toCvCopy( msg, sensor_msgs::image_encodings::BGR8 );
                // �洢��ȡ��ʱ���ͼ��
                img_rec_time = msg->header.stamp.toSec(); //��ȡ��ǰͼ���ʱ���-��img_rec_time
                image_get = cv_ptr_compressed->image; //��ȡͼ��
                // �ͷ��ڴ�
                cv_ptr_compressed->image.release();
            }
            catch ( cv_bridge::Exception &e )   //��׽ cv_bridge::toCvCopy ���������׳��� cv_bridge::Exception �쳣��
            {
                printf( "Could not convert from '%s' to 'bgr8' !!! ", msg->format.c_str() );
            }
            mutex_image_callback.lock(); //���ϣ�������unlock()֮��Ĵ��룬������Դ�޷�����������
            g_received_compressed_img_msg.pop_front();  //������ǰ���һ��ѹ��ͼ������
            mutex_image_callback.unlock();  //����
        }
        else  //�������ѹ�����ݣ�
        {
            // ���������û�����ݣ���ͣ��ǰ�߳�1s���Լ���CPU��ʹ��
            while ( g_received_img_msg.size() == 0 ) 
            {
                ros::spinOnce();
                std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
                std::this_thread::yield();
            }
            // ��ǰ�����������
            sensor_msgs::ImageConstPtr msg = g_received_img_msg.front(); //��ȡ��ǰ���һ��ͼ��
            image_get = cv_bridge::toCvCopy( msg, sensor_msgs::image_encodings::BGR8 )->image.clone(); //����ͼ��תΪopencv��ʽ
            img_rec_time = msg->header.stamp.toSec(); //��ȡʱ���
            mutex_image_callback.lock();
            g_received_img_msg.pop_front(); //ͬ����Ҫ���ϲ�������ǰ���ͼ��
            mutex_image_callback.unlock();
        }
        process_image( image_get, img_rec_time ); //����һ֡���ݴ�����
    }
}

//ͨ��һ���̻߳�ȡͼ����Ϣ
void R3LIVE::image_comp_callback( const sensor_msgs::CompressedImageConstPtr &msg )
{
    std::unique_lock< std::mutex > lock2( mutex_image_callback );//ʹ�û�������mutex��������������Դ��ȷ���ڴ���ͼ��ʱ������������������ʹ���ڶ��̻߳����У�����ص������ǰ�ȫ�ġ�
    //ʹ��unique_lock������ mutex��    lock��unique_lock��ʵ������
    //mutex_image_callback��һ��std::mutex���͵ı�����������������Ҫ������һ������   

    if ( sub_image_typed == 1 )  //
    {
        return; // Avoid subscribe the same image twice.
    }
    sub_image_typed = 2;
    g_received_compressed_img_msg.push_back( msg );
    // ����ǵ�һ���յ�ͼƬ��������һ���̣߳���������image_comp_callback�ص��н��յ�ѹ��ͼƬ���ڲ���ʵ��ѭ������process_image()����
    if ( g_flag_if_first_rec_img )
    {
        g_flag_if_first_rec_img = 0;
        // ͨ���̳߳�k��������service_process_img_buffer����������ͼ��
         // �ڲ���ʵ��ѭ������process_image()����
        m_thread_pool_ptr->commit_task( &R3LIVE::service_process_img_buffer, this ); //ͼ�����߳�����������
    }
    return;
}

// ANCHOR - image_callback   ���պʹ�������ROS�Ļ����ͼ����Ϣ
void R3LIVE::image_callback( const sensor_msgs::ImageConstPtr &msg )
{
    std::unique_lock< std::mutex > lock( mutex_image_callback );//ʹ�û�������mutex��������������Դ��ȷ���ڴ���ͼ��ʱ������������������ʹ���ڶ��̻߳����У�����ص������ǰ�ȫ�ġ�
    //ʹ��unique_lock������ mutex    lock��unique_lock��ʵ������
    //mutex_image_callback��һ��std::mutex���͵ı�����������������Ҫ��������Դ�����ε�����   

    if ( sub_image_typed == 2 )  //����Ƿ����
    {
        return; // Avoid subscribe the same image twice.
    }
    sub_image_typed = 1; //����Ϊ1�����������Ͳ�������

    if ( g_flag_if_first_rec_img )  //����Ƿ��ǵ�һ�ν���ͼ��
    {
        g_flag_if_first_rec_img = 0;   //���Ϊ�ѽ���
        m_thread_pool_ptr->commit_task( &R3LIVE::service_process_img_buffer, this );//ͼ�����߳������������ύ�����̳߳أ�ȥִ��service_process_img_buffer������thisָ��ǰʵ�����Ķ���
    }
    // ��ͼ����Ϣתopencv��ʽ
    cv::Mat temp_img = cv_bridge::toCvCopy( msg, sensor_msgs::image_encodings::BGR8 )->image.clone();
    //���� cv_bridge::toCvCopy�� msg ��ͼ����ϢתΪopencv��ʽ��
    // ����image��Ա���᷵��һ��Matͼ��
    //.clone()�����

    // ͼ��Ԥ����Ȼ�󱣴浽m_queue_image_with_pose���У�����msg->header.stamp.toSec()Ϊ����Ϊ��λ��ʱ���
    process_image( temp_img, msg->header.stamp.toSec());
}

double last_accept_time = 0;
int    buffer_max_frame = 0;   //�����֡��
int    total_frame_count = 0;  //�����ͼ��֡��


//1.���ʱ�������ʼ������
//2.���� service_pub_rgb_maps �� service_VIO_update �߳�
//3.ȥ������ͼ����
void   R3LIVE::process_image( cv::Mat &temp_img, double msg_time )  //һ֡���ݵ�Ԥ����
{
    cv::Mat img_get;
    // ���ͼ��rows�Ƿ�����
    if ( temp_img.rows == 0 )  //rowsΪͼ���������Ϊ��������
    {
        cout << "Process image error, image rows =0 " << endl;
        return;
    }
    // ���ʱ����Ƿ�����
    if ( msg_time < last_accept_time ) //last_accept_time��ʼΪ0
    {
        cout << "Error, image time revert!!" << endl;
        return;
    }
    // ����ͼ�����Ƶ�ʣ���ֹƵ�ʹ���
    if ( ( msg_time - last_accept_time ) < ( 1.0 / m_control_image_freq ) * 0.9 ) //m_control_image_freq�趨��ͼ�����Ƶ�ʣ�ʱ�������С��Ҫ���������˳�����ֹ̫Ƶ��
    {
        return;
    }
    last_accept_time = msg_time; //��¼��һ֡ʱ�䣬������һ֡����
    // ����ǵ�һ������
    if ( m_camera_start_ros_tim < 0 )  //��һ�����У�
    {
        m_camera_start_ros_tim = msg_time; //�޸ĵ�һ�����б��
        m_vio_scale_factor = m_vio_image_width * m_image_downsample_ratio / temp_img.cols; // 320 * 24      ͼ����*1/ͼ������������һ���������ӣ����ڽ�ͼ���������ŵ��ʺ� VIO ����ĳߴ�
        // load_vio_parameters();  ����vio����
        //���� set_initial_camera_parameter �����������������Գ�ʼ��������� ,,,��ϵͳ����Э�������
        set_initial_camera_parameter( g_lio_state, m_camera_intrinsic.data(), m_camera_dist_coeffs.data(), m_camera_ext_R.data(),
                                      m_camera_ext_t.data(), m_vio_scale_factor );
        cv::eigen2cv( g_cam_K, intrinsic ); //�� Eigen ���� g_cam_K ת��Ϊ OpenCV ���� intrinsic
        cv::eigen2cv( g_cam_dist, dist_coeffs ); //�� Eigen ���� g_cam_dist ת��Ϊ OpenCV ���� dist_coeffs
        // ��ʼ������  ʹ�� OpenCV �� initUndistortRectifyMap ��������ʼ��ȥ�����У��ӳ��
        //m_ud_map1 �� m_ud_map2�������ȥ�����У��ӳ��ͼ��
        initUndistortRectifyMap( intrinsic, dist_coeffs, cv::Mat(), intrinsic, cv::Size( m_vio_image_width / m_vio_scale_factor, m_vio_image_heigh / m_vio_scale_factor ),
                                 CV_16SC2, m_ud_map1, m_ud_map2 );
        // ���������߳�
        //�� &R3LIVE::service_pub_rgb_maps ��Ϊ�����ύ�� m_thread_pool_ptr����̳߳���
        m_thread_pool_ptr->commit_task( &R3LIVE::service_pub_rgb_maps, this);//��һ�����е�ʱ��Ҳ�ᴥ��rgb map�ķ����̣߳���RGB���Ƶ�ͼ��ֳ����ӵ��ƣ�1000������
        m_thread_pool_ptr->commit_task( &R3LIVE::service_VIO_update, this); //��һ�����е�ʱ��ᴥ�����̣߳�����ֱ�ӵ���VIO����ESIKF�Ĳ���
        // ��ʼ�����ݼ�¼��
        m_mvs_recorder.init( g_cam_K, m_vio_image_width / m_vio_scale_factor, &m_map_rgb_pts );//�����ڲΣ�ȫ�ֵ�ͼ
        m_mvs_recorder.set_working_dir( m_map_output_dir );
    }
    //ͼ���²�����
    if ( m_image_downsample_ratio != 1.0 )
    {
        cv::resize( temp_img, img_get, cv::Size( m_vio_image_width / m_vio_scale_factor, m_vio_image_heigh / m_vio_scale_factor ) );
    }
    else
    {
        img_get = temp_img; // clone ?
    }
    std::shared_ptr< Image_frame > img_pose = std::make_shared< Image_frame >( g_cam_K );//���ø�֡ͼ����Ϣ����img_pose
    // �Ƿ񷢲�ԭʼimg
    if ( m_if_pub_raw_img )
    {
        img_pose->m_raw_img = img_get; //��img_get��Ϊԭʼͼ��
    }
    // ��img_getΪ���룬����ȥ���䣬�����img_pose->m_img
    cv::remap( img_get, img_pose->m_img, m_ud_map1, m_ud_map2, cv::INTER_LINEAR ); //������ͼ�� img_get ����������ӳ�䵽Ŀ��ͼ�� img_pose->m_img ��
    // cv::imshow("sub Img", img_pose->m_img);
    img_pose->m_timestamp = msg_time;  //��ʱ�������img_pose->m_timestamp
    img_pose->init_cubic_interpolation();  // ת�Ҷ�ͼ
    img_pose->image_equalize();              // ֱ��ͼ���⻯������ͼ��Ч��
    m_camera_data_mutex.lock();
    m_queue_image_with_pose.push_back( img_pose );    // ���浽����
    m_camera_data_mutex.unlock();
    total_frame_count++;

    // ����buffer���������ϼӴ�buffer_max_frame��С
    if ( m_queue_image_with_pose.size() > buffer_max_frame )
    {
        buffer_max_frame = m_queue_image_with_pose.size();
    }

    // cout << "Image queue size = " << m_queue_image_with_pose.size() << endl;
}

//û���õ������� ROS ���������������Ӿ�������̼� (VIO) ��������������������Щ�������浽�ڲ����ݽṹ��
void R3LIVE::load_vio_parameters()
{

    std::vector< double > camera_intrinsic_data, camera_dist_coeffs_data, camera_ext_R_data, camera_ext_t_data;//�������
    m_ros_node_handle.getParam( "r3live_vio/image_width", m_vio_image_width );//��ROS��������ȡͼ����
    m_ros_node_handle.getParam( "r3live_vio/image_height", m_vio_image_heigh );
    m_ros_node_handle.getParam( "r3live_vio/camera_intrinsic", camera_intrinsic_data );  //��ROS������������������ڲε�camera_intrinsic_data
    m_ros_node_handle.getParam( "r3live_vio/camera_dist_coeffs", camera_dist_coeffs_data ); //�������ϵ��
    m_ros_node_handle.getParam( "r3live_vio/camera_ext_R", camera_ext_R_data ); //��������ת����
    m_ros_node_handle.getParam( "r3live_vio/camera_ext_t", camera_ext_t_data ); //������ƽ������

    CV_Assert( ( m_vio_image_width != 0 && m_vio_image_heigh != 0 ) ); //ȷ��ͼ���Ⱥ͸߶Ȳ�Ϊ�㡣
    //�����������������⣺����еȴ�
    if ( ( camera_intrinsic_data.size() != 9 ) || ( camera_dist_coeffs_data.size() != 5 ) || ( camera_ext_R_data.size() != 9 ) ||
         ( camera_ext_t_data.size() != 3 ) )
    {

        cout << ANSI_COLOR_RED_BOLD << "Load VIO parameter fail!!!, please check!!!" << endl;
        printf( "Load camera data size = %d, %d, %d, %d\n", ( int ) camera_intrinsic_data.size(), camera_dist_coeffs_data.size(),
                camera_ext_R_data.size(), camera_ext_t_data.size() );
        cout << ANSI_COLOR_RESET << endl;
        std::this_thread::sleep_for( std::chrono::seconds( 3000000 ) );
    }
    //����������ת��Ϊ Eigen ���� :
    m_camera_intrinsic = Eigen::Map< Eigen::Matrix< double, 3, 3, Eigen::RowMajor > >( camera_intrinsic_data.data() ); //���ڴ�����ֱ�Ӹ���Map��
    m_camera_dist_coeffs = Eigen::Map< Eigen::Matrix< double, 5, 1 > >( camera_dist_coeffs_data.data() );
    m_camera_ext_R = Eigen::Map< Eigen::Matrix< double, 3, 3, Eigen::RowMajor > >( camera_ext_R_data.data() );
    m_camera_ext_t = Eigen::Map< Eigen::Matrix< double, 3, 1 > >( camera_ext_t_data.data() );
    //��ӡ���
    cout << "[Ros_parameter]: r3live_vio/Camera Intrinsic: " << endl;
    cout << m_camera_intrinsic << endl;
    cout << "[Ros_parameter]: r3live_vio/Camera distcoeff: " << m_camera_dist_coeffs.transpose() << endl;
    cout << "[Ros_parameter]: r3live_vio/Camera extrinsic R: " << endl;
    cout << m_camera_ext_R << endl;
    cout << "[Ros_parameter]: r3live_vio/Camera extrinsic T: " << m_camera_ext_t.transpose() << endl;
    std::this_thread::sleep_for( std::chrono::seconds( 1 ) );
}

//����IMU��λ�ˣ�����һ֡ͼ���λ�˺��ڲ�
void R3LIVE::set_image_pose( std::shared_ptr< Image_frame > &image_pose, const StatesGroup &state )
{
    mat_3_3 rot_mat = state.rot_end;    //g_lio_state;  ϵͳ״̬����ת��ƽ�ƣ���IMU����������ϵ����ת
    vec_3   t_vec = state.pos_end;          //g_lio_state;  ϵͳ״̬����ת��ƽ�ƣ���IMU����������ϵ��ƽ��
    // �������������ϵ��λ�ã�����
    vec_3   pose_t = rot_mat * state.pos_ext_i2c + t_vec;   // IMU����������ϵ����ת * IMU�������ƽ�� + IMU����������ϵ��ƽ��== �������������ϵ��ƽ��
     // �������������ϵ����̬
    mat_3_3 R_w2c = rot_mat * state.rot_ext_i2c;   //  IMU����������ϵ����ת * IMU���������ת == �������������ϵ����ת

    image_pose->set_pose( eigen_q( R_w2c ), pose_t );  //ʹ��eigen_q��R_w2cתΪ��Ԫ������������һ֡ͼ���λ��
    //��ȡ����ڲθ�����һ֡��
    image_pose->fx = state.cam_intrinsic( 0 ); 
    image_pose->fy = state.cam_intrinsic( 1 );
    image_pose->cx = state.cam_intrinsic( 2 );
    image_pose->cy = state.cam_intrinsic( 3 );
    //������һ֡����ڲ�
    image_pose->m_cam_K << image_pose->fx, 0, image_pose->cx, 0, image_pose->fy, image_pose->cy, 0, 0, 1;
    scope_color( ANSI_COLOR_CYAN_BOLD );
    // cout << "Set Image Pose frm [" << image_pose->m_frame_idx << "], pose: " << eigen_q(rot_mat).coeffs().transpose()
    // << " | " << t_vec.transpose()
    // << " | " << eigen_q(rot_mat).angularDistance( eigen_q::Identity()) *57.3 << endl;
    // image_pose->inverse_pose();
}

//���������λ����Ϣ�����·����Ϣ
void R3LIVE::publish_camera_odom( std::shared_ptr< Image_frame > &image, double msg_time )
{
    eigen_q            odom_q = image->m_pose_w2c_q;//��ȡ��һ֡��λ����Ϣ
    vec_3              odom_t = image->m_pose_w2c_t;
    nav_msgs::Odometry camera_odom;  //�����̼�
    camera_odom.header.frame_id = "world";
    camera_odom.child_frame_id = "/aft_mapped";
    camera_odom.header.stamp = ros::Time::now(); // ros::Time().fromSec(last_timestamp_lidar);����ʱ���
    camera_odom.pose.pose.orientation.x = odom_q.x(); //����λ�˵���Ԫ��
    camera_odom.pose.pose.orientation.y = odom_q.y();
    camera_odom.pose.pose.orientation.z = odom_q.z();
    camera_odom.pose.pose.orientation.w = odom_q.w();
    camera_odom.pose.pose.position.x = odom_t( 0 );  //λ��
    camera_odom.pose.pose.position.y = odom_t( 1 );
    camera_odom.pose.pose.position.z = odom_t( 2 );
    pub_odom_cam.publish( camera_odom );  //���������̼���Ϣ

    //PoseStamped ͨ���ᱻ������¼��ʷλ�ˣ����γ�һ��·����camera_path��
    //��ʾ����ʱ���������ϵ��Ϣ��λ����Ϣ
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
    //camera_path ��������¼�����λ�ú���̬��ʷ������ÿ�ε���ʱ���µ�λ����Ϣ��ӵ�·����
    camera_path.header.frame_id = "world";
    camera_path.poses.push_back( msg_pose ); //�� PoseStamped ��Ϣ��ӵ� camera_path.poses �У��γ�·����һ���µ�
    pub_path_cam.publish( camera_path );//����
}

//û���õ�
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

// ANCHOR - VIO preintegration    �� VIO ϵͳ�д��� IMU ���ݣ�ͨ��Ԥ���ּ������״̬
bool R3LIVE::vio_preintegration( StatesGroup &state_in, StatesGroup &state_out, double current_frame_time )
{
    state_out = state_in;  //���״̬����Ϊ����״̬
    // ��鵱ǰ֡��ʱ���Ƿ�С�ڵ�����һ�θ��µ�ʱ�䣬ͼ��ʱ��������µģ�û�д����
    if ( current_frame_time <= state_in.last_update_time ) 
    {
        // cout << ANSI_COLOR_RED_BOLD << "Error current_frame_time <= state_in.last_update_time | " <<
        // current_frame_time - state_in.last_update_time << ANSI_COLOR_RESET << endl;
        return false;
    }
    mtx_buffer.lock();
    std::deque< sensor_msgs::Imu::ConstPtr > vio_imu_queue;   //����IMU���ݵĶ���
    // ����imu_buffer_vio�����е�Ԫ�أ�������뵽vio_imu_queue
    for ( auto it = imu_buffer_vio.begin(); it != imu_buffer_vio.end(); it++ )
    {
        vio_imu_queue.push_back( *it ); //������ӵ�vio_imu_queue
        // ���ʱ������ڵ�ǰ֡��ʱ�䣬������ѭ������˼���ǣ������µ���һ֡ͼ���ֹͣ����ֻ������ǰ������
        if ( ( *it )->header.stamp.toSec() > current_frame_time )
        {
            break;
        }
    }
    // ��imu_buffer_vio������Ϊ��ʱִ��ѭ��������Ԥ��������ʱ��ֻ����ǰ0.2������ݣ���̫�ϵ�������Ϊû�й�ϵ
    while ( !imu_buffer_vio.empty() )
    {
        // ��ȡimu_buffer_vio�����е�һ��Ԫ�ص�ʱ���
        double imu_time = imu_buffer_vio.front()->header.stamp.toSec();
        // imu��current_frame_time��ʱ���
        if ( imu_time < current_frame_time - 0.2 ) //ֻ����ǰ0.2������ݣ�̫�ϵ�������Ϊû�й�ϵ
        {
            // ����Ԫ�ش��������Ƴ�
            imu_buffer_vio.pop_front();
        }
        else
        {
            break;
        }
    }
    // cout << "Current VIO_imu buffer size = " << imu_buffer_vio.size() << endl;
    //����IMUԤ����
    state_out = m_imu_process->imu_preintegration( state_out, vio_imu_queue, current_frame_time - vio_imu_queue.back()->header.stamp.toSec() );
    eigen_q q_diff( state_out.rot_end.transpose() * state_in.rot_end ); //�������ǰ������״̬����ת����
    // cout << "Pos diff = " << (state_out.pos_end - state_in.pos_end).transpose() << endl;
    // cout << "Euler diff = " << q_diff.angularDistance(eigen_q::Identity()) * 57.3 << endl;
    mtx_buffer.unlock();
    // ����ʱ����Ϣ
    state_out.last_update_time = current_frame_time; //����ǰͼ��ʱ�丳�������ϵͳ״̬
    return true;
}

// ANCHOR - huber_loss  ������ʧ��huber�˺���   ���� Huber ��ʧ��������������
//Huber ��ʧ������һ�����ڴ���ع��������ʧ������������˾������;��������ŵ㣬����߶��쳣ֵ��³����
//���ݸ�������ͶӰ��reprojection_error������Ⱥ����ֵ��outlier_threshold�������������������
double get_huber_loss_scale( double reprojection_error, double outlier_threshold = 1.0 )
{
    // http://ceres-solver.org/nnls_modeling.html#lossfunction
    double scale = 1.0;
    if ( reprojection_error / outlier_threshold < 1.0 )  //��ʾ��ͶӰ�����Խ�С��������ֵ����ʱ�����ص� scale ����Ϊ 1.0��
    {
        scale = 1.0;
    }
    else
    {
        scale = ( 2 * sqrt( reprojection_error ) / sqrt( outlier_threshold ) - 1.0 ) / reprojection_error;  //�����ʽ�� Huber ��ʧ�����ļ�������ֵ���б궨�������Դ����Ĵ���ʽ
    }
    return scale;
}

// ANCHOR - VIO_esikf    ����֡��֡VIO   ������ͶӰ��� =�������Ż�ϵͳ״̬
const int minimum_iteration_pts = 10;  //��С���ܹ��Ż��ĵ���
bool      R3LIVE::vio_esikf( StatesGroup &state_in, Rgbmap_tracker &op_track ) //���룺���״̬����������
{
    Common_tools::Timer tim;  //����ʱ��
    tim.tic();   //��ʼ��ʱ
    scope_color( ANSI_COLOR_BLUE_BOLD );       //���ÿ���̨������ı���ɫΪ��ɫ����
    StatesGroup state_iter = state_in;   //�½�һ��״̬����
    if ( !m_if_estimate_intrinsic ) // When disable the online intrinsic calibration.  �����ܽ�������ڲα궨ʱ��
    {
        state_iter.cam_intrinsic << g_cam_K( 0, 0 ), g_cam_K( 1, 1 ), g_cam_K( 0, 2 ), g_cam_K( 1, 2 ); //ֱ�ӽ�������������Ϊ�ڲ�
    }

    if ( !m_if_estimate_i2c_extrinsic )  //����ʹ�ñ궨��IMU�����֮�����α궨ʱ��
    {
        state_iter.pos_ext_i2c = m_inital_pos_ext_i2c;    //��ԭʼ��������Ϊ�����
        state_iter.rot_ext_i2c = m_inital_rot_ext_i2c;       //�轫ԭʼ��������Ϊ�����
    }

    Eigen::Matrix< double, -1, -1 >                       H_mat;  //���̶���С�ľ���
    Eigen::Matrix< double, -1, 1 >                        meas_vec;   //������һ����̬��С�������� meas_vec
    Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES > G, H_T_H, I_STATE;  //������������СΪ DIM_OF_STATES x DIM_OF_STATES �ľ���
    Eigen::Matrix< double, DIM_OF_STATES, 1 >             solution;  //�洢���
    Eigen::Matrix< double, -1, -1 >                       K, KH;   //���̶���С�ľ���
    Eigen::Matrix< double, DIM_OF_STATES, DIM_OF_STATES > K_1;   //������һ����СΪ DIM_OF_STATES x DIM_OF_STATES �ľ��� K_1

    Eigen::SparseMatrix< double > H_mat_spa, H_T_H_spa, K_spa, KH_spa, vec_spa, I_STATE_spa;  //���弸��ϡ�����
    I_STATE.setIdentity();  //��λ����
    I_STATE_spa = I_STATE.sparseView();  //����ĵ�λ����ϡ�����
    double fx, fy, cx, cy, time_td;  //���ڴ洢������ڲκ���������

    int                   total_pt_size = op_track.m_map_rgb_pts_in_current_frame_pos.size();//��ȡ��ǰ֡���ٵ��ĵ�ͼ���� RGB �������
    //�������� std::vector���ֱ����ڴ洢��һ�κ͵�ǰ����ͶӰ���
    std::vector< double > last_reprojection_error_vec( total_pt_size ), current_reprojection_error_vec( total_pt_size );
    //����10���㲻���Ż���
    if ( total_pt_size < minimum_iteration_pts )
    {
        state_in = state_iter;
        return false;
    }
    H_mat.resize( total_pt_size * 2, DIM_OF_STATES );//����������*2    ������29
    meas_vec.resize( total_pt_size * 2, 1 );   //����* 2 �к� 1 ��
    double last_repro_err = 3e8;  //�����ϴ���ͶӰ����ֵ
    int    avail_pt_count = 0;  //��Ч���ٵ������
    double last_avr_repro_err = 0;  //���ڸ�����һ�ε�ƽ����ͶӰ���

    double acc_reprojection_error = 0;  //�ۻ���ͶӰ���
    double img_res_scale = 1.0;  //ͼ��ֱ�����������
    //�Ż����Σ�
    for ( int iter_count = 0; iter_count < esikf_iter_times; iter_count++ )
    {

        // cout << "========== Iter " << iter_count << " =========" << endl;
        mat_3_3 R_imu = state_iter.rot_end;   //��IMU����������ϵ����ת
        vec_3   t_imu = state_iter.pos_end;    //��IMU����������ϵ��ƽ��
        vec_3   t_c2w = R_imu * state_iter.pos_ext_i2c + t_imu;   //�����������������ϵ�е�λ��
        mat_3_3 R_c2w = R_imu * state_iter.rot_ext_i2c; // �����������������ϵ����ת����
        //��ȡ����ڲ�
        fx = state_iter.cam_intrinsic( 0 );
        fy = state_iter.cam_intrinsic( 1 );
        cx = state_iter.cam_intrinsic( 2 );
        cy = state_iter.cam_intrinsic( 3 );
        time_td = state_iter.td_ext_i2c_delta;//ʱ���

        vec_3   t_w2c = -R_c2w.transpose() * t_c2w;  //�����������絽���
        mat_3_3 R_w2c = R_c2w.transpose();//�����������絽���
        int     pt_idx = -1;  //�������
        acc_reprojection_error = 0;  //�ۻ���ͶӰ���
        vec_3               pt_3d_w, pt_3d_cam;   //��������ϵ���������ϵ�е� 3D ��
        vec_2               pt_img_measure, pt_img_proj, pt_img_vel;  //ͼ������ϵ�еĵ㣬�ֱ��ʾ�����㡢ͶӰ����ٶȵ�
        eigen_mat_d< 2, 3 > mat_pre;  // 2x3 �� 3x3 �ľ���
        eigen_mat_d< 3, 3 > mat_A, mat_B, mat_C, mat_D, pt_hat;
        H_mat.setZero(); //ȫ����Ϊ0
        solution.setZero();
        meas_vec.setZero();
        avail_pt_count = 0;  //��¼��Ч�������
        //������ǰ��һ֡����׷�ٵ��ĵ�
        for ( auto it = op_track.m_map_rgb_pts_in_last_frame_pos.begin(); it != op_track.m_map_rgb_pts_in_last_frame_pos.end(); it++ )
        {
            pt_3d_w = ( ( RGB_pts * ) it->first )->get_pos();  //��ȡ�õ�����ά��������ϵ�е�λ�� 
            pt_img_vel = ( ( RGB_pts * ) it->first )->m_img_vel;  //��ȡ�õ���ٶ�
            pt_img_measure = vec_2( it->second.x, it->second.y );  // �ø��ٵ�ͼ��   ͨ�������� ����һ֡ͼ���е�λ�ã�
            pt_3d_cam = R_w2c * pt_3d_w + t_w2c;   //����ͼ�����������ϵ�任����ǰ֡�������ϵ
            // 1. ��������ڲν��������ϵת������������ϵ��������
            // 2. �������-imuʱ��ƫ��µ�����
            //���յõ� ����ͼ���ڵ�ǰ֡ͼ���ͶӰ�������� pt_img_proj
            pt_img_proj = vec_2( fx * pt_3d_cam( 0 ) / pt_3d_cam( 2 ) + cx, fy * pt_3d_cam( 1 ) / pt_3d_cam( 2 ) + cy ) + time_td * pt_img_vel;
            //������ͶӰ���������������Ĵ�С�������ڵ�����⣬ֻ������ȷ��huber�˺���
            double repro_err = ( pt_img_proj - pt_img_measure ).norm();  //��ͼ��ӳ�䵽��һ֡��xy��  -  ��ͼ��ӳ�䵽��һ֡������������һ֡��.norm()�������������
            double huber_loss_scale = get_huber_loss_scale( repro_err );  //��������С��ȷ����ʧ�˺���
            pt_idx++;  //������+1����ʼ��һ������0
            acc_reprojection_error += repro_err;  //�ۼ����е����ͶӰ���
            // if (iter_count == 0 || ((repro_err - last_reprojection_error_vec[pt_idx]) < 1.5))
            //����ͶӰ����¼��last_reprojection_error_vec��
            if ( iter_count == 0 || ( ( repro_err - last_avr_repro_err * 5.0 ) < 0 ) )   //��һ�ε����Ż� ���� ��ͶӰ����ƽ����ͶӰ����屶��С
            {
                last_reprojection_error_vec[ pt_idx ] = repro_err;
            }
            else
            {
                last_reprojection_error_vec[ pt_idx ] = repro_err;
            }
            avail_pt_count++;  //���������1
            // Appendix E of r2live_Supplementary_material.
            // https://github.com/hku-mars/r2live/blob/master/supply/r2live_Supplementary_material.pdf
            // ���ض���������ſɱȣ�
            //�þ������ڽ��������ϵ�е� 3D ��ӳ�䵽ͼ������ϵ��
            mat_pre << fx / pt_3d_cam( 2 ), 0, -fx * pt_3d_cam( 0 ) / pt_3d_cam( 2 ), 0, fy / pt_3d_cam( 2 ), -fy * pt_3d_cam( 1 ) / pt_3d_cam( 2 );
            //����������ϵ�µ���ά��ת����IMU��
            pt_hat = Sophus::SO3d::hat( ( R_imu.transpose() * ( pt_3d_w - t_imu ) ) );  //IMUϵ�µ�ͼ�� p(IMU)^ = R(IMU <-- W) * ( p(W) - p(IMU) )
            //�����ſɱȾ���:
            mat_A = state_iter.rot_ext_i2c.transpose() * pt_hat;  //��ά����IMU-�����
            mat_B = -state_iter.rot_ext_i2c.transpose() * ( R_imu.transpose() );   //���-������
            mat_C = Sophus::SO3d::hat( pt_3d_cam );   //���� -�����
            mat_D = -state_iter.rot_ext_i2c.transpose();  //��� -��IMU
            meas_vec.block( pt_idx * 2, 0, 2, 1 ) = ( pt_img_proj - pt_img_measure ) * huber_loss_scale / img_res_scale;  //�۲��������

            //�Ը��������������ſɱȣ���Ρ��ٶȡ��ڲ�
            H_mat.block( pt_idx * 2, 0, 2, 3 ) = mat_pre * mat_A * huber_loss_scale;  // H, 1-2�У�ǰ3��, ��R(IMU)�ſɱ�
            H_mat.block( pt_idx * 2, 3, 2, 3 ) = mat_pre * mat_B * huber_loss_scale;  // H, 1-2�У�4-6�У���P(IMU)�ſɱ�
            if ( DIM_OF_STATES > 24 )
            {
                // Estimate time td.   ��ʱ���ĵ���
                H_mat.block( pt_idx * 2, 24, 2, 1 ) = pt_img_vel * huber_loss_scale;  // H��1-2�У� 25-26�У��������ٶ��ſɱ�
                // H_mat(pt_idx * 2, 24) = pt_img_vel(0) * huber_loss_scale;
                // H_mat(pt_idx * 2 + 1, 24) = pt_img_vel(1) * huber_loss_scale;
            }
            if ( m_if_estimate_i2c_extrinsic )
            {
                H_mat.block( pt_idx * 2, 18, 2, 3 ) = mat_pre * mat_C * huber_loss_scale;  //H ,1-2�У�19-21�У������R(IMU<--C)�ſɱ�
                H_mat.block( pt_idx * 2, 21, 2, 3 ) = mat_pre * mat_D * huber_loss_scale;  //H ,1-2�У�22-24�У������t(IMU<--C)�ſɱ�
            }

            if ( m_if_estimate_intrinsic )
            {
                H_mat( pt_idx * 2, 25 ) = pt_3d_cam( 0 ) / pt_3d_cam( 2 ) * huber_loss_scale;  //H,1�У�26�У����ڲ�fx�ſɱ�
                H_mat( pt_idx * 2 + 1, 26 ) = pt_3d_cam( 1 ) / pt_3d_cam( 2 ) * huber_loss_scale;  //H,2�У�27�У����ڲ�fy�ſɱ�
                H_mat( pt_idx * 2, 27 ) = 1 * huber_loss_scale;  //H,1�У�28�У����ڲ�cx�ſɱ�
                H_mat( pt_idx * 2 + 1, 28 ) = 1 * huber_loss_scale;  //H,2�У�29�У����ڲ�cy�ſɱ�
            }
        }
        H_mat = H_mat / img_res_scale;  //����
        acc_reprojection_error /= total_pt_size;   //��ÿ����ƽ������ͶӰ���

        last_avr_repro_err = acc_reprojection_error;  //��¼���ε�ƽ����ͶӰ���
        if ( avail_pt_count < minimum_iteration_pts )  //�������С����С����
        {
            break;
        }

        H_mat_spa = H_mat.sparseView();  //תΪϡ�����   ��ʡ�ڴ�ͼ���ʱ��
        Eigen::SparseMatrix< double > Hsub_T_temp_mat = H_mat_spa.transpose();  //ת�ã�Hsub_T_temp_mat
        vec_spa = ( state_iter - state_in ).sparseView();  //ϵͳ״̬�仯
        H_T_H_spa = Hsub_T_temp_mat * H_mat_spa; //�ſɱ� �� �ſɱ�ת��
        // Notice that we have combine some matrix using () in order to boost the matrix multiplication.
        Eigen::SparseMatrix< double > temp_inv_mat =
            ( ( H_T_H_spa.toDense() + eigen_mat< -1, -1 >( state_in.cov * m_cam_measurement_weight ).inverse() ).inverse() ).sparseView();
        KH_spa = temp_inv_mat * ( Hsub_T_temp_mat * H_mat_spa );//�������������
        //���ڼ�����º��״̬���ơ�������˵���������Ԥ��ֵ���µĲ�������
        solution = ( temp_inv_mat * ( Hsub_T_temp_mat * ( ( -1 * meas_vec.sparseView() ) ) ) - ( I_STATE_spa - KH_spa ) * vec_spa ).toDense();  //���ڼ�����º��״̬����

        state_iter = state_iter + solution;//����״̬
        //����ƽ����ͶӰ���  �ӽ��� ��һ��ƽ����ͶӰ���   =�� �����Ż�
        if ( fabs( acc_reprojection_error - last_repro_err ) < 0.01 )
        {
            break;
        }
        last_repro_err = acc_reprojection_error;  //��¼����ƽ����ͶӰ���
    }
    //�����Ż����ˣ�
    if ( avail_pt_count >= minimum_iteration_pts )
    {
        state_iter.cov = ( ( I_STATE_spa - KH_spa ) * state_iter.cov.sparseView() ).toDense();  //����Э�������
    }

    state_iter.td_ext_i2c += state_iter.td_ext_i2c_delta;  //????????
    state_iter.td_ext_i2c_delta = 0;  //IMU����������
    state_in = state_iter;  //���������״̬ =���Ż����״̬
    return true;
}

//��2���Ż���֡����ͼVIO   ����һ�����µ�״̬�ͷ�������ϣ����ø��ٵ�Ĺ������ٽ���ESIKF״̬����
//�ؼ�����۲��������״̬�����ĸ���ƫ������H����
bool R3LIVE::vio_photometric( StatesGroup &state_in, Rgbmap_tracker &op_track, std::shared_ptr< Image_frame > &image )
{
    Common_tools::Timer tim;  //��ʱ��
    tim.tic(); //��ʼ��ʱ
    StatesGroup state_iter = state_in;  //����ϵͳ״̬
    if (!m_if_estimate_intrinsic)     // When disable the online intrinsic calibration.
    {
        state_iter.cam_intrinsic << g_cam_K(0, 0), g_cam_K(1, 1), g_cam_K(0, 2), g_cam_K(1, 2);  //����ڲ� =��state_iter.cam_intrinsic
    }
    if (!m_if_estimate_i2c_extrinsic) // When disable the online extrinsic calibration.
    {
        state_iter.pos_ext_i2c = m_inital_pos_ext_i2c;  //IMU�������ε�ƽ��
        state_iter.rot_ext_i2c = m_inital_rot_ext_i2c;  //IMU�������ε���ת
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

    int                   total_pt_size = op_track.m_map_rgb_pts_in_current_frame_pos.size();  //��ȡ���ٵ������
    std::vector< double > last_reprojection_error_vec( total_pt_size ), current_reprojection_error_vec( total_pt_size ); //�洢��һ�κ͵�ǰ��ͶӰ��������
    if ( total_pt_size < minimum_iteration_pts )  //�����������˳��Ż�
    {
        state_in = state_iter;
        return false;
    }

    int err_size = 3;
    H_mat.resize( total_pt_size * err_size, DIM_OF_STATES );//�����ſɱȾ����С
    meas_vec.resize( total_pt_size * err_size, 1 );  //�۲�����
    R_mat_inv.resize( total_pt_size * err_size, total_pt_size * err_size );  //��ʾЭ����������

    double last_repro_err = 3e8;//��ʼ����ͶӰ���
    int    avail_pt_count = 0;
    double last_avr_repro_err = 0;  //��һ��ƽ����ͶӰ���
    int    if_esikf = 1; //�Ƿ�����esikf

    double acc_photometric_error = 0; //�����ۼƹ�����
#if DEBUG_PHOTOMETRIC
    printf("==== [Image frame %d] ====\r\n", g_camera_frame_idx);
#endif
    for ( int iter_count = 0; iter_count < 2; iter_count++ )//��������
    {
        mat_3_3 R_imu = state_iter.rot_end;//��ȡIMU�����Բ�����Ԫ������ת�����λ��������
        vec_3   t_imu = state_iter.pos_end;
        vec_3   t_c2w = R_imu * state_iter.pos_ext_i2c + t_imu; //�����������������ϵ��ƽ������
        mat_3_3 R_c2w = R_imu * state_iter.rot_ext_i2c; // world to camera frame�����������ϵ����������ϵ����ת���� 
        //����ڲ�:
        fx = state_iter.cam_intrinsic( 0 );
        fy = state_iter.cam_intrinsic( 1 );
        cx = state_iter.cam_intrinsic( 2 );
        cy = state_iter.cam_intrinsic( 3 );
        time_td = state_iter.td_ext_i2c_delta; //�����ʱ��ƫ������

        vec_3   t_w2c = -R_c2w.transpose() * t_c2w; //������������ϵ���������ϵ��ƽ������ t_w2c
        mat_3_3 R_w2c = R_c2w.transpose();//������������ϵ���������ϵ����ת���� 
        int     pt_idx = -1;  //�������
        acc_photometric_error = 0; //��ʼ���ۼƹ�����
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
        tim.tic( "Build_cost" );//��ʼ��ʱ��"Build_cost"
        //�洢RGB�������Ӧͼ��λ�õ�ӳ��
        ///�������е�ͼ�㣬�����ſɱȾ���͹۲�����
        for ( auto it = op_track.m_map_rgb_pts_in_last_frame_pos.begin(); it != op_track.m_map_rgb_pts_in_last_frame_pos.end(); it++ )
        {
            if ( ( ( RGB_pts * ) it->first )->m_N_rgb < 3 )//û��rgb����
            {
                continue;
            }
            pt_idx++;//��������1
            pt_3d_w = ( ( RGB_pts * ) it->first )->get_pos();  //��ȡ��ά��
            // ��������ع�ʱ���imu��ʱ���
            pt_img_vel = ( ( RGB_pts * ) it->first )->m_img_vel;//��ȡͼ�����ٶȡ�
            pt_img_measure = vec_2( it->second.x, it->second.y );  //��ȡͼ���ά��
            // ����ͼ��ת���������ϵ�º�ͶӰ�����ƽ��
            pt_3d_cam = R_w2c * pt_3d_w + t_w2c; //����ͼ�����������ϵת�����������ϵ
            pt_img_proj = vec_2( fx * pt_3d_cam( 0 ) / pt_3d_cam( 2 ) + cx, fy * pt_3d_cam( 1 ) / pt_3d_cam( 2 ) + cy ) + time_td * pt_img_vel;//����ͶӰ����������

            // ÿ��ͼ��֡�ĸ��ٵ��ڵ�ͼ�д���һ����������˿ɻ�ȡ�ϴ��ںϺ�ĵ�ͼ��rgbֵ��Э����
            vec_3   pt_rgb = ( ( RGB_pts * ) it->first )->get_rgb();  //��ȡ��ά���RGBֵ
            mat_3_3 pt_rgb_info = mat_3_3::Zero();
            mat_3_3 pt_rgb_cov = ( ( RGB_pts * ) it->first )->get_rgb_cov();  //��ͼ��rgb�Խ�Э�������
            // ��Ϊ�۲�����Э��������ڸ��¹�ʽ�г��������λ��
            for ( int i = 0; i < 3; i++ )
            {
                pt_rgb_info( i, i ) = 1.0 / pt_rgb_cov( i, i ) ; //����Э�������������ĶԽ���Ԫ�ء�
                R_mat_inv( pt_idx * err_size + i, pt_idx * err_size + i ) = pt_rgb_info( i, i ); //����Ϊ��Э��������Ԫ��
                // R_mat_inv( pt_idx * err_size + i, pt_idx * err_size + i ) =  1.0;
            }
            vec_3  obs_rgb_dx, obs_rgb_dy;
            // ��ȡ��һ֡�и��ٵ�������ؼ�rgbֵ
            //ͶӰ���Ӱ���ֵ��ȡrgb������x��y�����ϵ�rgb��ֵ
            vec_3  obs_rgb = image->get_rgb( pt_img_proj( 0 ), pt_img_proj( 1 ), 0, &obs_rgb_dx, &obs_rgb_dy );
            vec_3  photometric_err_vec = ( obs_rgb - pt_rgb );  //rgb���: ͼ���Ӧ��rgbֵ - ��ά��ͼ��rgbֵ
            // huber loss��������������Ⱥ����Ż���Ӱ��
            double huber_loss_scale = get_huber_loss_scale( photometric_err_vec.norm() );//����Huber��ʧ����������
            photometric_err_vec *= huber_loss_scale; //��Huber��ʧ��������Ӧ����RGB�������
            double photometric_err = photometric_err_vec.transpose() * pt_rgb_info * photometric_err_vec;//����Ӱ�����:          e^T * info_rgb(map) * e

            acc_photometric_error += photometric_err; //���������Ӱ������ۼӵ��ܵĹ�����

            last_reprojection_error_vec[ pt_idx ] = photometric_err;//��¼ÿ����Ĺ�����

            mat_photometric.setZero();
            mat_photometric.col( 0 ) = obs_rgb_dx; //��x��y�����RGB��ֵ�ֱ𸳸�mat_photometric������С�
            mat_photometric.col( 1 ) = obs_rgb_dy;

            avail_pt_count++; //���ӿ��õ����
            // �ڲ�ͶӰ�Ե�ͼ����ſɱȾ���
            // ������������������    ,������������������ſɱȾ��� 
            mat_pre << fx / pt_3d_cam( 2 ), 0, -fx * pt_3d_cam( 0 ) / pt_3d_cam( 2 ), 0, fy / pt_3d_cam( 2 ), -fy * pt_3d_cam( 1 ) / pt_3d_cam( 2 );
            mat_d_pho_d_img = mat_photometric * mat_pre; //����ͼ��������ſɱȾ���
            //jacobian��vio_esikf��ͬ    ����IMU����ϵ�µ�ͼ����ſɱȾ���
            pt_hat = Sophus::SO3d::hat( ( R_imu.transpose() * ( pt_3d_w - t_imu ) ) );    //IMUϵ�µ�ͼ�� p(IMU)^ = R(IMU <-- W) * ( p(W) - p(IMU) )

            /// �۲����������ſɱȾ�����Ƶ�����r2live��¼E
            /// ����̬R���ſɱ�
            mat_A = state_iter.rot_ext_i2c.transpose() * pt_hat;//3 * 3�� R��C <-- IMU) * p(imu)^
            /// ��λ��t���ſɱ�
            mat_B = -state_iter.rot_ext_i2c.transpose() * ( R_imu.transpose() );// - R(C <--IMU) * R(IMU <-- W)
            /// �������ת�����ſɱ�
            mat_C = Sophus::SO3d::hat( pt_3d_cam );// p(C)^
            /// �����λ�������ſɱ�
            mat_D = -state_iter.rot_ext_i2c.transpose();// - R(C <-- IMU)
            meas_vec.block( pt_idx * 3, 0, 3, 1 ) = photometric_err_vec ; //�۲��������
            /// ��ʽ�󵼵Ľ��
            H_mat.block( pt_idx * 3, 0, 3, 3 ) = mat_d_pho_d_img * mat_A * huber_loss_scale; //�ֱ�����˶������ת��λ�Ƶ��ſɱȾ���
            H_mat.block( pt_idx * 3, 3, 3, 3 ) = mat_d_pho_d_img * mat_B * huber_loss_scale;
            if ( 1 )
            {
                if ( m_if_estimate_i2c_extrinsic )
                {
                    H_mat.block( pt_idx * 3, 18, 3, 3 ) = mat_d_pho_d_img * mat_C * huber_loss_scale;  //�����ת��λ�Ƶ��ſɱ�
                    H_mat.block( pt_idx * 3, 21, 3, 3 ) = mat_d_pho_d_img * mat_D * huber_loss_scale;
                }
            }
        }
        R_mat_inv_spa = R_mat_inv.sparseView(); //ϡ��
       
        last_avr_repro_err = acc_photometric_error; //��������һ֡��ƽ����ͶӰ���
        if ( avail_pt_count < minimum_iteration_pts )//�����ǰ���õĵ���(avail_pt_count) С����С��������(minimum_iteration_pts)�����˳�ѭ��
        {
            break;
        }
        // Esikf
        tim.tic( "Iter" ); //("Iter") ��ʼ��ʱ
        if ( if_esikf )  //ʹ��esikf
        {

            /// sparseViewʹ����������������ڴ�
            /// ����ļ�������K�Ĺ���
            H_mat_spa = H_mat.sparseView();//ϡ��
            Eigen::SparseMatrix< double > Hsub_T_temp_mat = H_mat_spa.transpose(); //ת�þ���
            vec_spa = ( state_iter - state_in ).sparseView();//ϵͳ״̬�仯
            H_T_H_spa = Hsub_T_temp_mat * R_mat_inv_spa * H_mat_spa; //������ϡ���ſɱȾ���ת���������ĳ˻�
            /// ��һ֡���ͼ�ص�����Խ���Ӿ�����Ȩ��Խ��
            Eigen::SparseMatrix< double > temp_inv_mat =
                ( H_T_H_spa.toDense() + ( state_in.cov * m_cam_measurement_weight ).inverse() ).inverse().sparseView();
            // ( H_T_H_spa.toDense() + ( state_in.cov ).inverse() ).inverse().sparseView();
            Eigen::SparseMatrix< double > Ht_R_inv = ( Hsub_T_temp_mat * R_mat_inv_spa );
            KH_spa = temp_inv_mat * Ht_R_inv * H_mat_spa; //������������� K
            solution = ( temp_inv_mat * ( Ht_R_inv * ( ( -1 * meas_vec.sparseView() ) ) ) - ( I_STATE_spa - KH_spa ) * vec_spa ).toDense(); //������״̬�ĸ�����
        }
        state_iter = state_iter + solution; //����״̬
#if DEBUG_PHOTOMETRIC
        cout << "Average photometric error: " <<  acc_photometric_error / total_pt_size << endl;
        cout << "Solved solution: "<< solution.transpose() << endl;
#else
        if ( ( acc_photometric_error / total_pt_size ) < 10 ) // By experience.���ÿ��Ĺ�������� 10�����˳�ѭ��
        {
            break;
        }
#endif
        if ( fabs( acc_photometric_error - last_repro_err ) < 0.01 )//��������жϵ�ǰ������ϴ����ı仯�Ƿ��С
        {
            break;
        }
        last_repro_err = acc_photometric_error;//��¼���ε��������
    }
    if ( if_esikf && avail_pt_count >= minimum_iteration_pts )//���ʹ�� ESIKF����չƽ����Ϣ�˲������ҿ��õĵ������ڻ������С��������������״̬Э�������
    {
        state_iter.cov = ( ( I_STATE_spa - KH_spa ) * state_iter.cov.sparseView() ).toDense();
    }
    state_iter.td_ext_i2c += state_iter.td_ext_i2c_delta;
    state_iter.td_ext_i2c_delta = 0;
    state_in = state_iter;//����״̬
    return true;
}

//���Ϸ����µ�ȫ��RGB��ͼ
void R3LIVE::service_pub_rgb_maps()
{
    int last_publish_map_idx = -3e8;  //����һ����ʼֵΪ�����Ĵ���������ʾ���һ�η����ĵ�ͼ����
    int sleep_time_aft_pub = 10; //����ÿ�η�����Ҫ�ȴ���ʱ��
    int number_of_pts_per_topic = 1000;  //ÿ������Ҫ�����ĵ��Ƶ����ĳ�ʼֵ
    if ( number_of_pts_per_topic < 0 )
    {
        return;
    }
    while ( 1 )  //���Ϸ����µ�ȫ��RGB��ͼ��
    {
        ros::spinOnce();  //��ѭ�����ڣ����������еĻص���������ѭ������ӦROS����Ϣ�Ļص�����
        std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );  //��ͣ���߳�10����
        pcl::PointCloud< pcl::PointXYZRGB > pc_rgb;  //����XZY��ɫ�ĵ�������
        sensor_msgs::PointCloud2            ros_pc_msg;  //����ROS�ĵ�������
        int pts_size = m_map_rgb_pts.m_rgb_pts_vec.size();  //��RGB���Ƶ�ͼ�л�ȡ    vector����ĵ�ͼrgb�㣨ָ����ʽ��������
        pc_rgb.resize( number_of_pts_per_topic ); //������С
        // for (int i = pts_size - 1; i > 0; i--)
        int pub_idx_size = 0;  //�����������
        int cur_topic_idx = 0;  //��ǰ���������
        if ( last_publish_map_idx == m_map_rgb_pts.m_last_updated_frame_idx )  //���������ֵ��ȣ�˵����ǰ֡�������Ѿ������������Ҫ�ٴδ���
        {
            continue;
        }
        last_publish_map_idx = m_map_rgb_pts.m_last_updated_frame_idx;  //��ֵ��������һ֡�����к�

        for ( int i = 0; i < pts_size; i++ )  //�����㴦��
        {
            if ( m_map_rgb_pts.m_rgb_pts_vec[ i ]->m_N_rgb < 1 ) //����õ�û��RGB������ѭ��
            {
                continue;
            }
            //��m_map_rgb_pts�е���Ϣ����pc_rgb
            pc_rgb.points[ pub_idx_size ].x = m_map_rgb_pts.m_rgb_pts_vec[ i ]->m_pos[ 0 ];
            pc_rgb.points[ pub_idx_size ].y = m_map_rgb_pts.m_rgb_pts_vec[ i ]->m_pos[ 1 ];
            pc_rgb.points[ pub_idx_size ].z = m_map_rgb_pts.m_rgb_pts_vec[ i ]->m_pos[ 2 ];
            pc_rgb.points[ pub_idx_size ].r = m_map_rgb_pts.m_rgb_pts_vec[ i ]->m_rgb[ 2 ];
            pc_rgb.points[ pub_idx_size ].g = m_map_rgb_pts.m_rgb_pts_vec[ i ]->m_rgb[ 1 ];
            pc_rgb.points[ pub_idx_size ].b = m_map_rgb_pts.m_rgb_pts_vec[ i ]->m_rgb[ 0 ];
            // pc_rgb.points[i].intensity = m_map_rgb_pts.m_rgb_pts_vec[i]->m_obs_dis;
            pub_idx_size++;  //����pc_rgb
            if ( pub_idx_size == number_of_pts_per_topic )  //�õ��ĵ�������ﵽ��1000����ʱ��
            {
                pub_idx_size = 0;  //������1000��������
                pcl::toROSMsg( pc_rgb, ros_pc_msg );  //���临�Ƹ�ros_pc_msg������1000���㷢��ȥ
                ros_pc_msg.header.frame_id = "world";   //  ����frame_id
                ros_pc_msg.header.stamp = ros::Time::now(); //����ʱ��
                if ( m_pub_rgb_render_pointcloud_ptr_vec[ cur_topic_idx ] == nullptr )  //�����һ��ָ���ǿյģ���û�б����
                {
                    m_pub_rgb_render_pointcloud_ptr_vec[ cur_topic_idx ] =     //����һ��ָ��ROS������������ָ�룬��ʵ����������洢��m_pub_rgb_render_pointcloud_ptr_vec
                        std::make_shared< ros::Publisher >(m_ros_node_handle.advertise< sensor_msgs::PointCloud2 >(
                            std::string( "/RGB_map_" ).append( std::to_string( cur_topic_idx ) ), 100 ) );
                    //����advertise�ᴴ������< sensor_msgs::PointCloud2 >���͵ķ�����������std::string( "/RGB_map_" ).append( std::to_string( cur_topic_idx ) )�Ƿ����Ļ�������
                }
                m_pub_rgb_render_pointcloud_ptr_vec[ cur_topic_idx ]->publish( ros_pc_msg );  //ʹ�øմ����ķ�����������������
                std::this_thread::sleep_for( std::chrono::microseconds( sleep_time_aft_pub ) );  //���̵߳ȴ�10����
                ros::spinOnce();  //
                cur_topic_idx++;
            }
        }
        //����󲻵�1000���㷢����ȥ����������ͬ��
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
            number_of_pts_per_topic *= 1.5;  //�������̫�࣬�����ÿ�η����ĵ���
            sleep_time_aft_pub *= 1.5;   //�����ʱ
        }
    }
}

//û���õ�
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

//����������뱣�沢��ʾ����
char R3LIVE::cv_keyboard_callback()
{
    char c = cv_wait_key( 1 ); //��ȡ����
    // return c;
    if ( c == 's' || c == 'S' )
    {
        scope_color( ANSI_COLOR_GREEN_BOLD );
        cout << "I capture the keyboard input!!!" << endl;
        m_mvs_recorder.export_to_mvs( m_map_rgb_pts );  //����ǰ�� RGB �������ݵ����� MVS �ļ���
        // m_map_rgb_pts.save_and_display_pointcloud( m_map_output_dir, std::string("/rgb_pt"), std::max(m_pub_pt_minimum_views, 5) );
        m_map_rgb_pts.save_and_display_pointcloud( m_map_output_dir, std::string("/rgb_pt"), m_pub_pt_minimum_views  );  //���沢��ʾ��������
    }
    return c;
}

// ANCHOR -  service_VIO_update ESIKF�Ż�����
void R3LIVE::service_VIO_update()
{
    // Init cv windows for debug
    //��ʼ����������
    op_track.set_intrinsic( g_cam_K, g_cam_dist * 0, cv::Size( m_vio_image_width / m_vio_scale_factor, m_vio_image_heigh / m_vio_scale_factor ) ); 
    op_track.m_maximum_vio_tracked_pts = m_maximum_vio_tracked_pts;//������׷�ٵ�����
    m_map_rgb_pts.m_minimum_depth_for_projection = m_tracker_minimum_depth; //��ȫ��RGB��ͼ,����ͶӰʱ����С��Ⱥ�������
    m_map_rgb_pts.m_maximum_depth_for_projection = m_tracker_maximum_depth;//��ȫ��RGB��ͼ,����ͶӰʱ����С��Ⱥ�������
    cv::imshow( "Control panel", generate_control_panel_img().clone() );  //��ʾ��Ϊ "Control panel" �Ĵ��ڣ����� generate_control_panel_img() �������ɵ�ͼ����䴰�ڡ�clone() ��������ȷ��ͼ��ĸ�������ʾ��
    Common_tools::Timer tim;  //����һ����ʱ������
    cv::Mat             img_get;
    while ( ros::ok() )   //һֱѭ����
    {
        cv_keyboard_callback();//��������S������ʵ����
        // ����Ƿ��յ���һ֡�����״�ɨ�裬û�յ�����ѭ���ȴ�
        while ( g_camera_lidar_queue.m_if_have_lidar_data == 0 )
        {
            ros::spinOnce();
            std::this_thread::sleep_for( std::chrono::milliseconds( THREAD_SLEEP_TIM ) );
            std::this_thread::yield();
            continue;
        }
        // ����Ƿ��յ�Ԥ������ͼ��,û����ѭ���ȴ�
        if ( m_queue_image_with_pose.size() == 0 )
        {
            ros::spinOnce();
            std::this_thread::sleep_for( std::chrono::milliseconds( THREAD_SLEEP_TIM ) );
            std::this_thread::yield();
            continue;
        }
        m_camera_data_mutex.lock();
        // ���m_queue_image_with_pose�����ڵĻ������ݴ���buffer,����ɵ�ͼ��֡���ڸ���,Ȼ��pop��
        while ( m_queue_image_with_pose.size() > m_maximum_image_buffer )
        {
            cout << ANSI_COLOR_BLUE_BOLD << "=== Pop image! current queue size = " << m_queue_image_with_pose.size() << " ===" << ANSI_COLOR_RESET
                 << endl;  //���m_queue_image_with_pose�Ĵ�С
            op_track.track_img( m_queue_image_with_pose.front(), -20 );//����������,��ǰ���һ��ͼ
            m_queue_image_with_pose.pop_front();//�����Ѿ��������ǰ���һ��ͼ
        }

        std::shared_ptr< Image_frame > img_pose = m_queue_image_with_pose.front();  //�õ�������ĵ�һ֡ͼ��
        double                             message_time = img_pose->m_timestamp;  //��ȡ��һ֡��ʱ���
        m_queue_image_with_pose.pop_front();  //������һ֡
        m_camera_data_mutex.unlock();
        g_camera_lidar_queue.m_last_visual_time = img_pose->m_timestamp + g_lio_state.td_ext_i2c;  //�ó����¹۲�ʱ�䣬��ǰ��ʱ��+ʱ�䲹��

        img_pose->set_frame_idx( g_camera_frame_idx );  //��ʼΪ0��Ϊÿһ��ͼ����
        tim.tic( "Frame" );  //����tim�е�tic���𶯼�ʱ

        if ( g_camera_frame_idx == 0 )  //�����һ֡��
        {
            std::vector< cv::Point2f >                pts_2d_vec;       // ѡ�еĵ�ͼ�㷴ͶӰ��ͼ���ϵ����� ���洢��ά��
            std::vector< std::shared_ptr< RGB_pts > > rgb_pts_vec;      // ѡ�еĵ�ͼ��
            // while ( ( m_map_rgb_pts.is_busy() ) || ( ( m_map_rgb_pts.m_rgb_pts_vec.size() <= 100 ) ) )
            
            // �ȴ���ͼ�е���ɫ�����100������ʱLIOģ������Global_map::append_points_to_global_map��������ȫ�ֵ�ͼ��ӵ�
            while ( ( ( m_map_rgb_pts.m_rgb_pts_vec.size() <= 100 ) ) )  
            {
                ros::spinOnce();
                std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
            }
            // ����IMU��λ�ˣ�����һ֡ͼ���λ�˺��ڲ�
            // ���ڵ�һ֡����������Ǿ�ֹ���˶�״̬��ֱ����ϵͳ״̬����λ�ˣ���Ϊ��û�ж�
            set_image_pose( img_pose, g_lio_state ); // For first frame pose, we suppose that the motion is static.
            // ����ȫ�ֵ�ͼģ�飬�������״̬��ѡ��һЩ�㣬����ά��ͶӰ��ͼ���ϣ�
            m_map_rgb_pts.selection_points_for_projection( img_pose, &rgb_pts_vec, &pts_2d_vec, m_track_windows_size / m_vio_scale_factor );
            // ��ʼ������ģ��
            op_track.init( img_pose, rgb_pts_vec, pts_2d_vec );  //��ʼ�������������룺һ֡ͼ�񣬸��ٵ㣬ͶӰ��
            g_camera_frame_idx++;   //ͼƬ����+1
            continue;
        }

        //����ͨ���Ա������lidar����ͷ��ʱ��������lidar��ʱ���������ȴ�lio�̰߳Ѹ���ļ��⴦���ꡣ
        //���Ž���Ԥ���ֲ��֣�ͨ��IMU���ͼ�����ʽ���λ�õ�Ԥ����
        g_camera_frame_idx++; //ͼƬ����+1
        tim.tic( "Wait" );   //����Wait��ʱ��
        // if_camera_can_process(): ���״������ݣ�����lidar buffer����ɵ��״�����ʱ�� > ��ǰ���ڴ����ͼ��ʱ������򷵻�true�������Դ���ͼ����
        while ( g_camera_lidar_queue.if_camera_can_process() == false )  //���ɴ���ͼ��
        {
            // ����������ѭ���ȴ������״�����
            ros::spinOnce();
            std::this_thread::sleep_for( std::chrono::milliseconds( THREAD_SLEEP_TIM ) );
            std::this_thread::yield();
            cv_keyboard_callback();  //�������뱣�����
        }
        g_cost_time_logger.record( tim, "Wait" );  //��¼ʱ�䣬���ȴ�ʱ��
        m_mutex_lio_process.lock();  //�����߳�
        tim.tic( "Frame" );    //Frame ��ʱ��ʼ
        tim.tic( "Track_img" );    //Track_img ��ʱ��ʼ
        StatesGroup state_out;  //���ֺ�õ���ϵͳ״̬
        //���գ�m_cam_measurement_weight ��������Ϊ���� 0.001 �� 0.01 ֮���һ��ֵ
        m_cam_measurement_weight = std::max( 0.001, std::min( 5.0 / m_number_of_new_visited_voxel, 0.01 ) ); 
        //��LIOԤ���ֵ���ǰ֡ʱ�̣� VIO ϵͳ�д��� IMU ���ݣ�ͨ��Ԥ���ּ������״̬
        if ( vio_preintegration( g_lio_state, state_out, img_pose->m_timestamp + g_lio_state.td_ext_i2c ) == false ) //����Ԥ����ʱ�����򿪣�������һ��ѭ����
        {
            m_mutex_lio_process.unlock();
            continue;
        }
        //������һLIOԤ���ֵĽ�����趨Ϊ��ǰ֡��Ӧ��λ��
        set_image_pose( img_pose, state_out );

        // ���� op_track.track_img( img_pose, -20 ) ���������㣬����һ֡ͼ���еĵ���ٵ���һ֡ͼ��
        //�����Ǹ���ͼ���ϵĵ㣬����һ֡�ĵ�ͨ�������� =����һ֡
        //��׷�ٵ��ĵ�ŵ�=��m_map_rgb_pts_in_last_frame_pos
        op_track.track_img( img_pose, -20 )


        //�����track_imgע���� LK_optical_flow_kernel::track_image����
        // �������٣�ͬʱȥ��outliers

        g_cost_time_logger.record( tim, "Track_img" ); //Track_img ��ʱ����
        // cout << "Track_img cost " << tim.toc( "Track_img" ) << endl;
        tim.tic( "Ransac" );  //Ransac��ʱ��ʼ
        set_image_pose( img_pose, state_out );//��󽫸��µ�img_pose��Ϊ�����������pose���ڲε�У׼  ����������

        // ANCHOR -  remove point using PnP.
        if ( op_track.remove_outlier_using_ransac_pnp( img_pose ) == 0 )  //ʹ�� RANSAC �� PnP �㷨ȥ���쳣�ĸ��ٵ�
        {
            cout << ANSI_COLOR_RED_BOLD << "****** Remove_outlier_using_ransac_pnp error*****" << ANSI_COLOR_RESET << endl;
        }
        g_cost_time_logger.record( tim, "Ransac" );  //Ransac��ʱ����
        tim.tic( "Vio_f2f" );   // Vio_f2f ��ʱ��ʼ
        bool res_esikf = true, res_photometric = true;
        wait_render_thread_finish();
        //�������׷�ٺ�Ԥ����ϵͳ״̬������ESIKF
        res_esikf = vio_esikf( state_out, op_track );   //������ͶӰ��� = �������Ż�ϵͳ״̬
        g_cost_time_logger.record( tim, "Vio_f2f" );   //��¼ "Vio_f2f"��ʱ����
        tim.tic( "Vio_f2m" );  //"Vio_f2m"��ʱ��ʼ
        //����֡����ͼ   ESIKF
        res_photometric = vio_photometric( state_out, op_track, img_pose );
        g_cost_time_logger.record( tim, "Vio_f2m" );  //��¼ "Vio_f2m"��ʱ���� 
        g_lio_state = state_out;//����ϵͳ״̬
        //���ɲ���ʾһ����ʽ�����Ǳ�壬�ṩʵʱ��ϵͳ״̬��Ϣ
        print_dash_board();
        //���Ż����λ�����趨��ǰ֡��λ��
        set_image_pose( img_pose, state_out );

        if ( 1 )
        {
            tim.tic( "Render" );//"Render"��ʼ��ʱ        ��Ⱦ
            // m_map_rgb_pts.render_pts_in_voxels(img_pose, m_last_added_rgb_pts_vec);
            if ( 1 ) // Using multiple threads for rendering
            {
                m_map_rgb_pts.m_if_get_all_pts_in_boxes_using_mp = 0;//ʹ�õ��̴߳�������
                // m_map_rgb_pts.render_pts_in_voxels_mp(img_pose, &m_map_rgb_pts.m_rgb_pts_in_recent_visited_voxels,
                // img_pose->m_timestamp);
                
                //�߳�5����Ҫ�����ǽ����Ƶ�ͼ��active��ͶӰ��ͼ���ϣ��ö�Ӧ�����ضԵ�ͼ��rgb�ľ�ֵ�ͷ����ñ�Ҷ˹�������и��¡�
                //���ú���ģ��std::make_shared���������ڴ�,   ����һ��ָ��m_render_thread
                //����ģ��Ĳ���Ϊstd::shared_future< void >
                //����ģ��� ��ʽ������ Ϊ 
                //m_thread_pool_ptr->commit_task(render_pts_in_voxels_mp, img_pose, & m_map_rgb_pts.m_voxels_recent_visited, img_pose->m_timestamp )
                m_render_thread = std::make_shared< std::shared_future< void > >( m_thread_pool_ptr->commit_task(
                    render_pts_in_voxels_mp, img_pose, &m_map_rgb_pts.m_voxels_recent_visited, img_pose->m_timestamp ) );
            }       //������Ⱦ��������
            else
            {
                m_map_rgb_pts.m_if_get_all_pts_in_boxes_using_mp = 0;
                // m_map_rgb_pts.render_pts_in_voxels( img_pose, m_map_rgb_pts.m_rgb_pts_in_recent_visited_voxels,
                // img_pose->m_timestamp );
            }
            m_map_rgb_pts.m_last_updated_frame_idx = img_pose->m_frame_idx;//����һ֡���и���m_last_updated_frame_idx
            g_cost_time_logger.record( tim, "Render" );  //������Ⱦ��ʱ

            tim.tic( "Mvs_record" );//��ʼ��ʱ��"Mvs_record"
            if ( m_if_record_mvs )  //?
            {
                // m_mvs_recorder.insert_image_and_pts( img_pose, m_map_rgb_pts.m_voxels_recent_visited );
                m_mvs_recorder.insert_image_and_pts( img_pose, m_map_rgb_pts.m_pts_last_hitted );//���ݼ�¼����¼��һ֡ͼ�����Ϣ��ȫ�ֵ�ͼ
            }
            g_cost_time_logger.record( tim, "Mvs_record" );  
        }
        // ANCHOR - render point cloud
        //��ӡ���ϵͳ����״̬��
        dump_lio_state_to_log( m_lio_state_fp );
        m_mutex_lio_process.unlock();//�߳̽���
        // cout << "Solve image pose cost " << tim.toc("Solve_pose") << endl;
        //������֡ͼ������ݣ�����ͼ��ͶӰ����
        m_map_rgb_pts.update_pose_for_projection( img_pose, -0.4 );
        op_track.update_and_append_track_pts( img_pose, m_map_rgb_pts, m_track_windows_size / m_vio_scale_factor, 1000000 );//������һ֡�ĸ��ٵ�
        g_cost_time_logger.record( tim, "Frame" );  //��¼������һ֡ͼ�������ĵ�ʱ��
        double frame_cost = tim.toc( "Frame" ); //��ȡ���Ϊ "Frame" ��ʱ��������ĵ�ʱ��
        g_image_vec.push_back( img_pose );//����ͼ��
        frame_cost_time_vec.push_back( frame_cost );//����ʱ��
        if ( g_image_vec.size() > 10 )  //ֻ������10֡�����ݣ�
        {
            g_image_vec.pop_front();
            frame_cost_time_vec.pop_front();
        }
        tim.tic( "Pub" );   //��ʱ��ʼ
        double display_cost_time = std::accumulate( frame_cost_time_vec.begin(), frame_cost_time_vec.end(), 0.0 ) / frame_cost_time_vec.size(); //��������֡����ʱ���ƽ��ֵ
        g_vio_frame_cost_time = display_cost_time;
        // publish_render_pts( m_pub_render_rgb_pts, m_map_rgb_pts );
        //�������λ�˺����·����Ϣ
        publish_camera_odom( img_pose, message_time );
        // publish_track_img( op_track.m_debug_track_img, display_cost_time );
        //��ͼ���ϻ��ƴ���ʱ�䲢��������
        publish_track_img( img_pose->m_raw_img, display_cost_time );

        if ( m_if_pub_raw_img ) //����ԭʼͼ��
        {
            publish_raw_img( img_pose->m_raw_img );
        }
        //ˢ����־
        if ( g_camera_lidar_queue.m_if_dump_log )
        {
            g_cost_time_logger.flush();
        }
        // cout << "Publish cost time " << tim.toc("Pub") << endl;
    }
}
