"""Single side NEXT only. Inputs/checkpoints must already exist."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    side=LaunchConfiguration('side')
    return LaunchDescription([
        DeclareLaunchArgument('side',choices=['left','right']),
        DeclareLaunchArgument('inference_config'),DeclareLaunchArgument('web_config'),
        Node(package='factr2_next',executable='next_infer',name=['next_inference_',side],
             parameters=[{'config_file':LaunchConfiguration('inference_config')}],output='screen'),
        Node(package='factr2_next',executable='next_visualize',name=['next_web_',side],
             parameters=[{'config_file':LaunchConfiguration('web_config')}],output='screen')])
