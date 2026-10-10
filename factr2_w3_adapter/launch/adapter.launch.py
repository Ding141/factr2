"""Select an explicit left/right mock/real ROS parameter profile."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def build(context):
    side = LaunchConfiguration('side').perform(context)
    profile = LaunchConfiguration('profile').perform(context)
    publish_hz = float(LaunchConfiguration('publish_hz').perform(context))
    if publish_hz not in (50.,100.):
        raise ValueError('publish_hz must be 50 or 100')
    if side not in ('left', 'right') or profile not in ('mock', 'real'):
        raise ValueError('Choose explicit side=left/right and profile=mock/real')
    path = Path(get_package_share_directory('factr2_w3_adapter')) / 'config' / f'{side}_{profile}.yaml'
    return [Node(package='factr2_w3_adapter', executable='w3_next_adapter',
                 name=f'w3_next_adapter_{side}', parameters=[str(path),{'publish_hz':publish_hz}], output='screen')]


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument('side'),
                              DeclareLaunchArgument('profile', default_value='real'),
                              DeclareLaunchArgument('publish_hz', default_value='50'),
                              OpaqueFunction(function=build)])
