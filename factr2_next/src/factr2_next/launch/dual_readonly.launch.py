"""Independent left/right NEXT nodes, no bridge/controllers/command clients."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    single=str(Path(get_package_share_directory('factr2_next'))/'launch/readonly.launch.py')
    actions=[]
    for side in ('left','right'):
        for kind in ('inference','web'):actions.append(DeclareLaunchArgument(side+'_'+kind+'_config'))
        actions.append(IncludeLaunchDescription(PythonLaunchDescriptionSource(single),launch_arguments={
            'side':side,'inference_config':LaunchConfiguration(side+'_inference_config'),
            'web_config':LaunchConfiguration(side+'_web_config')}.items()))
    return LaunchDescription(actions)
