import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('icet'),
        'config',
        'icet_config.yaml'
    )

    return LaunchDescription([
        Node(
            package='icet',
            executable='odometry_node',
            name='odometry_node',
            output='screen',
            parameters=[config],
        ),
    ])
