from setuptools import setup
import os
from glob import glob

package_name = 'yolo_in_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'yolo_in_ros'), glob('yolo_in_ros/*.py')),
        # (os.path.join('lib', 'python3.6', 'site-packages', package_name), glob('yolo_in_ros/*.txt')),
        # (os.path.join('lib', 'python3.6', 'site-packages', package_name), glob('yolo_in_ros/*.cfg')),
        # (os.path.join('lib', 'python3.6', 'site-packages', package_name), glob('yolo_in_ros/*.weights')),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        (os.path.join('share/', package_name, 'config'), glob('config/*.txt')),
        (os.path.join('share/', package_name, 'config'), glob('config/*.cfg')),
        (os.path.join('share/', package_name, 'config'), glob('config/*.weights')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='berk',
    maintainer_email='burhanberk.kurhan@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        "yolo_in_ros=yolo_in_ros.yolo_in_ros:main"],
    },
)
