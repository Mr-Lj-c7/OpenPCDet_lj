import rospy
import rosbag
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import struct
 
def pointcloud2_to_pcd(point_cloud2_msg, filename):
    # 更新头部信息以包含intensity
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {point_cloud2_msg.width}
HEIGHT {point_cloud2_msg.height}
VIEWPOINT 0 0 0 1 0 0 0
POINTS {point_cloud2_msg.width * point_cloud2_msg.height}
DATA ascii
"""
 
    # 将点云数据（包括intensity）转换为ASCII格式并保存到PCD文件
    with open(filename, 'w') as f:
        f.write(header)
        for p in point_cloud2.read_points(point_cloud2_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
            f.write(f"{' '.join(str(value) for value in p)}\n")
 
def main():
    bag_file = '/home/dxy/SUSTechPOINTS/备份/rosbag/train.bag'
    topic = '/velodyne_points'
    output_directory = '/home/dxy/SUSTechPOINTS/备份/lidar/'
    frame_count = 0
 
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic]):
            if 1:
                filename = f"{output_directory}{t.to_nsec()}.pcd"
                pointcloud2_to_pcd(msg, filename)
                frame_count += 1
                print(f"Processed frame {frame_count}: Saved {filename}")
            else:
                print(f"Message is not of type PointCloud2: {type(msg)}")
 
    print(f"Total frames processed: {frame_count}")
 
if __name__ == "__main__":
    main()
