import os
import numpy as np
 
def read_pcd(filepath):
    lidar = []
    header_passed = False
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('DATA'):
                header_passed = True
                continue
            if header_passed:
                linestr = line.split()
                if len(linestr) == 3:
                    linestr_convert = list(map(float, linestr)) + [1.0]
                    linestr_convert[2] += 0
                    print("!!!!!!!!!!!!!!!!!!!!!!ERROR")
                    lidar.append(linestr_convert)
                elif len(linestr) == 4:
                    linestr_convert = list(map(float, linestr))
                    linestr_convert[2] += 0
                    lidar.append(linestr_convert)
    return np.array(lidar)
 
def pcd2bin(pcdfolder, binfolder, start_idx, end_idx):
    ori_path = pcdfolder
    des_path = binfolder
    if not os.path.exists(des_path):
        os.makedirs(des_path)
 
    for idx in range(start_idx, end_idx + 1):
        filename = f"{idx:06d}"  # 格式化文件名，确保是六位数字，例如000001
        velodyne_file = os.path.join(ori_path, filename + '.pcd')
        if os.path.exists(velodyne_file):  # 确保文件存在
            pl = read_pcd(velodyne_file)
            pl = pl.reshape(-1, 4).astype(np.float32)
            velodyne_file_new = os.path.join(des_path, filename + '.bin')
            pl.tofile(velodyne_file_new)
        else:
            print(f"File not found: {velodyne_file}")
 
if __name__ == "__main__":
    pcdfolder = "/home/dxy/SUSTechPOINTS/data/备份/lidar_copy"
    binfolder = "/home/dxy/SUSTechPOINTS/data/备份/lidar_bin"
    
    # 可以在这里设置开始和结束的帧
    start_frame = 1
    end_frame = 35
    
    pcd2bin(pcdfolder, binfolder, start_idx=start_frame, end_idx=end_frame)
