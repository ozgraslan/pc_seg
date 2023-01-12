#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2, PointField 
from std_msgs.msg import Header

import json
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

config_dict = {
    "serial": "",
    "color_format": "RS2_FORMAT_RGB8",
    "color_resolution": "0,720",
    "depth_format": "RS2_FORMAT_Z16",
    "depth_resolution": "0,720",
    "fps": "30",
    "visual_preset": ""
}
intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
intrinsic.width = 1280
intrinsic.height = 720

# function taken from https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0
def point_cloud(points, parent_frame):
    """ Creates a point cloud message.
    Args:
        points: Nx7 array of xyz positions (m) and rgba colors (0..1)
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    """
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()

    fields = [PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyzrgba')]

    header = Header(frame_id=parent_frame, stamp=rospy.Time.now())

    return PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 7),
        row_step=(itemsize * 7 * points.shape[0]),
        data=data
    )

def main():
    pub = rospy.Publisher('segmented_pc', PointCloud2, queue_size=1)
    rospy.init_node('pc_segmentation_node', anonymous=True)
    rate = rospy.Rate(30) # 30hz
    config_json = json.dumps(config_dict)

    rs_cfg = o3d.t.io.RealSenseSensorConfig(json.loads(config_json))
    rs = o3d.t.io.RealSenseSensor()
    rs.init_sensor(rs_cfg, 0)
    rs.start_capture(False)  # true: start recording with capture
    while not rospy.is_shutdown():
        im_rgbd = rs.capture_frame(True, True)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(im_rgbd.color.to_legacy(), im_rgbd.depth.to_legacy(), depth_trunc=1.1, convert_rgb_to_intensity=False)
        pcd_new = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic, np.identity(4))
        pcd_new = pcd_new.voxel_down_sample(voxel_size=0.012)
        pcd_new.transform([[1, 0, 0, 0],  [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 1, 1]])
        _, inliers = pcd_new.segment_plane(distance_threshold=0.015,
                                                            ransac_n=3,
                                                            num_iterations=1000)
        outlier_cloud = pcd_new.select_by_index(inliers, invert=True)
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(outlier_cloud.cluster_dbscan(eps=0.04, min_points=40, print_progress=False))
        max_label = labels.max()
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))[labels >= 0]
        points = np.asarray(outlier_cloud.points)[labels >= 0]
        cloud = np.concatenate([points, colors], axis=-1)
        pub.publish(point_cloud(cloud, "map"))
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass