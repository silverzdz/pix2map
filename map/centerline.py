import os
os.chdir("/home/zdz")
tracking_dataset_dir = 'argoverse-api/argoverse-tracking/sample/'
from argoverse.map_representation.map_api import ArgoverseMap
am = ArgoverseMap()
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader


if __name__ == '__main__':
    log_index = 0
    frame_index = 50
    idx = 50

    argoverse_loader = ArgoverseTrackingLoader(tracking_dataset_dir)
    log_id = argoverse_loader.log_list[log_index]
    argoverse_data = argoverse_loader[log_index]
    city_name = argoverse_data.city_name

    lidar_pts = argoverse_data.get_lidar(idx)
    x,y,_ = argoverse_data.get_pose(frame_index).translation
    print("pose:{},{}".format(x,y))

    nearest_centerline = am.get_nearest_centerline([x,y], 'PIT')

    print(nearest_centerline)