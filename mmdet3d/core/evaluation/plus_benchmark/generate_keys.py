from distutils.errors import DistutilsTemplateError
import json
import math
from traceback import print_tb
import numpy as np

'''
type code: https://github.com/PlusAI/common_protobuf/blob/5e6b69597640a3f84b72a6a4f02ad37c9ab27ec6/perception/obstacle_detection.proto#L93,
anno_doc: https://docs.google.com/document/d/1NqLQUIXfK5ZrrTsrCMEMOtf81YhavcAXh8qg1jj3AZs/edit#heading=h.bjwvdizea3a1

Type group:

VAN,SUV,CAR                                                     -> Car
TRUCK,BUS,HEAVY_EQUIPMENT,LIGHTTRUCK,TRAILER                    -> Truck
PEDESTRIAN,PERSON_SITTING                                       -> Pedestrian
MOTO,BICYCLE                                                    -> Cyclist
UNKNOWN_STATIONARY,BARRIER,CONE,TRAM,MOVABLE_SIGN,LICENSE_PLATE -> Dontcare

'''
type_hub = {
    0: "DONTCARE",
    1: "UNKNOWN",
    11: "CAR",
    12: "PEDESTRIAN",
    14: "BICYCLE",
    15: "VAN",
    16: "BUS",
    17: "TRUCK",
    18: "TRAM",
    19: "MOTO",
    20: "BARRIER",
    21: "CONE",
    23: "MOVABLE_SIGN",
    26: "LICENSE_PLATE",
    27: "SUV",
    28: "LIGHTTRUCK",
    29: "TRAILER",
    # deprecated types, don't use these. Use Attribute instead
    22: "HEAVY_EQUIPMENT",
    3: "UNKNOWN_STATIONARY",
    13: "PERSON_SITTING",
}

def parse_label(bag_label_file, 
                object_type_by_length_threshold=6, 
                time_skip=0.45):
    """extract anno_info from the label_json file of every bag
       and generate the downsampled label keys

    Args:
        bag_label_file (_type_): the label json file of every bag
        object_type_by_length_threshold (int, optional): for object name assign, Defaults to 6m.
        time_skip (float, optional): for keys downsample. Defaults to 0.45s.

    Returns:
        _type_: _description_
    """
    with open(bag_label_file, 'r') as f:
        label_data = json.load(f, encoding='utf-8')

    parsed_labels = {}
    for obj in label_data['objects']:
        for obj_box in obj['bounds']:
            box_timestamp = float(str(obj_box['timestamp']) + '.' + ('%09d' % obj_box['timestamp_nano']))
            loc_x = obj_box['position']['x']
            loc_y = obj_box['position']['y']
            loc_z = obj_box['position']['z']
            dim_l = obj['size']['x']
            dim_w = obj['size']['y']
            dim_h = obj['size']['z']

            if 'status_flags' in obj_box:
                has_3d_label = True if not ('has_3d_label' in obj_box['status_flags']) else (
                    obj_box['status_flags']['has_3d_label'])
            else:
                has_3d_label = True

            if obj_box['direction']['x'] == 0.:
                obj_box['direction']['x'] += 0.000001
            rot_z = math.atan2(obj_box['direction']['y'], obj_box['direction']['x'])
            parsed_label_data = {'box3d_lidar': np.array([loc_x, loc_y, loc_z, dim_l, dim_w, dim_h, rot_z])}
            if dim_l <= 1 and dim_w <= 1:
                type_name = 'Pedestrian'
            elif 1 < dim_l <= 2 and dim_w <= 1 and dim_h > 1:
                type_name = 'Cyclist'
            elif 3.5 <= dim_l < object_type_by_length_threshold:
                type_name = 'Car'
            elif dim_l >= object_type_by_length_threshold:
                type_name = 'Truck'
            else:
                # the type_name will be further confirmed when generate label pkls
                type_name = 'Car'
            
            # some old label info may don't have 'type'
            # and the 'type' will be grouped to Car,Truck,Pedestrian,Cyclist when generate label pkls
            if 'type' in obj:
                parsed_label_data['type'] = type_hub[obj['type']]
            else:
                parsed_label_data['type'] = None
            parsed_label_data['name'] = type_name
            parsed_label_data['has_3d_label'] = has_3d_label
            
            if box_timestamp not in parsed_labels:
                parsed_labels[box_timestamp] = []
            parsed_labels[box_timestamp].append(parsed_label_data)
    
    # downsample thr label keys
    time_order_keys = sorted(parsed_labels)
    output_keys = []
    last_time = 0
    for time_order_key in time_order_keys:
        if time_order_key - last_time >= time_skip:
            output_keys.append(time_order_key)
            last_time = time_order_key
    # drop the first and last frame
    output_keys = output_keys[1:-1]
    return parsed_labels, output_keys


def align_pcd_labels(pcd_ts, label_ts):
    th = 0.005 # 5ms
    delta_t = np.abs(pcd_ts - label_ts)
    idx = delta_t.argmin()
    min_delta_t = delta_t[idx]
    if min_delta_t > th:
        res = -1
    else:
        res = pcd_ts[idx]
    return res


def demo(bag_name):
    import rosbag
    import fastbag
    
    bag_file_path = f"/mnt/intel/jupyterhub/swc/datasets/raw_data/new_L4/CN_L4_origin_benchmark/bags/{bag_name}"
    label_file_path = f"/mnt/intel/jupyterhub/swc/datasets/raw_data/new_L4/CN_L4_origin_benchmark/labels/{bag_name}.json"
    # generate pcd timestamps
    pcd_ts = []
    start_timestamp = 0
    max_interval = 0.05 # s
    collected_topic = set()
    global lidar_topics
    
    with fastbag.Reader(bag_file_path) if fastbag.Reader.is_readable(bag_file_path) \
                                else rosbag.Bag(bag_file_path) as bag:
        all_topics = bag.get_type_and_topic_info().topics.keys()
        pc_topic = filter(lambda x: x in lidar_topics, all_topics)
        pc_topic = list(pc_topic)
        lidar_related_topic_num = len(pc_topic)
        for topic, msg, temp in bag.read_messages(topics=lidar_topics):
            # Collect lidar data continuously
            topic_timestamp = msg.header.stamp.to_sec()
            if abs(topic_timestamp - start_timestamp) >= max_interval:
                
                if lidar_related_topic_num - len(collected_topic) <=1 \
                    and len(collected_topic) > 0:
                    pcd_ts.append(start_timestamp)
                start_timestamp = topic_timestamp
                collected_topic = {topic}
            else:
                # Otherwise, keep collecting new message
                collected_topic.add(topic)
    pcd_ts = np.asarray(pcd_ts)
    # generate label timestamps
    parsed_labels, downsampled_keys = parse_label(label_file_path)
    # align
    aligned_pcd_ts_list = []
    for label_ts in downsampled_keys:
        aligned_pcd_ts = align_pcd_labels(pcd_ts, label_ts)
        if aligned_pcd_ts == -1:
            continue
        aligned_pcd_ts_list.append(aligned_pcd_ts)
    return downsampled_keys, aligned_pcd_ts_list


def save_keys():
    import os
    raw_label_dir = "/mnt/intel/jupyterhub/swc/datasets/raw_data/new_L4/CN_L4_origin_benchmark/labels/"
    raw_label_names = os.listdir(raw_label_dir)
    label_ts_dict = {}
    pcd_ts_dict = {}
    for label_name in raw_label_names:
        bag_name = label_name[:-5]
        print(f"bag: {bag_name}")
        _, downsampled_keys = parse_label(raw_label_dir+label_name)
        label_ts_dict[bag_name] = downsampled_keys
        # pcd_ts_dict[bag_name] = aligned_pcd_ts_list

    json_str = json.dumps(label_ts_dict, indent=4)
    label_ts_dict_name = '/mnt/intel/jupyterhub/swc/datasets/raw_data/new_L4/CN_L4_origin_benchmark/keyframes/label_ts_keys.json'
    with open(label_ts_dict_name, 'w') as json_file:
        json_file.write(json_str)
        
    # json_str = json.dumps(pcd_ts_dict, indent=4)
    # pcd_ts_dict_name = '/mnt/intel/jupyterhub/swc/datasets/raw_data/new_L4/CN_L4_origin_benchmark/keyframes/pcd_ts_keys.json'
    # with open(pcd_ts_dict_name, 'w') as json_file:
    #     json_file.write(json_str)

if __name__ == '__main__':
    # lidar_topics = ['/livox/lidar/horizon_front',
    #             '/livox/lidar/horizon_left',
    #             '/livox/lidar/horizon_right',
    #             ]
    # # 20210827T165426_j7-00010_24_1to21.db
    # # 20210827T165426_j7-00010_15_43to63.db
    # bag_name = '20210827T165426_j7-00010_24_1to21.db'

    # repeat_keys_dict= {}
    # repeat_pcd_dict = {}
    # for i in range(2):
    #     downsampled_keys,aligned_pcd_ts_list = demo(bag_name)
    #     repeat_keys_dict[i] = downsampled_keys
    #     repeat_pcd_dict[i] = aligned_pcd_ts_list
    #     print(f"label_size: {len(downsampled_keys)}, pcd_size: {len(aligned_pcd_ts_list)}")
    
    # print("label: ")
    # for first, second in zip(repeat_keys_dict[0], repeat_keys_dict[1]):
    #     print(f"{first==second} {first} == {second}")
    # print("pcd: ")
    # for first, second in zip(repeat_pcd_dict[0], repeat_pcd_dict[1]):
    #     print(f"{first==second} {first} == {second}")
    save_keys()