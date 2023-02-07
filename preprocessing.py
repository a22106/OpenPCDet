import pandas as pd
import pathlib as pl
import json
import numpy as np
from glob import glob
import os
import argparse
from tqdm import tqdm
from collections import Counter
import sys

CLASSES = ["Car", "Small_Car", "Light_Car", "SUV", "Van", "Small_Truck", "Medium_Truck", "Large_Truck", "Bus", "Mini_Bus", "Special_Vehicle","Two_Wheeler", "Kickboard", "Adult", "Kid"]

# check the python version. if the python version is under 3.11, print the warning message
if int(str(sys.version_info[0]) + str(sys.version_info[1])) < 311:
    # print("Warning: The python version is under 3.11. The program works faster if the python version is >= 3.11")
    print("경고: 파이썬 버전이 3.11보다 낮습니다. 파이썬 버전이 3.11 이상이면 프로그램이 더 빠르게 작동합니다.")

def get_json_data_3d(json_files_path: list, save_data_info: str = None, save_stat_to: str = None):
    # if there is a file on save_data_info or save_stat_to ask the user if they want to overwrite the file
    if os.path.isfile(save_data_info):
        answer_info = input(f"Warning: The file {save_data_info} already exists. Do you want to overwrite it? (y/n):")
    if os.path.isfile(save_stat_to):
        answer_stat = input(f"Warning: The file {save_stat_to} already exists. Do you want to overwrite it? (y/n): ")
        if answer_stat.lower() == "n":
            class_statistic = pd.read_csv(save_stat_to)
    
    if answer_info.lower() != "n":
        dp = pd.DataFrame(columns=["class", "position x", "position y", "position z", "rotation", "scale l", "scale w", "scale h", "file_name"])
        # print("Collecting 3d label data...")
        print("3D 레이블 데이터 수집 중...")
        for file in tqdm(json_files_path):
            with open(file, 'r') as f:
                json_data = json.load(f)
                for i in range(len(json_data)):
                    file_path = file.split("/")
                    # file_name is among the file_path if the attribute contains "_Suwon" or "_Pangyo" add this attribute and the last attribute with underbar "_" without extension
                    file_name = "_" + file_path[-2] + "_" + file_path[-1].split(".")[0] if "_Suwon" in file_path or "_Pangyo" in file_path else file_path[-1].split(".")[0]
                    row = [json_data[i]["obj_type"], 
                        json_data[i]["psr"]["position"]["x"], 
                        json_data[i]["psr"]["position"]["y"], 
                        json_data[i]["psr"]["position"]["z"], 
                        json_data[i]["psr"]["rotation"]['z'], 
                        json_data[i]["psr"]["scale"]["x"], 
                        json_data[i]["psr"]["scale"]["y"], 
                        json_data[i]["psr"]["scale"]["z"], 
                        file_name]
                    frame_data = pd.DataFrame([row], columns=["class", "position x", "position y", "position z", "rotation", "scale l", "scale w", "scale h", "file_name"])
                dp = pd.concat([dp, frame_data], axis=0)

        # make the dp's index to descending order
        dp = dp.reset_index(drop=True)
        dp.to_csv(save_data_info, index=False)
        # print(f"Saved the data info to '{save_data_info}'")
        print(f"데이터 정보를 '{save_data_info}'에 저장하였습니다.")
    else:
        dp_stat = pd.read_csv(save_data_info)
    
        def get_class_simple_statistic(dp_stat: pd.DataFrame):
            #클래스 종류 및 개수
            classes = Counter(dp_stat["class"])
            print(classes)
            # print f"classes: {classes}" for every 3 classes and goes to the next line
            print(f"classes: {', '.join([f'{key}: {value}' for key, value in classes.items()])}")
            
            # position x, y, z 범위
            # min, max of position x, y, z
            # print(f"min of position x: {min(dp_stat['position x']):.2f}, max of position x: {max(dp_stat['position x']):.2f}")
            # print(f"min of position y: {min(dp_stat['position y']):.2f}, max of position y: {max(dp_stat['position y']):.2f}")
            # print(f"min of position z: {min(dp_stat['position z']):.2f}, max of position z: {max(dp_stat['position z']):.2f}")

            # # rotation 범위
            # print(f"min of rotation: {min(dp_stat['rotation']):.2f}, max of rotation: {max(dp_stat['rotation']):.2f}")

            # # scale l, w, h 범위
            # print(f"min of scale l: {min(dp_stat['scale l']):.2f}, max of scale l: {max(dp_stat['scale l']):.2f}")
            # print(f"min of scale w: {min(dp_stat['scale w']):.2f}, max of scale w: {max(dp_stat['scale w']):.2f}")
            # print(f"min of scale h: {min(dp_stat['scale h']):.2f}, max of scale h: {max(dp_stat['scale h']):.2f}")
            return classes
    
    classes = get_class_simple_statistic(dp_stat)
            
    if answer_stat.lower() != "n":
        dp_stat = dp.copy()
        
        # remove the rows if class is not in the CLASSES
        dp_stat = dp_stat[dp_stat["class"].isin(CLASSES)]

        # remove the rows if scale l, w is over 20
        dp_stat = dp_stat[(dp_stat["scale l"] <= 20) & (dp_stat["scale w"] <= 20) & (dp_stat["scale h"] <= 20)]
        dp_stat = dp_stat.reset_index(drop=True)

        # if the value is under 0, change it to positive value
        # if the value is 0, remove it
        dp_stat["scale l"] = dp_stat["scale l"].apply(lambda x: abs(x) if x != 0 else np.nan)
        dp_stat["scale w"] = dp_stat["scale w"].apply(lambda x: abs(x) if x != 0 else np.nan)
        dp_stat["scale h"] = dp_stat["scale h"].apply(lambda x: abs(x) if x != 0 else np.nan)
        dp_stat = dp_stat.dropna(axis=0)
        dp_stat = dp_stat.reset_index(drop=True)


        get_class_simple_statistic(dp_stat)
        
        # average of scale l, w, h for each class. Make this to pandas dataframe
        # the row's index is class name, and the column's name is average of scale l, w, h
        class_scal_avg = pd.DataFrame(columns=["scale l", "scale w", "scale h"])
        for c in classes.keys():
            class_scal_avg.loc[c] = [dp_stat[dp_stat["class"] == c]["scale l"].mean(), dp_stat[dp_stat["class"] == c]["scale w"].mean(), dp_stat[dp_stat["class"] == c]["scale h"].mean()]
        
        # min & max of position x, y, z and scale l, w, h for each class. Make this to pandas dataframe
        # the row's index is class name, and the column's name is min & max of position x, y, z and scale l, w, h
        # each values is .3f
        class_statistic = pd.DataFrame(columns=["count", "position x min", "position x max", "position y min", "position y max", "position z min", "position z max", "scale l min", "scale l max", "scale w min", "scale w max", "scale h min", "scale h max", "scale l avg", "scale w avg", "scale h avg"])
        for c in classes.keys():
            class_statistic.loc[c] = [classes[c],
                                    f"{min(dp_stat[dp_stat['class'] == c]['position x']):.3f}", 
                                    f"{max(dp_stat[dp_stat['class'] == c]['position x']):.3f}", 
                                    f"{min(dp_stat[dp_stat['class'] == c]['position y']):.3f}", 
                                    f"{max(dp_stat[dp_stat['class'] == c]['position y']):.3f}", 
                                    f"{min(dp_stat[dp_stat['class'] == c]['position z']):.3f}", 
                                    f"{max(dp_stat[dp_stat['class'] == c]['position z']):.3f}", 
                                    f"{min(dp_stat[dp_stat['class'] == c]['scale l']):.3f}", 
                                    f"{max(dp_stat[dp_stat['class'] == c]['scale l']):.3f}", 
                                    f"{min(dp_stat[dp_stat['class'] == c]['scale w']):.3f}", 
                                    f"{max(dp_stat[dp_stat['class'] == c]['scale w']):.3f}", 
                                    f"{min(dp_stat[dp_stat['class'] == c]['scale h']):.3f}", 
                                    f"{max(dp_stat[dp_stat['class'] == c]['scale h']):.3f}", 
                                    f"{class_scal_avg.loc[c]['scale l']:.3f}", 
                                    f"{class_scal_avg.loc[c]['scale w']:.3f}", 
                                    f"{class_scal_avg.loc[c]['scale h']:.3f}"]
            
        # save stat to csv file
        class_statistic.to_csv(save_stat_to)
        print(f"save stat to '{save_stat_to}'")
    else:
        class_statistic = pd.read_csv(save_stat_to)
        get_class_simple_statistic(dp_stat)
        
    
    return dp_stat, class_statistic

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_path', '-f', type=str, help='path to the dataset folder')
    parser.add_argument('--to_path', '-t', type=str, help='path to the output folder', default=None)
    parser.add_argument('--all', '-a', action='store_true', help='if you want to convert all files(pcd, json) in the dataset folder')
    parser.add_argument('--pcd', '-p', action='store_true', help='if you want to convert pcd files')
    parser.add_argument('--label_3d', '-l3', action='store_true', help='if you want to convert label files')
    parser.add_argument('--img', '-i', action='store_true', help='if you want to convert image files')
    parser.add_argument('--label_2d', '-l2', action='store_true', help='if you want to convert label files')
    parser.add_argument('--stat_3d', '-s3', action='store_true', help='if you want to get the statistic of the dataset')
    parser.add_argument('--stat_output_3d', '-so3', type=str, help='path to the output file of the statistic', default="./stat_3d.csv")
    parser.add_argument('--stat_2d', '-s2', action='store_true', help='if you want to get the statistic of the dataset')
    parser.add_argument('--stat_output_2d', '-so2', type=str, help='path to the output file of the statistic', default="./stat_2d.csv")
    parser.add_argument('--info_output_3d', '-n3', type=str, help='path to the info file of the dataset', default="./info_3d.csv")
    parser.add_argument('--info_output_2d', '-n2', type=str, help='path to the info file of the dataset', default="./info_2d.csv")
    
    args = parser.parse_args()
    print(f"Dataset path: {args.from_path}")
    
    if args.all:
        args.pcd = True
        args.label_3d = True
        args.img = True
        args.label_2d = True

    if args.from_path is None:
        args.from_path = "./dataset"
    if args.to_path is None:
        args.to_path = "./output"

    # make the output folder
    if args.to_path is not None:
        os.makedirs(args.to_path, exist_ok=True)

    # get the file list of the dataset folder
    print("Collecting the 3d label file list of the dataset")
    json_paths = sorted([str(p) for p in pl.Path(args.from_path).rglob("*.json") if "calib" not in str(p) and "2d_label" not in str(p)])
    
    if args.stat_3d is True:
        # get the statistic of the dataset
        print("Collecting the statistic of the dataset")
        dp_info, class_statistic = get_json_data_3d(json_paths, save_data_info=args.info_output_3d, save_stat_to=args.stat_output_3d)
        print("Dataset statistic:")
        print(dp_info)
        print("Class statistic:")
        print(class_statistic)