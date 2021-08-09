# _*_ coding: utf-8 _*_

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
try:
    from osgeo import gdal
    from osgeo import ogr
    from osgeo import osr
except ImportError:
    import gdal
    import ogr
    import osr

def pixel2World(geoMatrix, x, y):
    xgeo = geoMatrix[0] + x*geoMatrix[1] + y*geoMatrix[2]
    ygeo = geoMatrix[3] + x*geoMatrix[4] + y*geoMatrix[5]
    return xgeo, ygeo

gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")  # 为了支持中文路径
gdal.SetConfigOption("SHAPE_ENCODING", "CP936")  # 为了使属性表字段支持中文

img_path = r'D:\BaiduNetdiskDownload\drone\image'
shp_path = r"D:\BaiduNetdiskDownload\drone\temp"
json_label_path = r'D:\BaiduNetdiskDownload\20210416_浙工大标定结果\B0001_Drone_ZJUT_china\1_0.05m_drone_png.json'

classes = {'草地': 1, '道路': 2, '耕地': 3, '建筑物': 4, '林地': 5, '裸地': 6, '水体': 7, '其他人工用地': 8}

data = pd.read_json(json_label_path)
info = data.iloc[3]
info = info.iloc[0][0]['VideoInfo']
data = info['mapFrameInfos']

for i in tqdm(range(len(data))):
    info = data[i]
    img_name = info['key']['FrameNum']
    print(img_name)
    if not os.path.exists(os.path.join(img_path, img_name)):
        continue
    img_data = gdal.Open(os.path.join(img_path, img_name))
    img_data_proj = img_data.GetProjection()
    img_data_geo = img_data.GetGeoTransform()
    img_data_geo = list(img_data_geo)
    img_data_geo[1] = -img_data_geo[1]
    img_data_geo = tuple(img_data_geo)
    print(img_data_geo)
    shape_name = img_name.split('.')[0] + ".shp"
    print(shape_name)
    labels = info['value']['mapTargets']

    spatial=osr.SpatialReference(img_data_proj)
    strVectorFile = os.path.join(shp_path, shape_name)  # 定义写入路径及文件名
    ogr.RegisterAll()  # 注册所有的驱动
    strDriverName = "ESRI Shapefile"  # 创建数据，这里创建ESRI的shp文件
    oDriver = ogr.GetDriverByName(strDriverName)
    if oDriver == None:
        print("%s 驱动不可用！\n", strDriverName)

    oDS = oDriver.CreateDataSource(strVectorFile)  # 创建数据源
    if oDS == None:
        print("创建文件【%s】失败！", strVectorFile)

    oLayer = oDS.CreateLayer("TestPolygon", srs = spatial, geom_type = ogr.wkbPolygon)
    if oLayer == None:
        print("图层创建失败！\n")

    '''下面添加矢量数据，属性表数据、矢量数据坐标'''
    oFieldID = ogr.FieldDefn("FieldID", ogr.OFTInteger)  # 创建一个叫FieldID的整型属性
    oLayer.CreateField(oFieldID, 1)   
    oFieldID = ogr.FieldDefn("LuType", ogr.OFTInteger)  # 创建一个叫LuType的整型属性
    oLayer.CreateField(oFieldID, 1)  
    oDefn = oLayer.GetLayerDefn()  # 定义要素
    #gardens = ogr.Geometry(ogr.wkbMultiPolygon)  # 定义总的多边形集
    
    for j in range(len(labels)):
        label = labels[j]['value']
        land_type = label['PropertyPages'][0]['PropertyPageDescript']
        ploy = []
        points = label['Vertex']
        for k in range(len(points)):
            temp_x = int(points[k]['fX'] * 1000)
            temp_y = int(points[k]['fY'] * 1000)
            ploy.append([temp_x, temp_y])
        # ploy = [np.array(ploy)]
        # cv2.fillPoly(mask, ploy, classes[land_type])
           
        # print(ploy)

        box1 = ogr.Geometry(ogr.wkbLinearRing)
        for point in ploy:
            x_col = float(point[1])
            y_row = float(point[0])
            X_col, Y_row = pixel2World(img_data_geo, x_col, y_row)
            box1.AddPoint(Y_row, X_col)
        oFeatureTriangle = ogr.Feature(oDefn)
        oFeatureTriangle.SetField(0, j)
        oFeatureTriangle.SetField(1, classes[land_type])
        garden1 = ogr.Geometry(ogr.wkbPolygon)  # 每次重新定义单多变形
        box1.CloseRings()
        garden1.AddGeometry(box1)  # 将轮廓坐标放在单多边形中
        
        geomTriangle = ogr.CreateGeometryFromWkt(str(garden1))  # 将封闭后的多边形集添加到属性表

        oFeatureTriangle.SetGeometry(geomTriangle)
        oLayer.CreateFeature(oFeatureTriangle)


    oDS.Destroy()


