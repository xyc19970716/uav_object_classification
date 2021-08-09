# _*_ coding: utf-8 _*_

import sys
import os
import glob
import io
import cv2
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')  # 打印出中文字符
try:
    from osgeo import gdal
    from osgeo import ogr
    from osgeo import osr
except ImportError:
    import gdal
    import ogr
    import osr


def world2Pixel(geoMatrix, x, y):
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate
    """
    ulX = geoMatrix[0]
    ulY = geoMatrix[3]
    xDist = geoMatrix[1]
    pixel = int((x - ulX) / xDist)
    line = int((ulY - y) / xDist)
    return (pixel, line)
classes = ['大豆', '中稻', '晚稻', '甘蔗', '玉米', '番薯', '芝麻', '茶叶']
        #    1      2       3       4       5       6       7       8
print([classes[i] for i in range(len(classes))])

Mutli_Tem_Images_Path = r'D:\BaiduNetdiskDownload\drone\label'#'../../data/processed'
Mutli_Tem_Images_Paths = glob.glob(os.path.join(Mutli_Tem_Images_Path, "*.png"))

Crop_Field_Samples_Vector_Path = r'D:\BaiduNetdiskDownload\drone\result_test'
Crop_Field_Samples_Vector_Paths = glob.glob(os.path.join(Crop_Field_Samples_Vector_Path, "*.shp"))

print(Mutli_Tem_Images_Paths)
print(Crop_Field_Samples_Vector_Paths)

index = 0
for i, path in enumerate(Mutli_Tem_Images_Paths):

    srcImage = gdal.Open(path)
    geoTrans = srcImage.GetGeoTransform()
    img = srcImage.ReadAsArray()
    # print(img.shape)
    # print(geoTrans)
    # Create an OGR layer from a boundary shapefile
    shapef = ogr.Open(Crop_Field_Samples_Vector_Paths[i], 1)

    lyr = shapef.GetLayer()
    
    for j, feature in enumerate(lyr):
        geom =feature.GetGeometryRef()
        if geom == None:
            continue
        points = geom.Boundary().GetPoints()
        # print(feature.GetField('LUType'))
        
        tempLon = []
        tempLat = []
        tempPix = []
        if points is None:
            continue
        for p in points:
            tempLon.append(p[0])
            tempLat.append(p[1])
        tempLon = np.array(tempLon)
        tempLat = np.array(tempLat)
        lonMax = tempLon.max()
        lonMin = tempLon.min()
        latMax = tempLat.max()
        latMin = tempLat.min()
        # print(lonMax, latMax, lonMin, latMin)
        # compute bounding box
        x1, y1 = world2Pixel(geoTrans, lonMax, latMin)
        x2, y2 = world2Pixel(geoTrans, lonMin, latMax)
        # print(x1, y1, x2, y2)
        # switch x-y to y-x
        tempPatch = img[y2:y1 , x2:x1]
        # print(tempPatch.shape)
        for p in points:
            x, y = world2Pixel(geoTrans, p[0], p[1])
            tempPix.append((x - x2, y - y2))
        # print(tempPix)
        mask = np.zeros((y1 - y2, x1 - x2), dtype="uint8")
        cv2.polylines(mask, np.int32([tempPix]), 1, 1)
        cv2.fillPoly(mask, np.int32([tempPix]), 1)
        showImage = cv2.add(tempPatch, np.zeros(np.shape(tempPatch), dtype=np.uint8), mask=mask)
        cv2.imshow('show', showImage)
        #cv2.waitKey(0)
        # print(np.max(showImage))
        feature.SetField('LUType', int(np.max(showImage))) # 这可能是个问题，shape的精度
        lyr.SetFeature(feature)
        # print(j)
