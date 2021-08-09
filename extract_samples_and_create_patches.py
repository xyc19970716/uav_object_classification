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

Mutli_Tem_Images_Path = r'D:\BaiduNetdiskDownload\drone\image'#'../../data/processed'
Mutli_Tem_Images_Paths = glob.glob(os.path.join(Mutli_Tem_Images_Path, "*.png"))

Crop_Field_Samples_Vector_Path = r'D:\BaiduNetdiskDownload\drone\temp'
Crop_Field_Samples_Vector_Paths = glob.glob(os.path.join(Crop_Field_Samples_Vector_Path, "*.shp"))

DataSet_Main_Dir_Path = 'D:\BaiduNetdiskDownload\drone'

Crop_Field_DataSet_Name = DataSet_Main_Dir_Path + '/data'

if not os.path.exists(Crop_Field_DataSet_Name):
    os.makedirs(Crop_Field_DataSet_Name)



# print(Mutli_Tem_Images_Paths)

index = 0
for i, path in enumerate(Mutli_Tem_Images_Paths):

    srcImage = gdal.Open(path)
    geoTrans = srcImage.GetGeoTransform()
    img = srcImage.ReadAsArray()
    pixel_size = img.shape[1]
    print(path)
    # print(img.shape)
    # print(geoTrans)
    # Create an OGR layer from a boundary shapefile
    shapef = ogr.Open(Crop_Field_Samples_Vector_Paths[i])

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
        if x1 <0:
            x1 = 0
        if x2 <0:
            x2 = 0
        if y1 <0:
            y1 = 0
        if y2 <0:
            y2 = 0
        if x1 > pixel_size:
            x1 = pixel_size
        if x2 > pixel_size:
            x2 = pixel_size
        if y1 > pixel_size:
            y1 = pixel_size
        if y2 > pixel_size:
            y2 = pixel_size
        # switch x-y to y-x
        print(x1,y1,x2,y2)
        tempPatch = img[:,y2:y1 , x2:x1]
        print(tempPatch.shape)
        for p in points:
            x, y = world2Pixel(geoTrans, p[0], p[1])
            tempPix.append((x - x2, y - y2))
        # print(tempPix)
        tempPatch = np.transpose(tempPatch,(1, 2, 0))
        mask = np.zeros((y1 - y2, x1 - x2), dtype="uint8")
        print(mask.shape)
        cv2.polylines(mask, np.int32([tempPix]), 1, 1)
        cv2.fillPoly(mask, np.int32([tempPix]), 1)
        showImage = cv2.add(tempPatch, np.zeros(np.shape(tempPatch), dtype=np.uint8), mask=mask)
        # cv2.imshow('show', showImage)
        #cv2.waitKey(0)


        # save a patch
        h, w ,c = tempPatch.shape

        Patch_Path = Crop_Field_DataSet_Name + '/{}'.format(feature.GetField('LUType'))

        if not os.path.exists(Patch_Path):
            os.makedirs(Patch_Path)

        GTiff_drv = gdal.GetDriverByName('GTiff')
        GTiff_raster = GTiff_drv.Create(
            Patch_Path + '/{}.tif'.format(index),
            w,
            h,
            c,
            gdal.GDT_Byte
        )
        index = index + 1
        if(GTiff_raster != None):    
            GTiff_raster.SetProjection(srcImage.GetProjection())
            tempTrans = list(geoTrans)
            tempTrans[0] = lonMin
            tempTrans[3] = latMax
            GTiff_raster.SetGeoTransform(tuple(tempTrans))
        showImage = np.transpose(showImage, (2, 0, 1))
        for k in range(3):
            GTiff_raster.GetRasterBand(k+1).WriteArray(showImage[k])
        print(j)
