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

from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

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

Mutli_Tem_Images_Path = r'G:\classification\img'#'../../data/processed'
Mutli_Tem_Images_Paths = glob.glob(os.path.join(Mutli_Tem_Images_Path, "*.JPG"))

Crop_Field_Samples_Vector_Path = r'G:\classification\test_shp'
Crop_Field_Samples_Vector_Paths = glob.glob(os.path.join(Crop_Field_Samples_Vector_Path, "*.shp"))


print(Mutli_Tem_Images_Paths)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

model = models.__dict__["resnet18"]()
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load('./model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()


index = 0
for i, path in enumerate(Mutli_Tem_Images_Paths):

    srcImage = gdal.Open(path)
    geoTrans = srcImage.GetGeoTransform()
    img = srcImage.ReadAsArray()
    print(img.shape)
    print(geoTrans)
    # Create an OGR layer from a boundary shapefile 
    shapef = ogr.Open(Crop_Field_Samples_Vector_Paths[i], 1) # 0 is read-only 1 is read-write

    lyr = shapef.GetLayer()
    
    for j, feature in enumerate(lyr):
        geom =feature.GetGeometryRef()
        if geom == None:
            continue
        points = geom.Boundary().GetPoints()
        
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
        print(lonMax, latMax, lonMin, latMin)
        # compute bounding box
        x1, y1 = world2Pixel(geoTrans, lonMax, latMin)
        x2, y2 = world2Pixel(geoTrans, lonMin, latMax)
        print(x1, y1, x2, y2)
        # switch x-y to y-x
        tempPatch = img[:,y2:y1 , x2:x1]
        print(tempPatch.shape)
        for p in points:
            x, y = world2Pixel(geoTrans, p[0], p[1])
            tempPix.append((x - x2, y - y2))
        print(tempPix)
        tempPatch = np.transpose(tempPatch,(1, 2, 0))
        mask = np.zeros((y1 - y2, x1 - x2), dtype="uint8")
        cv2.polylines(mask, np.int32([tempPix]), 1, 1)
        cv2.fillPoly(mask, np.int32([tempPix]), 1)
        showImage = cv2.add(tempPatch, np.zeros(np.shape(tempPatch), dtype=np.uint8), mask=mask)
        # cv2.imshow('show', showImage)
        #cv2.waitKey(0)
        showImage = Image.fromarray(showImage)# Image.fromarray(cv2.cvtColor(showImage,cv2.COLOR_BGR2RGB)) 
        input_tensor = trans(showImage)
        input_tensor = input_tensor.unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            pred_y = torch.max(output,1)[1]
            pred_y_data = pred_y.detach().cpu().numpy()
        feature.SetField('LUType', int(pred_y_data))
        lyr.SetFeature(feature)
        print(j, int(pred_y_data))
