# -*- coding: utf-8 -*-
"""
Created on 2016/8/1
Asymptotic Radiative Transfer Model (ART)
author: shaodonghang
"""
import time
import numpy             as     np
import datetime          as     datetime
import matplotlib.pylab  as     plt
import netCDF4           as     nc
import sys
import math
# import gdal
import ogr
from   gdalconst         import *
from   osr               import SpatialReference
from   osgeo             import gdal
from   osgeo.gdalconst   import *
# import libtiff           as     tif
# from   libtiff           import TIFF     #read tiff data
from   datetime          import timedelta
from   scipy.interpolate import griddata
from   netCDF4           import Dataset
# from   matplotlib.pyplot import savefig  
#---------------------README----------------
# Particle-Size Distribution: silt(SI),sand(SA),clay(CL),Soil Organic Matter(SOM), % weight, g/100g
# Bulk Density (BD),	g/cm3
# Gravel content(GRAV), % volume
############### old region of NC data #########################
#nc_path = 'F:/BaiduYunDownload/'
#outpath = 'F:/UHB_soil_result/'
###############################################################
#read soil and estimate data
driver  = gdal.GetDriverByName('HFA')
driver.Register()
#gdal.AllRegister() 单独注册某一类型的数据驱动，这样的话可以读也可以写，可以新建数据集
gdal.AllRegister()
fn      ='E:\\albedo\\reflectance\\A2014001_reflectance_Grid_2D.img'
ds      = gdal.Open(fn, GA_ReadOnly)
# print ds.GetDriver().ShortName
if ds is None:
   print 'Could not open ' + fn
   sys.exit(1)
#read the X and Y direction pixels of raster dataset
cols    = ds.RasterXSize
rows    = ds.RasterYSize
#read snow reflectance
band1   = ds.GetRasterBand(1)
band2   = ds.GetRasterBand(2)
band3   = ds.GetRasterBand(3)
band4   = ds.GetRasterBand(4)
band5   = ds.GetRasterBand(5)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
band6   = ds.GetRasterBand(6)
band7   = ds.GetRasterBand(7)

reflectance1  = band1.ReadAsArray(0, 0, cols, rows)
reflectance2  = band2.ReadAsArray(0, 0, cols, rows)
reflectance3  = band3.ReadAsArray(0, 0, cols, rows)
reflectance4  = band4.ReadAsArray(0, 0, cols, rows)
reflectance5  = band5.ReadAsArray(0, 0, cols, rows)
reflectance6  = band6.ReadAsArray(0, 0, cols, rows)
reflectance7  = band7.ReadAsArray(0, 0, cols, rows)

reflectance1  = np.array(reflectance1)
reflectance2  = np.array(reflectance2)
reflectance3  = np.array(reflectance3)
reflectance4  = np.array(reflectance4)
reflectance5  = np.array(reflectance5)
reflectance6  = np.array(reflectance6)
reflectance7  = np.array(reflectance7)

#read angle data
fn_angle  ='E:\\albedo\\angle\\A2014001_angle_Grid_2D.tif'
ds_angle  = gdal.Open(fn_angle, GA_ReadOnly)
#angle1=solar zenith angle, angle2=view zenith angle, angle3=relative azimuth angle
band8     = ds_angle.GetRasterBand(1)
band9     = ds_angle.GetRasterBand(2)
band10    = ds_angle.GetRasterBand(3)

angle1    = band8.ReadAsArray(0, 0, cols, rows)
angle2    = band9.ReadAsArray(0, 0, cols, rows)
angle3    = band10.ReadAsArray(0, 0, cols, rows)

angle1    = np.array(angle1)
angle2    = np.array(angle2)
angle3    = np.array(angle3)

plt.imshow(angle1,origin='lower left')
plt.colorbar()
plt.savefig('D:\\angle1.jpg')  #save the figure
plt.show()
plt.clf()
#print 'data',np.shape(data)
noDataValue   = band1.GetNoDataValue()
projection    = ds.GetProjection()
geotransform  = ds.GetGeoTransform()
print 'cols',cols
print 'rows',rows
# cols         =np.array(cols)
# rows         =np.array(rows)
# xList=[]
# yList=[]
x0 =geotransform[0]  #/* top left x 左上角x坐标*/
x1 =geotransform[1]  #/* w--e pixel resolution 东西方向上的像素分辨率*/
x2 =geotransform[2]  #/* rotation, 0 if image is "north up" 如果北边朝上，地图的旋转角度*/
x3 =geotransform[3]  #/* top left y 左上角y坐标*/
x4 =geotransform[4]  #/* rotation, 0 if image is "north up" 如果北边朝上，地图的旋转角度*/
x5 =geotransform[5]  #/* n-s pixel resolution 南北方向上的像素分辨率*/
print 'x0=',x0
print 'x1=',x1
print 'x2=',x2
print 'x3=',x3
print 'x4=',x4
print 'x5=',x5
ncols      =144
nrows      =84
xllcorner  =100.041666667
yllcorner  =38.3958333333
Xresolution=0.0083333333
Yresolution=-0.0083333333
left       =xllcorner;          right=left+Xresolution*ncols
bottom     =yllcorner;          top  =bottom+Yresolution*nrows
# dimensions and its boundary coordination
xmin  = left+0.5*Xresolution;   xmax = right-0.5*Xresolution
ymin  = bottom+0.5*Yresolution; ymax = top-0.5*Yresolution
xi    = np.arange(xmin,xmax,Xresolution)
yi    = np.arange(ymin,ymax,Yresolution)

print 'yi=',np.shape(xi)
print 'xi=',np.shape(yi)

xi,yi = np.meshgrid(xi, yi)
point =np.array([yi.flatten(),xi.flatten()]).T
print 'point',point
# for col in range(0,cols):
    # xTempList=[]
    # yTempList=[]
    # for row in range(0,rows):
    #     x    =geotransform[0]+col*geotransform[1]+row*geotransform[2]
    #     y    =geotransform[3]+col*geotransform[5]
    #     x    =np.array(x)
    #     y    =np.array(y)
        # x,y  =np.meshgrid(x,y)
        # xTempList.append(x)
        # yTempList.append(y)
        # point=np.array([y.flatten(),x.flatten()]).T
    # xList.append(xTempList)
    # yList.append(yTempList)
# xList,yList=np.meshgrid(xList,yList)
# point=np.array([xList.flatten(),yList.flatten()]).T
# x,y  =np.meshgrid(x,y)
# point=np.array([y.flatten(),x.flatten()]).T
# print 'xList',xList
# print 'yList',yList
# lat_nc        = ncfile.variables['lat']
# lon_nc        = ncfile.variables['lon']
# lat_nc        = np.array(lat_nc[2000:3000])
# lon_nc        = np.array(lon_nc[3000:4000])
# lon_nc,lat_nc = np.meshgrid(lon_nc,lat_nc)
# points        = np.array([lat_nc.flatten(),lon_nc.flatten()]).T
#for i in range(0, rows, yBSize): 
#if i + yBSize < rows: 
#        numRows = yBSize 
#else: 
#        numRows = rows – i 
#    for j in range(0, cols, xBSize): 
#        if j + xBSize < cols: 
#            numCols = xBSize 
#        else: 
#            numCols = colsnumCols = cols – j
#        data = band.ReadAsArray(j, i, numCols, numRows)

def Albedo_ART(angle1, angle2, angle3, reflectance):

    A  =  1.247
    B  =  1.186
    C  =  5.157
    u0 =  np.cos(angle1*0.01745)   #pi/180=0.01745
    u  =  np.cos(angle2*0.01745)
 
    s0 =  np.sin(angle1*0.01745)
    s  =  np.sin(angle2*0.01745)
    s1 =  np.cos(angle3*0.01745)

    angle4 = np.arccos(-u*u0+s*s0*s1)

    P  = 11.1*np.exp(-0.087*angle4*180.0/3.141592653589793)+1.1*np.exp(-0.014*angle4*180.00/3.141592653589793)
    R0 = ((A+B*(u+u0)+C*u*u0+P)*1.0)/(4.0*(u+u0))

    uu0= 3*(1+2*u0)/7.00

    uu = 3*(1+2*u)/7.00
    print 'uu0=',uu0
    print 'uu=',uu
    print 'R0=',R0
    Rs = pow((reflectance/R0),(R0/(uu0*uu)))  #Rs为积雪光谱白空反照率
    Rp = pow(Rs,uu0)                   #Rp为积雪光谱黑空反照率
    f  = 0.3                            #f为天空散射光因子
    result = (1-f)*Rp+f*Rs             #result位积雪光谱反照率，f为天空散射光因子，取地表反照率产品MOD43中的0.3
    print 'result=', result

    return result
#return Albedo_ART()

# call function Albedo_ART
Albedo_b1 = Albedo_ART(angle1,angle2,angle3,reflectance1)
Albedo_b2 = Albedo_ART(angle1,angle2,angle3,reflectance2)
Albedo_b3 = Albedo_ART(angle1,angle2,angle3,reflectance3)
Albedo_b4 = Albedo_ART(angle1,angle2,angle3,reflectance4)
Albedo_b5 = Albedo_ART(angle1,angle2,angle3,reflectance5) 
Albedo_b6 = Albedo_ART(angle1,angle2,angle3,reflectance6)
Albedo_b7 = Albedo_ART(angle1,angle2,angle3,reflectance7)

#窄波段反照率向宽波段反照率转换;;窄波段反照率向宽波段反照率转换，A=-0.0093+0.157a1+0.2789a2+0.3829a3+0.1131a5+0.069a7
ALBEDO = -0.0093+0.157*Albedo_b1+0.2789*Albedo_b2+0.3829*Albedo_b3+0.1131*Albedo_b5+0.069*Albedo_b7

#create the output image
driver      = ds.GetDriver()
#复制一份数据驱动
outfilename ='E:\\albedo\\output\\albedo.tif'
outDataset  = driver.Create(outfilename,cols,rows,1,GDT_Float32)
if outDataset is None:
    print 'Could not create albedo.tif'
    sys.exit(1)
#create new dataset
outBand=outDataset.GetRasterBand(1)
#write the data
outBand.WriteArray(ALBEDO,0,0)
#flush data to disk, set the NoData value and calculate stats
outBand.FlushCache()
outBand.SetNoDataValue(-99)
# georeference the image and set the projection
outDataset.SetGeoTransform(ds.GetGeoTransform())
outDataset.SetProjection(ds.GetProjection())
#*******************************************************************************
'''
;+
;project:ART积雪反照率模型
;author:shaodonghang
;date:2016-4-24
;-
pro Albedo_ART
    ;根据MOD09A1反射率和角度计算反照率
    ;打开反射率及角度数据
    fn_reflectance=dialog_pickfile(title='选择反射率数据', get_path=work_dir)
    cd, work_dir
    ;fn_angle=dialog_pickfile(title='选择角度数据')
    
    ;读入数据
    envi_open_file, fn_reflectance, r_fid=fid_reflectance
    ;envi_open_file, fn_angle, r_fid=fid_angle
    envi_file_query,fid_reflectance,ns=ns,nl=nl,nb=nb,dims=dims,$
      data_type=data_type,interleave=interleave,offset=offset
    map_info=envi_get_map_info(fid=fid_reflectance)
    
    ;for i=0, nb-1 do begin
      ;读取反射率数据
      ref_b1=envi_get_data(fid=fid_reflectance,dims=dims,pos=0)
      ref_b2=envi_get_data(fid=fid_reflectance,dims=dims,pos=1)
      ref_b3=envi_get_data(fid=fid_reflectance,dims=dims,pos=2)
      ref_b4=envi_get_data(fid=fid_reflectance,dims=dims,pos=3)
      ref_b5=envi_get_data(fid=fid_reflectance,dims=dims,pos=4)
      ref_b6=envi_get_data(fid=fid_reflectance,dims=dims,pos=5)
      ref_b7=envi_get_data(fid=fid_reflectance,dims=dims,pos=6)
      ;读取角度数据
      angle1=envi_get_data(fid=fid_reflectance,dims=dims,pos=7)
      angle2=envi_get_data(fid=fid_reflectance,dims=dims,pos=8)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
      angle3=envi_get_data(fid=fid_reflectance,dims=dims,pos=9)
     
    
      ;调用ART_BRDF函数计算Albedo
      Albedo_result_b1=ART_BRDF(angle1,angle2,angle3,ref_b1)
      Albedo_result_b2=ART_BRDF(angle1,angle2,angle3,ref_b2)
      Albedo_result_b3=ART_BRDF(angle1,angle2,angle3,ref_b3)
      Albedo_result_b4=ART_BRDF(angle1,angle2,angle3,ref_b4)
      Albedo_result_b5=ART_BRDF(angle1,angle2,angle3,ref_b5)
      Albedo_result_b6=ART_BRDF(angle1,angle2,angle3,ref_b6)
      Albedo_result_b7=ART_BRDF(angle1,angle2,angle3,ref_b7)
      ;#####################################################################
;      num_cols=ns
;      num_rows=nl
;      num_nb=nb    ;只用1波段反演 nb=1
;      ALBEDO=fltarr(num_cols,num_rows)
    ;print,Albedo_result_b1
    ;endfor
    ;窄波段反照率向宽波段反照率转换;;窄波段反照率向宽波段反照率转换，A=-0.0093+0.157a1+0.2789a2+0.3829a3+0.1131a5+0.069a7
     ALBEDO=-0.0093+0.157*Albedo_result_b1+0.2789*Albedo_result_b2+0.3829*Albedo_result_b3+0.1131*Albedo_result_b5+0.069*Albedo_result_b7
     z2=max(ALBEDO)
     z1=min(ALBEDO)
     print,z1,z2
     ;print,ALBEDO
    ;保存反照率结果
;    file=FILE_BASENAME(fn)
;    filetime=strpos(file,'A+1',/reverse_search)
;    filename=strmid(file,filetime,9)   
;    out_name = imgpath+filename+'_Albedo'+StrTrim(i+1,2)+'.tif'
;     o_fn=dialog_pickfile(title='反照率结果保存为')
;     ALBEDO=(ALBEDO>0)<1
     ;ALBEDO=ALBEDO*255
;     write_tiff,o_fn,ALBEDO
     o_fn=dialog_pickfile(title='反照率结果保存为')
     ENVI_WRITE_ENVI_FILE, ALBEDO, OUT_NAME=o_fn, /NO_COPY, $
       NS=NS, NL=NL, NB=1, DATA_TYPE=4, INTERLEAVE=INTERLEAVE, $
       OFFSET=OFFSET, MAP_INFO=MAP_INFO
     ;#######################################################################    
        
;          map_info = envi_get_map_info(fid=fid_reflectance) 
;          o_fn=dialog_pickfile(title='反照率结果保存为')
;          openw,lun,o_fn,/get_lun
;          writeu,lun,envi_get_data(fid=fid_reflectance, dims=dims, pos=ALBEDO)
;          free_lun,lun
;          ENVI_SETUP_HEAD, fname=o_fn, $
;          ns=ns, nl=nl, nb=1, $
;          interleave=0, data_type=data_type, $
;          offset=0, /write,$
;          MAP_INFO = MAP_INFO
     ;########################################################################
;         o_fn=dialog_pickfile(title='DN值保存为')
;         openw,lun,o_fn,/get_lun
;         printf,lun,ALBEDO
;         free_lun,lun
     ;########################################################################
END
    ;snow BRDF
FUNCTION ART_BRDF,angle1,angle2,angle3,data
    
    ;angle5=abs(angle3-angle4) ;相对方位角
    
    ;angle3=abs(!angle5-!angle6)
   
    ;    
    ;����
    A=1.247
    B=1.186
    C=5.157
    ;

    ;
    u0=cos(angle1*!DTOR)
    u=cos(angle2*!DTOR)


    ;
    ;cos(a*3.14/180)
    s0=sin(angle1*!DTOR)
    s=sin(angle2*!DTOR)

    ;
;    sita = acos(-u*u0+s*s0*cos(Relative*!DTOR))
;    P = 11.1*exp(-0.087*sita*180.0/!DPI)+1.1*exp(-0.014*sita*180.00/!DPI)
;    R0 = ((A+B*(u+u0)+C*u*u0+P)*1.0)/(4.0*(u+u0))
;    print,R0,P,sita,u,u0
;    ku=3*(1+2*u)/7.00
;    ku0=3*(1+2*u0)/7.00
;    v=ku*ku0/R0
;    print,ku,ku0,v,1/v
    ;
    angle4=acos(-u*u0+s*s0*cos(angle3*!DTOR))

    P=11.1*exp(-0.087*angle4*180.0/!DPI)+1.1*exp(-0.014*angle4*180.00/!DPI)

    R0 = float(((A+B*(u+u0)+C*u*u0+P)*1.0)/(4.0*(u+u0)))

    uu0=3*(1+2*u0)/7.00

    uu=3*(1+2*u)/7.00
    ;print,uu0,uu
    ;print,R0
    Rs=(data/R0)^(R0/(uu0*uu))       ;Rs为积雪光谱白空反照率
    Rp=Rs^uu0                        ;Rp为积雪光谱黑空反照率
    f=0.3                            ;f为天空散射光因子
    result=(1-f)*Rp+f*Rs             ;result位积雪光谱反照率，f为天空散射光因子，取地表反照率产品MOD43中的0.3
    ;PRINT,result
    RETURN,result
END 
'''      
# ****************************************************************************************** 
newcols = 226
newrows = 157

# lat_uhh = np.loadtxt('F:/Python/data/ivt_upstreamHEIHE.txt',skiprows=6);lat_uhh=np.flipud(lat_uhh)
# lon_uhh = np.loadtxt('F:/Python/data/lat_upstreamHEIHE.txt',skiprows=6);lon_uhh=np.flipud(lon_uhh)
# ivt_uhh = np.loadtxt('F:/Python/data/lon_upstreamHEIHE.txt',skiprows=6);ivt_uhh=np.flipud(ivt_uhh)
plt.imshow(ALBEDO,origin='lower left')
plt.colorbar()
plt.savefig('D:\\ALBEDO.jpg')  #save the figure
plt.show()
plt.clf()
print 'end' 