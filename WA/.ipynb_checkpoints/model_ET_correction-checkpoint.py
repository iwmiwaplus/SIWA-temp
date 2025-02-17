# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 08:22:17 2019

@author: sse
"""
import os
import numpy as np
import gdal
import xarray as xr
#import glob
#import datetime
import warnings
import time


#%% Functions
def open_nc(nc,timechunk=1,chunksize=1000):
    dts=xr.open_dataset(nc)
    key=list(dts.keys())[0]
    var=dts[key].chunk({"time": timechunk, "latitude": chunksize, "longitude": chunksize}) #.ffill("time")
    return var,key

def open_nc_e(nc):
    dts=xr.open_dataset(nc)
    key=list(dts.keys())[0]
    var=dts[key]
    return var,key

def OpenAsArray(fh, bandnumber = 1, dtype = 'float32', nan_values = False):
    """
    Open a map as an numpy array. 
    
    Parameters
    ----------
    fh: str
        Filehandle to map to open.
    bandnumber : int, optional 
        Band or layer to open as array, default is 1.
    dtype : str, optional
        Datatype of output array, default is 'float32'.
    nan_values : boolean, optional
        Convert he no-data-values into np.nan values, note that dtype needs to
        be a float if True. Default is False.
        
    Returns
    -------
    Array : ndarray
        Array with the pixel values.
    """
    datatypes = {"uint8": np.uint8, "int8": np.int8, "uint16": np.uint16, "int16":  np.int16, "Int16":  np.int16, "uint32": np.uint32,
    "int32": np.int32, "float32": np.float32, "float64": np.float64, "complex64": np.complex64, "complex128": np.complex128,
    "Int32": np.int32, "Float32": np.float32, "Float64": np.float64, "Complex64": np.complex64, "Complex128": np.complex128,}
    DataSet = gdal.Open(fh, gdal.GA_ReadOnly)
    Type = DataSet.GetDriver().ShortName
    if Type == 'HDF4':
        Subdataset = gdal.Open(DataSet.GetSubDatasets()[bandnumber][0])
        NDV = int(Subdataset.GetMetadata()['_FillValue'])
    else:
        Subdataset = DataSet.GetRasterBand(bandnumber)
        NDV = Subdataset.GetNoDataValue()
    Array = Subdataset.ReadAsArray().astype(datatypes[dtype])
    if nan_values:
        Array[Array == NDV] = np.nan
    Array = Array.astype(np.float32)
    return Array

#%% main
def correct_et(MAIN_FOLDER,p_in,e_in, aridity, start_year, end_year, chunks=[1,1000,1000]):
    '''
    Arguments:
        
    ## required   
    MAIN_FOLDER='$PATH/nc/'
    p_in = '$PATH/p_monthly.nc' # Monthly Precipitation
    e_in = '$PATH/e_monthly.nc' # Monthly Actual Evapotranspiration
    i_in = '$PATH/i_monthly.nc' # Monthly Interception
    rd_in = '$PATH/nRD_monthly.nc' # Monthly Number of Rainy days
    lu_in = '$PATH/lcc_yearly.nc' # Yearly WaPOR Land Cover Map
    smsat_file = '$PATH/thetasat.nc' #Saturated Water Content (%)
    start_year=2009 
    
    #default
    f_perc=1 # percolation factor
    f_Smax=0.9 #threshold for percolation
    cf =  20 #f_Ssat soil mositure correction factor to componsate the variation in filling up and drying in a month
    f_bf = 0.1 # base flow factor (multiplier of SM for estimating base flow)
 
    '''
    warnings.filterwarnings("ignore", message='invalid value encountered in greater')
    warnings.filterwarnings("ignore", message='divide by zero encountered in true_divide')
    warnings.filterwarnings("ignore", message='invalid value encountered in true_divide')
    warnings.filterwarnings("ignore", message='overflow encountered in exp')

    tchunk=chunks[0]
    chunk=chunks[1]
    
    
    #########
    #to remove: just to check root depth computations
    #LU_files = becgis.list_files_in_folder(os.path.join(MAIN_FOLDER, 'LU'))
    #driver, ndv, xsize, ysize, geot, projection = becgis.get_geoinfo(LU_files[0])
    
    
    
    Pt,_=open_nc(p_in,timechunk=tchunk,chunksize=chunk)
#    Pt,_ = open_nc_e(p_in)
    E,_=open_nc(e_in,timechunk=tchunk,chunksize=chunk)
#    E,_ = open_nc_e(e_in)
    A,_ = open_nc(aridity,timechunk=tchunk,chunksize=chunk)
    
    Ari = A[0]
    
    for j in range(len(E.time)):
        

#        for j in range(end_year - start_year+1):
        t1 = j
        t2 = (j+1)   

        for t in range(t1,t2):
#             print('time: ', t)
            P = Pt.isel(time=t)
            ETa = E.isel(time=t)
            
            
#             Correct ETa for desert areas
            ETa = ETa.where(ETa > 0, P)
            ETa = P.where((ETa < P) & ( Ari < 0.2),ETa)
#             ETa = ETa.where(ETa > 0, (P.where((P != 0) & (ari < 0.3)), P*0 ))
    

            if t == 0:
                eta = ETa
            else:
                eta = xr.concat([eta, ETa], dim='time')
                
            
            
            del P
            del ETa
    
    # force time dimension of output DataArray equal to input time dimension
    eta['time']=E['time']
    
    #change coordinates order to [time,latitude,longitude]
    eta=eta.transpose('time','latitude','longitude')

#####################
   
    attrs={"units":"mm/month", "source": "-", "quantity":"Total_ET_M"}
    eta.attrs=attrs
    eta.name = 'Total_ET_M'

    ### Write netCDF files
    # chunks = [1, 300, 300]
    comp = dict(zlib=True, 
                least_significant_digit=2, 
                chunksizes=chunks)
    
    ##total ET 
    start = time.time()
    print("\n\nwriting the ET_a netcdf file\n\n")
    eta_path=os.path.join(MAIN_FOLDER,'eta_monthly.nc')
    encoding = {"Total_ET_M": comp}
    eta.to_netcdf(eta_path,encoding=encoding)
    del eta
    end = time.time()
    print('\n',end - start)
