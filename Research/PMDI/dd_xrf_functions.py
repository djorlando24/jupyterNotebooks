#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
    X-RAY FLUORESCENCE SIGNAL TRAPPING CORRECTION FUNCTIONS
    from PMDICanister_02_signalTrappingModel.ipynb
    
    @author Daniel Duke <daniel.duke@monash.edu>
    @copyright (c) 2022 LTRAC
    @license GPL-3.0+
    @version 0.0.1
    @date 14/03/2023
        __   ____________    ___    ______
       / /  /_  ____ __  \  /   |  / ____/
      / /    / /   / /_/ / / /| | / /
     / /___ / /   / _, _/ / ___ |/ /_________
    /_____//_/   /_/ |__\/_/  |_|\__________/

    Laboratory for Turbulence Research in Aerospace & Combustion (LTRAC)
    Monash University, Australia

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

    Version history:
        26/02/2023 - First version.
        14/03/2023 - Updated to load required libraries and tested ok.
"""

import numpy as np
import h5py
import scipy.optimize, scipy.integrate, scipy.interpolate



def mergeInterpolateScans(xData, yData, zData, positionDecimalPlaces=3):
    '''
        Take a set of HDF objects and stitch together using linear interpolation
        onto a rectilinear grid that can easily be plotted and analysed.
        'positionDecimalPlaces' arg will round off the position locations in case
        of small variations between scans.
    '''
    
    # Pull all data into flat lists.
    xAll = []; yAll = []; zAll = []
    for k in xData:
        xAll.extend(xData[k][...].ravel())
        yAll.extend(yData[k][...].ravel())
        zAll.extend(zData[k][...].ravel())
        
    # Find unique values for x and y across all data sets to build points to interpolate onto.
    xp = np.unique(np.round(xAll,positionDecimalPlaces))
    yp = np.unique(np.round(yAll,positionDecimalPlaces))
    x, y = np.meshgrid(xp, yp)
    
    # Interpolation of z
    L1=scipy.interpolate.LinearNDInterpolator([(xAll[n], yAll[n]) for n in range(len(xAll))], zAll)
    
    pts = [(x.ravel()[n], y.ravel()[n]) for n in np.arange(np.product(x.shape))]
    z = L1(pts).reshape(x.shape)
    
    # Strictly mask points that are not just outside the convex hull, but
    # outside the rectilinear grid defined by each scan.
    mask = np.zeros_like(z)
    for k in xData:
        xRange = (np.nanmin(xData[k]), np.nanmax(xData[k]))
        yRange = (np.nanmin(yData[k]), np.nanmax(yData[k]))
        mask[(x>=xRange[0])&(x<=xRange[1])&(y>=yRange[0])&(y<=yRange[1])] += 1
    z[mask==0] = np.nan
    
    # Remove completely empty rows and columns of the array
    nonEmpty = np.max(mask,axis=1)>0
    x = x[nonEmpty,...]
    y = y[nonEmpty,...]
    z = z[nonEmpty,...]
    nonEmpty = np.max(mask,axis=0)>0
    x = x[:,nonEmpty,...]
    y = y[:,nonEmpty,...]
    z = z[:,nonEmpty,...]

    return x,y,z

def readHorizontalScans(xData, yData, tData, zData, positionDecimalPlaces=3, timeDecimalPlaces=0, fillValue=np.nan):
    '''
        Take a set of HDF objects as repeated scans in time and output for horizontal scans a 2D matrix
        of (x,y,time).
    '''
    
    xAll = []; yAll = []; tAll = []; zAll = []
    for k in xData:
        xAll.extend(xData[k][...].ravel())
        yAll.extend(yData[k][...].ravel())
        tAll.extend(tData[k][...].ravel())
        zAll.extend(zData[k][...].ravel())
    
    # Find unique values for x and y across all data sets.
    if isinstance(positionDecimalPlaces,tuple):
        xDecimal, yDecimal = positionDecimalPlaces
    else:
        xDecimal = positionDecimalPlaces
        yDecimal = positionDecimalPlaces
    xp = np.unique(np.round(xAll,xDecimal))
    yp = np.unique(np.round(yAll,yDecimal))
    tp = np.unique(np.round(tAll,timeDecimalPlaces))

    x, y, t = np.meshgrid(xp, yp, tp, indexing='ij')
    
    # We will fill the result matrix using nearest-neighbour Delaunay triangulation
    L1=scipy.interpolate.NearestNDInterpolator([(xAll[n], yAll[n], tAll[n]) for n in range(len(xAll))], zAll)
    
    pts = [(x.ravel()[n], y.ravel()[n], t.ravel()[n]) for n in np.arange(np.product(x.shape))]
    z = L1(pts).reshape(x.shape)
    
    
    # Strictly mask points that are not just outside the convex hull, but
    # outside the rectilinear grid defined by each scan.
    mask = np.zeros_like(z); tol=1e-3
    for k in xData:
        xRange = (np.nanmin(xData[k])-tol, np.nanmax(xData[k])+tol)
        yRange = (np.nanmin(yData[k])-tol, np.nanmax(yData[k])+tol)
        tRange = (np.nanmin(tData[k])-tol, np.nanmax(tData[k])+tol)
        xTest = (x>=xRange[0])&(x<=xRange[1])
        yTest = (y>=yRange[0])&(y<=yRange[1]) # skip exact y check for horizontal scans
        tTest = (t>=tRange[0])&(t<=tRange[1])
        mask[xTest&yTest&tTest] += 1
    if fillValue is not None:
        z[mask==0] = fillValue
    
    return x,y,t,z

def secant(x,r=1.0):
    ''' 
        Secant of a circle radius r at distance x from the centerline.
        Returns zero when outside the radius.
    '''
    s = np.zeros_like(x)
    s[np.abs(x)<=r] = 2*np.sqrt(r**2 - x[np.abs(x)<=r]**2)
    return s

def secantAnnulus(x,ri,ro):
    '''
        Secant of an annulus with inner radius ri and outer ro, at
        distance x from the centerline.
        Returns zero when outside the radius
    '''
    s = secant(x,ro)
    s -= secant(x,ri)
    return s

def secantPartialInsideCircle(x,z,R,rDet,thetaDet):
    '''
        Find the length of a segment of a secant from point (x,z) inside a circle of radius R
        to a detector outside the circle at radius rDet and angle thetaDet from the origin.
        
        This function can accept 1D vectors for x and z.
    '''
    rA = np.sqrt(x**2 + z**2)
    
    sd = np.sqrt ( (rDet*np.cos(thetaDet) - x)**2 + (rDet*np.sin(thetaDet) - z)**2 )
    sdQ= np.sqrt ( (rDet*np.cos(thetaDet) + x)**2 + (rDet*np.sin(thetaDet) + z)**2 )    

    cosGamma = (-rDet**2 + x**2 + z**2 + sdQ**2)/(2*sdQ*rA)
    qb = 2*cosGamma*rA
    qc = x**2 + z**2 - R**2

    
    # solve gfsq avoiding imaginary solutions
    de = (qb**2)/4 - qc
    sdi = (-qb/2 + np.sqrt(np.abs(de))) * (de>=0) 
    #sdi2 = (-qb/2 - np.sqrt(np.abs(de))) * (de>=0) 
    
    #sdi = np.nanmax(np.vstack((sdi1,sdi2)),axis=0)
    
    # no negative solutions allowed
    sd[sd<0]=0
    sdi[sdi<0]=0 
    
    # zero solutions outside the circle
    sdi[rA>R]=0
    
    return sd, sdi

def secantRayProjector(x,z,R,rDet,thetaDet,rayResolutionPts=150):
    '''
        For points outside the circle, project a ray and determine the path length of secant that
        passes through the circle. int rayResolutionPts set the number of points along each ray.
        
        This function can accept 1D vectors for x and z.
    '''
    # find ray from point x,z to detector and check if it passes thru the circle
    zDet = rDet*np.sin(thetaDet)
    xDet = rDet*np.cos(thetaDet)
    m = (rDet*np.sin(thetaDet) - z)/(rDet*np.cos(thetaDet) - x)
    c = z - m*x
    xp = np.linspace(x,xDet,rayResolutionPts)
    zp = m*xp + c
    rp = np.sqrt(xp**2 + zp**2)
    rayData = [xp,zp,rp<=R]
    
    sd,sdi = secantPartialInsideCircle(xp,zp,R,rDet,thetaDet)
    
    # set points outside circle with rays passing thru circle to the max value along the ray.
    rA = np.sqrt(x**2 + z**2)
    outsideCircle = rA>R
    sdi_outside = np.zeros_like(x)
    #print(sdi_outside.shape, sdi.shape)
    sdi_outside[outsideCircle] = np.nanmax(sdi[:,outsideCircle],axis=0)
    
    return sdi_outside, rayData
    
def secantPathLengthWrapper(x,z,R,rDet,thetaDet,rayResolutionPts=150):
    ''' 
        A wrapper function to compute the internal and external part of the projected path length
        and add them together. Other returned values are passed through.
        
        This function can accept 1D vectors for x and z.
    '''
    sd,sdi=secantPartialInsideCircle(x,z,R,rDet,thetaDet)
    sdi_outside,rayData=secantRayProjector(x,z,R,rDet,thetaDet,rayResolutionPts)
    return sdi+sdi_outside, sd, rayData

def calcR2(f, xdata, ydata, popt):
    residuals = ydata- f(xdata, *popt)
    ss_tot = np.sum((ydata-np.mean(ydata))**2)
    return 1 - (np.nansum(residuals**2) / ss_tot)


################################################################################################################
if __name__ == '__main__':
    print("This file is intended to be loaded as a module from iPython notebook")
