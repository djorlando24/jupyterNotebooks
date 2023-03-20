"""
    X-RAY FLUORESCENCE SIGNAL TRAPPING CORRECTION FUNCTIONS
    from PMDICanister_03_signalTrappingApply.ipynb,
    for re-use on multiple files and scans.
    
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
        14/03/2023 - First version.
"""

from dd_xrf_functions import *
import numpy as np
import h5py


'''
    Convert the raw beamline Y position to a useful coordinate
    (relative to the valve height).
        
    This was determined based on the dummy scans of an empty can, from the previous notebook
'''
def yTransform(y):
    return 75-y

'''
    Calculate absorption through canister wall and rectangular external plastic parts of canister holder
    for transverse co-ordinates x with can aligned at center x0, inner radius ri, wall thickness thk.
    
    Assuming by convention all lengths in mm, and mu is attenuation length in _cm_
'''
def canisterWallExtFn(x,x0,ri,thkCan,muCan, thkExtGradient, thkExtConst, muExt, riValve):
    
    if (ri<0) | (muCan<0) | (thkExtConst<0) | (muExt<0) | (riValve<0): return 0
    if (np.abs(x0)>5) | (np.abs(thkExtGradient)>5) | (np.abs(thkCan)>1.5) | (riValve>4): return 0
    
    return secantAnnulus(x-x0,ri,ri+thkCan)*muCan*0.1 +\
           secant(x-x0,riValve)*muCan*0.1 +\
           (thkExtGradient*(x-x0) + thkExtConst - secantAnnulus(x-x0,ri,ri+thkCan))*muExt*0.1

'''
    Apply signal trapping corrections in the x,z plane. Assuming a cylindrical canister and uniform external
    environment with a uniform internal liquid density (but nonuniform fluorescing tracer concentration
    possible).
    
    Can handle n-dimensional array data by flattening everything and assessing each sample one at a time.
'''
def applySignalTrapping(x_All,y_All,fluor_All,pin_All,I0_All,\
                        propellant = '134',ethMassFrac=0.08, tracer='I', line='a'):
    
        
    # Constants determined by fitting to empty canister data.
    x0=0                 # x-alignment of center of the canister.
    thkCan=0.706982      # canister wall thickness [mm]
    muCan=1.732022       # attenuation/rho of the canister wall at incident beam energy [1/cm]
    muExt=0.455318       # attenuation/rho of the external components
    
    # Functions that depend on y.
    thkExtGradient = lambda y: 0.2   # attenuation gradient in x of the external environment [1/cm/mm]
    thkExtConst = lambda y: 75.      # attenuation average of the external environment [1/cm]
    
    # canister inner wall radius
    ri = lambda y: (10.14)*(y>=0) + (3.38)*(y<0)
    
    # effective radius of the metering valve or canister dome part [mm]
    riValve = lambda y: (-0.1444*y + 4.275)*(y<=14)*(y>=6.75) + (3.25)*(y<6.75)
    
    # Density of fluid at lab temperature (20degC) [kg/m3]
    rhoFluid = {'134':1225.3, '152':911.97, '1234E':1293., 'EtOH':789.2}
    
    # X-ray constants
    
    # attenuation/rho of the liquid formulation at the incident energy [1/cm]
    muFluid = {'134':0.328336,'152':0.222837,'1234E':0.271999,'EtOH':0.186543}
    
    # attenuation/rho of the fluorescing tracer at the incident energy [1/cm]
    muTracer  = {'I':19.4738,'Ba':12.933*137.327/233.38}
    
    # attenuation/rho of the liquid formulation at the emission energy [1/cm]
    muFluid_Ka  = {'I':{'134':0.572199,'152':0.364844,'1234E':0.464389,'EtOH':0.330578},\
                   'Ba':{'134':0.464644,'152':0.30246,'1234E':0.379642,'EtOH':0.289781}}
    
    # attenuation/rho of the canister wall at the emission energy [1/cm]
    muCan_Ka  = {'I':muCan * 3.4816/1.38491,'Ba':muCan * 2.54821/1.38491}
    
    # attenuation/rho of the air at the emission energy [1/cm]
    muAir_Ka  = {'I':0.0004244,'Ba':0.000359733}
    
    # fraction of emission at the chosen K emission line for analysis (a=alpha, b=beta)
    lineYield = {'a':1.54, 'b':0.18} # very similar for both I and Ba
    
    fluorYield = 0.85 # fluorescence yield for this edge.  Very similar for both I and Ba
    
    # Detector constants
    rDet = 275.                        # distance from canister axis to the detector
    thetaDet = 67*np.pi/180.           # angle of the detector with respect to the X axis of the scan.
    thetaDet += 90.
    solidDet = np.pi*(15**2)/(rDet**2) # approximate solid angle of the detector relative to canister axis.
    
    # Calculation fluid mixture absorption, accounting for any density differences.
    rhoMix = (rhoFluid[propellant]*(1-ethMassFrac) + rhoFluid['EtOH']*ethMassFrac)*1e-3 # g/cm^3
    muFluidMix = muFluid[propellant]*(1-ethMassFrac) + muFluid['EtOH']*ethMassFrac # 1/cm
    muFluidMix_emission = muFluid_Ka[tracer][propellant]*(1-ethMassFrac) +\
                          muFluid_Ka[tracer]['EtOH']*ethMassFrac # 1/cm
    
    
    # Setup arrays
    x = x_All.ravel() - x0 # shift x positions so canister is in the centre of the coordinate system.
    y = y_All.ravel()
    f = fluor_All.ravel()
    p = pin_All.ravel()
    I0 = I0_All.ravel()
    correction = np.ones_like(x)
    
    # Loop each element in the matrix
    for i in range(len(x)):
        
        # Determine attenuation lengths of incident beam in ext & wall, assuming z-symmetry.
        incidentAbsModel = canisterWallExtFn(x[i],0,ri(y[i]),thkCan,muCan, thkExtGradient(y[i]),\
                                            thkExtConst(y[i]), muExt, riValve(y[i]))
        
        # Determine path length of incident beam thru the fluid in the can.
        # Missing step here: Assume full instead of using Pin to check full/empty!
        fluidPathLength = secant(x[i],r=ri(y[i]))
    
        # Add the absorption due to the liquid through secant.
        beamAttenuation = incidentAbsModel + fluidPathLength*muFluidMix*0.1
        # 'beamAttenuation' should now be close to the Pin Diode -log(I/I0) value.

        # Take half the incident attenuation and calculate the flux at z=0 by accounting for I0 variation.
        incidentFluxVariation = I0[i]/np.nanmean(I0)
        beamAbsorptionActual = 1 - (p[i]/I0[i])
        fluxAtFocus = np.exp(-beamAttenuation*0.5) * beamAbsorptionActual * incidentFluxVariation
        
        # Determine amount of emitted radiation in the solid angle, assuming the tracer is present
        possibleEmission = lineYield[line] * fluorYield * solidDet * fluxAtFocus \
                           / sum(list(lineYield.values()))
        
        # Determine attenuation of emitted radiation through the liquid to the wall.
        # Sample several positions either side of z=0 to account for contributions along the beam path.
        z_ = np.linspace(-2.5,2.5,5) # mm
        s1,s2,rayData = secantPathLengthWrapper(x[i]*np.ones_like(z_),z_,ri(y[i])*np.ones_like(z_),rDet,thetaDet)
        attenLengthEmission_fluid = np.nanmean(s1)*muFluidMix_emission*0.1
        
        # Determine attenuation of emitted radiation through the wall and exterior and air.
        attenLengthEmission_can = thkCan*muCan_Ka[tracer]*0.1
        attenLengthEmission_air = np.nanmean(s2)*muAir_Ka[tracer]*0.1
        
        # Generate the lumped constant for the signal trapping correction for all above effects:
        # attenuation of the emission & amount of flux absorbed at the focus.
        correction[i]=np.exp(- attenLengthEmission_fluid - attenLengthEmission_can - attenLengthEmission_air )
        correction[i]*=fluxAtFocus
    
    return correction.reshape(fluor_All.shape), rhoMix


'''
    Read HDF file and load scans for fluorescence.
    Don't apply signal trapping corrections yet.
'''
def readFluorescence(filename, fluorLine='I Ka', scanType='HorizontalScans',\
                     positionDecimalPlaces=3, timeDecimalPlaces=0, fillValue=np.nan):

    readerFunction =readHorizontalScans
    timeVar = 'shotCounter'
    if ('BaSO4' in filename):
        timeVar = 'shakeTimer'
        #readerFunction = readTimeScans
        
    
    with h5py.File(filename,'r') as H:
    
        # Check for existence of group
        if not scanType in H: return None,None,None,None,None,None
    
        # Read data.
        x,y,t,zF = readerFunction( H['%s/x' % scanType], H['%s/y' % scanType],\
                                        H['%s/%s' % (scanType,timeVar)],\
                                        H['%s/%s/integral' % (scanType,fluorLine)],\
                                        positionDecimalPlaces, timeDecimalPlaces, fillValue )
        
        x,y,t,zP = readerFunction( H['%s/x' % scanType], H['%s/y' % scanType],\
                                        H['%s/%s' % (scanType,timeVar)],\
                                        H['%s/pinDiode' % scanType],\
                                        positionDecimalPlaces, timeDecimalPlaces, fillValue )

        x,y,t,I0 = readerFunction( H['%s/x' % scanType], H['%s/y' % scanType],\
                                        H['%s/%s' % (scanType,timeVar)],\
                                        H['%s/diamondMonitor' % scanType],\
                                        positionDecimalPlaces, timeDecimalPlaces, fillValue )

        # Calculations
        pinDiode = -np.log(zP/I0)
        
        # Outlier masking for big horizontal scans
        if len(x.shape) >= 3:
            for i in range(x.shape[1]):
                for j in range(x.shape[2]):
                    dz = np.diff(zF[...,i,j])
                    dz = np.hstack((0, dz))
                    # mask large sudden changes in level
                    zF[np.abs(dz)>100,i,j] = np.nan

    return x,y,t,zP,I0,zF
