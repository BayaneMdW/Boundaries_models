import numpy as np
import pandas as pd


def Formisano1979(x, args):
    
    '''
    Formisano 1979 : Magnetopause and Bow shock models. Give positions of the boudaries in plans (XY) with Z=0 and (XZ) with Y=0.
    function's arguments :
        - x :  X axis (array) in Re (earth radii)
        - args : coefficients Aij are determined from many boundary crossings and they depend on upstream conditions. 
          Default parameter for the bow shock and the magnetopause respectively are :
            default_bs_formisano = [0.52,1,1.05,0.13,-0.16,-0.08,47.53,-0.42,0.67,-613] 
            default_mp_formisano = [0.65,1,1.16,0.03,-0.28,-0.11,21.41,0.46,-0.36,-221]
            
     return : DataFrame (Pandas) with the position (X,Y,Z) in Re of the wanted boudary to plot (XY) and (XZ) plans.
    '''
    
    
    a11,a22,a33,a12,a13,a23,a14,a24,a34,a44 = args[0],args[1],args[2],args[3],args[4],args[5],args[6],args[7],args[8],args[9]
    
    a_y = a22
    b_y = a12*x + a24
    c_y = a11*x**2 + a14*x + a44
    
    delta_y =(b_y**2-4*a_y*c_y)
   
    
    ym = (-b_y - np.sqrt(delta_y))/(2*a_y)
    yp = (-b_y + np.sqrt(delta_y))/(2*a_y)
  
    a_z = a33
    b_z = a13*x + a34
    c_z = a11*x**2 + a14*x + a44
    
    delta_z =(b_z**2-4*a_z*c_z)
    
    zm = (-b_z - np.sqrt(delta_z))/(2*a_z)
    zp = (-b_z + np.sqrt(delta_z))/(2*a_z)
    
    
    pos=pd.DataFrame({'X' : np.concatenate([x, x]),
                      'Y' : np.concatenate([yp, ym]),
                      'Z' : np.concatenate([zp, zm]),})        
    
    return pos.dropna()