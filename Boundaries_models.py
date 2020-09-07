import numpy as np
import pandas as pd



def Fairfield1971(x,args):
    
    ''' 
    Fairfield 1971 : Magnetopause and Bow shock models. Give positions of the boudaries in plans (XY) with Z=0 and (XZ) with Y=0.
    function's arguments :
        - x :  X axis (array) in Re (earth radii)
        - args : coefficients Aij are determined from many boundary crossings and they depend on upstream conditions. 
          
        --> Default parameter for the bow shock and the magnetopause respectively are :
            default_bs_fairfield = [0.0296,-0.0381,-1.28,45.644,-652.1]
            default_mp_fairfield = [-0.0942,0.3818,0.498,17.992,-248.12]
            
     return : DataFrame (Pandas) with the position (X,Y,Z) in Re of the wanted boudary to plot (XY) and (XZ) plans.
    '''
    
    
    A,B,C,D,E = args[0],args[1],args[2],args[3],args[4]
    
    a = 1
    b = A*x + C
    c = B*x**2 + D*x + E
    
    delta = b**2-4*a*c
    
    ym = (-b - np.sqrt(delta))/(2*a)
    yp = (-b + np.sqrt(delta))/(2*a)
    
    pos=pd.DataFrame({'X' : np.concatenate([x, x[::-1]]),
                      'Y' : np.concatenate([yp, ym[::-1]]),
                      'Z' : np.concatenate([yp, ym[::-1]]),})        
    
    return pos.dropna()


def Formisano1979(x, args):
    
    '''
    Formisano 1979 : Magnetopause and Bow shock models. Give positions of the boudaries in plans (XY) with Z=0 and (XZ) with Y=0.
    function's arguments :
        - x :  X axis (array) in Re (earth radii)
        - args : coefficients Aij are determined from many boundary crossings and they depend on upstream conditions. 
          
        --> Default parameter for the bow shock and the magnetopause respectively are :
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
    
    
    pos=pd.DataFrame({'X' : np.concatenate([x, x[::-1]]),
                      'Y' : np.concatenate([yp, ym[::-1]]),
                      'Z' : np.concatenate([zp, zm[::-1]]),})        
    
    return pos.dropna()






def BS_Jerab2005( Np, V, Ma, B, gamma=2.15 ):
    
    '''
    Jerab 2005 Bow shock model. Give positions of the box shock in plans (XY) with Z=0 and (XZ) with Y=0 as a function of the upstream solar wind.
    function's arguments :
        - Np : Proton density of the upstream conditions 
        - V  : Speed of the solar wind
        - Ma : Alfven Mach number
        - B  : Intensity of interplanetary magnetic field 
        - gamma : Polytropic index ( default gamma=2.15)
        
        
        --> mean parameters :  Np=7.35, V=425.5, Ma=11.23, B=5.49
     
     return : DataFrame (Pandas) with the position (X,Y,Z) in Re of the bow shock to plot (XY) and (XZ) plans.
    '''
    
    def make_Rav(theta,phi):
        a11 = 0.45 
        a22 = 1
        a33 = 0.8
        a12 = 0.18
        a14 = 46.6
        a24 = -2.2
        a34 = -0.6
        a44 = -618

        a = a11*np.cos(theta)**2 + np.sin(theta)**2 *( a22*np.cos(phi)**2 + a33*np.sin(phi)**2 )
        b = a14*np.cos(theta) +  np.sin(theta) *( a24*np.cos(phi) + a34*np.sin(phi) )
        c = a44

        delta = b**2 -4*a*c

        R = (-b + np.sqrt(delta))/(2*a)
        return R

    
    
    C = 91.55
    D = 0.937*(0.846 + 0.042*B )
    R0 = make_Rav(0,0)
    
    theta = np.linspace(0,2.5,200)
    phi = [np.pi,0]
    
    x,y= [],[]
    
    for p in phi:
        Rav = make_Rav(theta,p)
        K = ((gamma-1)*Ma**2+2)/((gamma+1)*(Ma**2-1))
        R = (Rav/R0)*(C/(Np*V**2)**(1/6))*(1+ D*K)

        x = np.concatenate([x, R*np.cos(theta)])
        y = np.concatenate([y, R*np.sin(theta)*np.cos(p)])
        

    pos = pd.DataFrame({'X' : x , 'Y' : y, 'Z' : y}) 
    
    return pos.sort_values('Y')


def MP_Shue1998(Pd, Bz):
    ''' Shue 1998 Magnetopause model. Returns the MP distance for given
    theta (r), dynamic pressure (in nPa) and Bz (in nT).
    * theta: Angle from the x axis (model is cylindrical symmetry)
    * PD: Dynamic Pressure in nPa
    * Bz: z component of IMF in nT'''
    r0 = (10.22+1.29*np.tanh(0.184*(Bz+8.14)))*Pd**(-1./6.6)

    a = (0.58-0.007*Bz)*(1+0.024*np.log(Pd))
    theta = np.arange( -np.pi+0.01, np.pi-0.01, 0.001)
    #theta = np.arange( -np.pi/2, np.pi/2+0.01, 0.01)

    r = r0*(2./(1+np.cos(theta)))**a

    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = r*np.sin(theta)

    pos=pd.DataFrame({'X' : x,
                      'Y' : y,
                      'Z' : z,})        
    
    return pos.dropna()