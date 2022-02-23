import math
import numpy as np

Epsilon = 0.0000001
print('changed9')

#	Indices Description
#	-------------------
#	NDVI       - Indicates vegetation regions and plant vigor. IndexImage
#	             pixel values lie in the range of [-1, 1]. Index values in
#	             the range of [0.2, 0.8] indicate vegetation.
#	
#	OSAVI      - Indicates areas having sparse vegetation with high soil
#	             influence. This index is insensitive to soil. IndexImage
#	             pixel values lie in the range of [-1, 1]. High index
#	             values indicate healthier and denser vegetation.
#	
#	SR         - Indicates areas having green vegetation. IndexImage pixel
#	             values are generally greater than 0. Typically index value
#	             greater than 3 signifies green vegetation.
#	
#	EVI        - Indicates vegetation in areas of high leaf area index.
#	             This index is insensitive to atmospheric and aerosol
#	             influences. IndexImage pixel values typically lie in the
#	             range of [-1, 1]. Index values in the range of [0.2, 0.8]
#	             indicate healthy vegetation. Data cube values should
#	             contain surface reflectance values and be in the range of
#	             [0, 1] for effective calculation of this index.
#	
#	GVI        - Emphasizes green vegetation regions while minimizing the
#	             background soil effects. IndexImage pixel values lie in
#	             the range of [-1, 1]. High index values indicate healthier
#	             and denser vegetation. This index was originally designed
#	             for the Landsat TM.
#	
#	MCARI      - Indicates relative abundance of chlorophyll. This index is
#	             illumination and non-photosynthetic materials invariant.
#	
#	MTVI       - This index can be used to estimate leaf area. This index
#	             is calculated as the area of a hypothetical triangle in
#	             spectral space that connects minimum chlorophyll
#	             absorption, NIR shoulder, and green peak reflectance.
#	
#	PRI        - This index indicates presence of carotenoid pigments
#	            (mainly xanthophyll pigments) in live foliage. Carotenoid
#	             pigments are indicative of photosynthetic light use
#	             efficiency. IndexImage pixel values lie in the range of
#	             [-1, 1]. The typical range for photo synthetically active
#	             vegetation lies in the range of [-0.04, 0.2].
#	
#	NDNI       - Indicates the relative nitrogen content in vegetation
#	             canopies. IndexImage pixel values lie in the range of
#	             [-1, 1]. Data cube values should contain surface
#	             reflectance values and be in the range of [0, 1] for
#	             effective calculation of this index.
#	
#	MSI        - Indicates relative water content in vegetation canopies.
#	             IndexImage pixel values lie in the range of 0 to more than
#	             3. Typical moisturized vegetation lies in the range of
#	             [0.4, 2].
#	
#	CAI        - Indicates areas containing dried plant material.
#	             IndexImage pixel values lie in the range of [-3, 4]. The
#	             typical range for dried plant material lies in the range
#	             of [-2, 4]. Data cube values should contain surface
#	             reflectance values and be in the range of [0, 1] for
#	             effective calculation of this index.
#	
#	CMR        - Indicates hydrothermally altered rocks containing clay and
#	             alunite. Data cube values should contain surface
#	             reflectance values and be in the range of [0, 1] for
#	             effective calculation of this index.
#	
#	NBR        - Indicates burned regions. IndexImage pixel values lie in
#	             the range of [-1, 1]. Index values greater than 0.1
#	             indicate burned regions. This index was originally
#	             designed for the Landsat TM.
#	
#	MNDWI      - Highlights open water bodies. This index is insensitive to
#	             built-up land, vegetation, and soil. IndexImage pixel
#	             values lie in the range of [-1, 1]. Index values greater
#	             than 0.09 indicate open water bodies. This index was
#	             originally designed for the Landsat TM.
#	
#	NDBI       - Highlights urban area built-up lands. IndexImage pixel
#	             values lie in the range of [-1, 1]. Index values lesser
#	             than 0 indicate built-up lands. This index was originally
#	             designed for the Landsat TM.
#	
#	NDMI       - Indicates muddy or shallow water pixels. IndexImage pixel
#	             values lie in the range of [-1, 1]. This index was
#	             originally designed to improve the accuracy of the QUick
#	             Atmospheric Correction (QUAC) algorithm.



# Define functions for spectral indices

def NDVI(R,NIR):
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    #LS8 = B4,B5    LS57 = B3,B4
    numerator=NIR - R
    denominator=NIR + R
    ndvi=np.where(np.abs(denominator)<Epsilon,0.0,(numerator) / (denominator))
#    ndvi = (NIR - R) / (NIR + R)
#    ndvi = ndvi.astype(np.float32)
    return ndvi
def LAI(B,R,NIR):
    # Boegh,E H. Soegaard, N. Broge, C. Hasager, N. Jensen, K. Schelde, and A. Thomsen. (2002). Airborne Multi-spectral Data for Quantifying Leaf Area Index, Nitrogen Concentration and Photosynthetic Efficiency in Agriculture. Remote Sensing of Environment 81, no. 2-3 (2002).
    
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    evi=0*np.ones(shape=(NIR.shape[0],B.shape[1]),dtype=np.float32)
    numerator=(NIR - R)
    denominator= (NIR + (6.0*R) - (7.5 * B) + 1.0)
    evi=np.where(np.abs(denominator)<Epsilon,0.0,2.5 * ((numerator) / (denominator) ))
    lai = (3.618 * evi)-0.118
    return lai
def EVI(B,R,NIR):
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    evi=0*np.ones(shape=(NIR.shape[0],B.shape[1]),dtype=np.float32)
    numerator=(NIR - R)
    denominator= (NIR + (6.0*R) - (7.5 * B) + 1.0)
    evi=np.where(np.abs(denominator)<Epsilon,0.0,2.5 * ((NIR - R) / (NIR + (6.0*R) - (7.5 * B) + 1.0)) )
    return evi
def EVI2(R,NIR):
    np.seterr(divide='ignore', invalid='ignore')
    evi2=0*np.ones(shape=(NIR.shape[0],NIR.shape[1]),dtype=np.float32)
    denominator=(NIR+R+1)
    evi2=(np.where(np.abs(denominator)<Epsilon,0.0,2.4*((NIR-R)/(NIR+R+1))))
    return evi2
def EVIN(B,R,NIR,L=1):
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    #rescale
    #Normalized Data
    normalizedB = ((B-B.min())/(B.max()-B.min()))
    normalizedR = ((R-R.min())/(R.max()-R.min()))
    normalizedNIR = ((NIR-NIR.min())/(NIR.max()-NIR.min()))
    #dominio di funzione
    evin=0*np.ones(shape=(NIR.shape[0],B.shape[1]),dtype=np.float32)
    numeratore=(normalizedNIR - normalizedR)
    denominator=(normalizedNIR + (6*normalizedR) - (7.5 * normalizedB) + 1)
    evin=np.where(np.abs(denominator)<Epsilon,0.0,2.5 * (numeratore / denominator))
    return evin
def SAVI(R,NIR,L=0.5):
    # Huete, A. "A Soil-Adjusted Vegetation Index (SAVI)." Remote Sensing of Environment 25 (1988): 295-309.
    # The L value is based on the amount of green vegetative cover. With VEGINDEX, L is a default of 0.5, which means, generally, areas of moderate green vegetative cover.
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    
    #LS8 = B4,B5    LS57 = B3,B4
    savi = ((NIR-R) / (NIR + R + L)) * (1+L)
    numerator=NIR-R
    denominator=NIR + R + L
    savi=np.where(np.abs(denominator)<Epsilon,0.0,savi)   
#    savi = ((NIR-R) / (NIR + R + L)) * (1+L)
#    savi = savi.astype(np.float32)
    return savi
def WDVI(R,NIR):
    wdvi=NIR-0.4*R
    return wdvi
def MSAVI(R,NIR):
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    
    #LS8 = B4,B5    LS57 = B3,B4   
    msavi = (2 * NIR + 1 - ((2*NIR+1)**2 - 8 * (NIR - R))**(1/2)) / 2
    msavi = msavi.astype(np.float32)
    return msavi
def NDMI(NIR,SWIR):
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    
    #LS8 = B5,B6    LS57 = B4,B5
    ndmi = (NIR-SWIR) / (NIR+SWIR)
    ndmi = ndmi.astype(np.float32)
    return ndmi
def NBR(NIR,SWIR):
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    
    #LS8 = B5,B7    LS57 = B4,B7
    nbr = (NIR - SWIR) / (NIR+SWIR)
    nbr = nbr.astype(np.float32)
    return nbr
def NBR2(SWIR1,SWIR2):
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    
    #LS8 = B6,B7      LS57 = B5,B7    
    nbr2 = (SWIR1 - SWIR2) / (SWIR1 + SWIR2)
    nbr2 = nbr2.astype(np.float32)
    return nbr2
def VDI(R,NIR):
    #Tucker, C. "Red and Photographic Infrared Linear Combinations for Monitoring Vegetation." Remote Sensing of Environment 8 (1979): 12-150.
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
      
    vdi = (NIR - R)
    vdi = vdi.astype(np.float32)
    return vdi
def RVI(R,NIR):
    # Birth, G., and G. McVey. "Measuring the Color of Growing Turf with a Reflectance Spectrophotometer." Agronomy Journal 60 (1968): 640-643.
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    rvi=np.where(np.abs(R)<Epsilon,0.0,(NIR) / (R))
#    rvi = (NIR / R)
#    rvi = rvi.astype(np.float32)
    return rvi
def TDVI(R,NIR):
    # Rouse, J., R. Haas, J. Schell, and D. Deering. Monitoring Vegetation Systems in the Great Plains with ERTS. Third ERTS Symposium, NASA (1973): 309-317.
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    
    tdvi = 1.5*((NIR-R)/((NIR**2+R+0.5)**(1/2)))
    tdvi = tdvi.astype(np.float32)
    return tdvi
def MSAVI2(R,NIR):
    # Qi, J. et al, 1994, "A modified soil vegetation adjusted index." Remote Sensing of Environment, Vol. 48, No. 2, 119-126.
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    msavi2=(2*NIR+1-((2 * NIR + 1)**2 - 8 * (NIR - R))**(1/2))/2.0
    sqrt_value=(2 * NIR + 1)**2 - 8 * (NIR - R)
    msavi2=(np.where(np.abs(sqrt_value)<0.0,0.0,msavi2))
    return msavi2
def GEMI(R,NIR):
    # Pinty, B., and M. Verstraete. GEMI: A Non-Linear Index to Monitor Global Vegetation from Satellites. Vegetation 101 (1992): 15-20.
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    eta=(2 * (NIR**2-R**2) + (1.5 * NIR) + (0.5 * R)) / (NIR + R + 0.5)
    
    denominator_eta=NIR + R + 0.5
    eta=(np.where(np.abs(denominator_eta)<Epsilon,0.0,eta))
    gemi = eta * (1-0.25 * eta) -((R-0.125) / (1-R))
    denominator_gemi=1-R
    gemi=(np.where(np.abs(denominator_gemi)<Epsilon,0.0,gemi))
    return gemi
def MTVI(G,R,NIR):
    # Haboudane, D. et al. "Hyperspectral Vegetation Indices and Novel Algorithms for Predicting Green LAI of Crop Canopies: Modeling and Validation in the Context of Precision Agriculture." Remote Sensing of Environment 90 (2004): 337-352.
    # Allow division by zero    
    mtvi = 1.2*(1.2*(NIR-G)-2.5*(R-G))
    return mtvi
def OSAVI(R,NIR):
    # Rondeaux, G., M. Steven and F. Baret. "Optimization of Soil-Adjusted Vegetation Indices." Remote Sensing of Environment 55 (1996): 95-107.
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    
    osavi = (NIR-R) / (NIR+R+0.16)
    osavi = osavi.astype(np.float32)
    return osavi
def MCARI2(G,R,NIR):
    # Haboudane, D. et al. "Hyperspectral Vegetation Indices and Novel Algorithms for Predicting Green LAI of Crop Canopies: Modeling and Validation in the Context of Precision Agriculture." Remote Sensing of Environment 90 (2004): 337-352.
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    
    mcari2 = (1.5 *(2.5 * (NIR-R) -1.3 *(NIR-G))) / ((2 * NIR+1)**2 - (6 * NIR - 5 * ((R)**(1/2)))- 0.5)**(1/2)
    mcari2 = mcari2.astype(np.float32)
    return mcari2                 
def MCARI(G,R,RedEdge,NIR):
    # Daughtry, C. et al. "Estimating Corn Leaf Chlorophyll Concentration from Leaf and Canopy Reflectance." Remote Sensing Environment 74 (2000): 229-239.
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    
    mcari = ((RedEdge-R)- 0.2 * (RedEdge - G)) * (RedEdge / R)
    mcari = mcari.astype(np.float32)
    return mcari
def NDRE(RedEdge,NIR):
    # Gitelson, A., and M. Merzlyak. "Spectral Reflectance Changes Associated with Autumn Senescence of Aesculus Hippocastanum L. and Acer Planoides L. Leaves." Journal of Plant Physiology 142 (1994): 286-292.
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    
    ndre = ((RedEdge-NIR)) / (RedEdge+NIR)
    ndre = ndre.astype(np.float32)
    return ndre
def MRENDVI(B,RedEdge,NIR):
    # Datt, B. "A New Reflectance Index for Remote Sensing of Chlorophyll Content in Higher Plants: Test Using Eucalyptus Leaves." Journal of Plant Physiology 154 (1999): 30-36.
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    
    mrendvi = (RedEdge-NIR) / (RedEdge+NIR -(2 * B))
    mrendvi = mrendvi.astype(np.float32)
    return mrendvi
def TCARI(G,R,RedEdge):
    # Datt, B. "A New Reflectance Index for Remote Sensing of Chlorophyll Content in Higher Plants: Test Using Eucalyptus Leaves." Journal of Plant Physiology 154 (1999): 30-36.
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    
    tcari = 3 * (RedEdge-R)-0.2 * (RedEdge-G) * (RedEdge / R)
    tcari = tcari.astype(np.float32)
    return tcari
def PSRI(B,R,RedEdge):
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    
    psri = (R- B) / RedEdge
    psri = psri.astype(np.float32)
    return psri

def NDWI2(G,NIR):
    # McFeeters (1996)
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    denominator=G+NIR
    ndwi2=(np.where(np.abs(denominator)<Epsilon,0.0,(G-NIR)/(G+NIR)))
    return ndwi2   
def NDTI(G,R):
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    
    ndti = (R-G)/(R+G)
    ndti = ndti.astype(np.float32)
    return ndti 
def SoilRI(G,R):
    # Allow division by zero
    ri = np.where(np.abs(G)<Epsilon,0.0,R*R/G*G*G)
    return ri 
def SoilCI(G,R):
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    ci = (G-R)/(G+R)
    ci = ri.astype(np.float32)
    return ri
def GreenNDVI(G,NIR):
   # Allow division by zero
   np.seterr(divide='ignore', invalid='ignore')
   
   greenndvi = (NIR-G)/(NIR+G)
   greenndvi = greenndvi.astype(np.float32)
   return greenndvi    
def CloIdx1(G,NIR):
   # Allow division by zero
   np.seterr(divide='ignore', invalid='ignore')
   
   cloidx1 = (NIR-G)/(NIR+G)
   cloidx1 = cloidx1.astype(np.float32)
   return cloidx1  
def CloIdx2(RedEdge,NIR):
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    
    cloidx2 = (NIR/RedEdge)-1
    cloidx2 = cloidx1.astype(np.float32)
    return cloidx2   
def IPVI(R,NIR):
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    ndvi=(NIR-R) / (NIR + R)
    ipvi = ((NIR/(NIR+R))/2)*(ndvi+1)
    ipvi = ipvi.astype(np.float32)
    return ipvi         
def TNDVI(R,NIR):
    ndvi=NDVI(R,NIR)
    tndvi_sqrt_val=ndvi+0.5
    tndvi=np.where(tndvi_sqrt_val<0,0.0,np.sqrt(tndvi_sqrt_val))
    return tndvi 
def TSAVI(R,NIR):
    np.seterr(divide='ignore', invalid='ignore')
    tsavi=(0.7 * (NIR - 0.7 * R - 0.9)) / (0.7 * NIR + R + 0.08 * (1.0 + 0.7 * 0.7))
    denominator=(0.7 * NIR + R + 0.08 * (1.0 + 0.7 * 0.7))
    tsavi=np.where(np.abs(denominator)<Epsilon,0.0,tsavi)
    return tsavi