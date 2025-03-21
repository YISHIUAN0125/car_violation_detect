from haversine import haversine, Unit
import numpy as np

#Taoyuan city (121.2168, 24.93759)

pt1 = (24.93759, 90-121.2168) # (lat, lon)
pt2 = (24.93759, 90-122.2168)
pt3 = (24.93759,   90-121.2168)
pt4 = (25.93759, 90-121.2168)

print(haversine(pt1, pt2, unit=Unit.METERS))
print(haversine(pt3, pt4, unit=Unit.METERS))

'''With haversine formula'''

def distance(pt1, pt2):
    pt1 = (np.deg2rad(pt1[0]), np.deg2rad(90-pt1[1]))
    pt2 = (np.deg2rad(pt2[0]), np.deg2rad(90-pt2[1]))
    
    d_square = np.sin((pt2[0]-pt1[0])/2)**2 + np.cos(pt2[0])*np.cos(pt1[0])*np.sin((pt2[1]-pt1[1])/2)**2
    d = 2*6371004*np.arcsin(np.sqrt(d_square))

    return d

print('with formula:')
print('lon:', distance(pt1, pt2))
print('lat', distance(pt3, pt4))