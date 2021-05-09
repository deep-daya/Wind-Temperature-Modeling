'''
This script read the monthly mean temperature, u, and v data
1. plot the x-y plane at the height of 1000 mbar (~100 m) on 
1) the global scale
2) and US only
for the year of 2000

2. Time series of all time available (01/1948 to ) using data
   at 100 m and averaged in x and y (US only)
'''

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def draw_map(lat1=-90, lat2=90, lon1=0, lon2=360, dlat=30., dlon=60., lon0=180):
    m = Basemap(llcrnrlon = lon1, urcrnrlon=lon2, llcrnrlat=lat1,urcrnrlat = lat2, projection='robin',lon_0=lon0,resolution='c')
      # resolution c, l, i, h, f in that order
    
    m.drawmapboundary(fill_color='white', zorder=-1)
    m.fillcontinents(color='0.8', lake_color='white', zorder=0)
    
    m.drawcoastlines(color='0.6', linewidth=0.5)
    m.drawcountries(color='0.6', linewidth=0.5)
    
    m.drawparallels(np.arange(lat1,lat2, dlat), labels=[1,0,0,1], dashes=[1,1], linewidth=0.25, color='0.5')
    m.drawmeridians(np.arange(lon1, lon2, dlon), labels=[1,0,0,1], dashes=[1,1], linewidth=0.25, color='0.5')
    return m

def draw_map_merc(lat1=-90, lat2=90, lon1=0, lon2=360, dlat=30., dlon=60., lon0=180):
    m = Basemap(llcrnrlon = lon1, urcrnrlon=lon2, llcrnrlat=lat1,urcrnrlat = lat2,projection='merc',lon_0=lon0,resolution='c')
      # resolution c, l, i, h, f in that order
    
    m.drawmapboundary(fill_color='white', zorder=-1)
    m.fillcontinents(color='0.8', lake_color='white', zorder=0)
    
    m.drawcoastlines(color='0.6', linewidth=0.5)
    m.drawcountries(color='0.6', linewidth=0.5)
    
    m.drawparallels(np.arange(lat1,lat2, dlat), labels=[1,0,0,1], dashes=[1,1], linewidth=0.25, color='0.5')
    m.drawmeridians(np.arange(lon1, lon2, dlon), labels=[1,0,0,1], dashes=[1,1], linewidth=0.25, color='0.5')
    return m


data = Dataset('./air.mon.mean.nc', 'r')
T = data.variables['air'][:,0]
t = data.variables['time'][:]
lat = data.variables['lat'][:]
lon = data.variables['lon'][:]
LON, LAT = np.meshgrid(lon, lat)
data.close()

data = Dataset('./uwnd.mon.mean.nc', 'r')
u = data.variables['uwnd'][:,0]
data.close()

data = Dataset('./vwnd.mon.mean.nc', 'r')
v = data.variables['vwnd'][:,0]
data.close()

dt = t[1] - t[0]
t_ind = int((2001-1948)*24*365/dt)  # the time index for 01/2000
print('t=', t[t_ind]/24/365+1800)

# extract data of the US
lat_us = np.argwhere(np.logical_and(lat > 29.06, lat<48.97))[:,0]
lon_us = np.argwhere(np.logical_and(lon > np.mod(-123.3,360),lon < np.mod(-81.2, 360)))[:,0]
lon_ind, lat_ind = np.meshgrid(lon_us, lat_us)

T_US = T[:, lat_ind, lon_ind]
u_US = u[:, lat_ind, lon_ind]
v_US = v[:, lat_ind, lon_ind]

# extract coordinates of the US
LON_US, LAT_US = np.meshgrid(lon[lon_us], lat[lat_us])

# plot x-y plane in 01/2000
plt.figure()
plt.subplot(221)
plt.contourf(LON_US,LAT_US, T_US[t_ind], cmap ='Spectral_r')
plt.ylabel('lat', fontsize=16)
plt.xlabel('lon', fontsize=16)
cbar = plt.colorbar()
cbar.set_label(r'T ($^oC$)', fontsize=16)
plt.tight_layout()

plt.subplot(222)
plt.contourf(LON_US,LAT_US, u_US[t_ind], cmap ='Spectral_r')
plt.ylabel('lat', fontsize=16)
plt.xlabel('lon', fontsize=16)
cbar = plt.colorbar()
cbar.set_label(r'u (m/s)', fontsize=16)
plt.tight_layout()

plt.subplot(223)
plt.contourf(LON_US,LAT_US, v_US[t_ind], cmap ='Spectral_r')
plt.ylabel('lat', fontsize=16)
plt.xlabel('lon', fontsize=16)
cbar = plt.colorbar()
cbar.set_label(r'v (m/s)', fontsize=16)
plt.tight_layout()

# plot time series 
u_ts = np.mean(u_US, axis=(1,2))
v_ts = np.mean(v_US, axis=(1,2))
T_ts = np.mean(T_US, axis=(1,2))

t_int = (t/365/24).astype(int) + 1800
plt.figure()
plt.subplot(311)
plt.plot(t_int, u_ts)
plt.legend()
plt.xlabel('year', fontsize=16)
plt.ylabel('u (m/s)', fontsize=16)

plt.subplot(312)
plt.plot(t_int, v_ts)
plt.legend()
plt.xlabel('year', fontsize=16)
plt.ylabel('v (m/s)', fontsize=16)

plt.subplot(313)
plt.plot(t_int, T_ts)
plt.xlabel('year', fontsize=16)
plt.ylabel(r'T ($^oC$)', fontsize=16)

# plot global data on a base map
plt.figure()
plt.subplot(221)
m = draw_map()
LON, LAT = m(LON, LAT)
m.pcolormesh(LON, LAT, T[t_ind], cmap='Spectral_r', rasterized=False, edgecolor='0.6', linewidth=0)
cbar = m.colorbar()
cbar.set_label(r'T ($^oC$)', fontsize=16)
cbar.solids.set_edgecolor("face")
#cbar.set_ticks([0,100])

plt.subplot(222)
m = draw_map()
m.pcolormesh(LON, LAT, u[t_ind], cmap='Spectral_r', rasterized=False, edgecolor='0.6', linewidth=0)
cbar = m.colorbar()
cbar.set_label('u (m/s)', fontsize=16)
cbar.solids.set_edgecolor("face")

plt.subplot(223)
m = draw_map()
m.pcolormesh(LON, LAT, v[t_ind], cmap='Spectral_r', rasterized=False, edgecolor='0.6', linewidth=0)
cbar = m.colorbar()
cbar.set_label('v (m/s)', fontsize=16)
cbar.solids.set_edgecolor("face")
plt.tight_layout()
plt.suptitle("Jan 2000", fontsize=16)

#plot US data on a map
plt.figure()
plt.subplot(221)
m = draw_map_merc(29.06, 48.97, -123.3, -81.2,5,10, (-123.3-81.2)/2)
LON, LAT = m(np.mod((LON_US+180),360)-180, LAT_US)
print('LON', np.mod((LON_US+180),360)-180)
m.pcolormesh(LON, LAT, T_US[t_ind], cmap='Spectral_r', rasterized=False, edgecolor='0.6', linewidth=0)
cbar = m.colorbar()
cbar.set_label(r'T ($^oC$)', fontsize=16)
cbar.solids.set_edgecolor("face")
#cbar.set_ticks([0,100])

plt.subplot(222)
m = draw_map_merc(29.06, 48.97, -123.3, -81.2, 5,10, (-123.3-81.2)/2)
m.pcolormesh(LON, LAT, u_US[t_ind], cmap='Spectral_r', rasterized=False, edgecolor='0.6', linewidth=0)
cbar = m.colorbar()
cbar.set_label('u (m/s)', fontsize=16)
cbar.solids.set_edgecolor("face")
#cbar.set_ticks([0,100])

plt.subplot(223)
m = draw_map_merc(29.06, 48.97, -123.3, -81.2,5,10, (-123.3-81.2)/2)
m.pcolormesh(LON, LAT, v_US[t_ind], cmap='Spectral_r', rasterized=False, edgecolor='0.6', linewidth=0)
cbar = m.colorbar()
cbar.set_label('v (m/s)', fontsize=16)
cbar.solids.set_edgecolor("face")
plt.tight_layout()
plt.suptitle("Jan 2000", fontsize=16)



plt.show()


