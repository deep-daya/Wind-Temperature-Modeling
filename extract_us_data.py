'''
This script read the daily temperature, u, and v data
1. plot the x-y plane at the height of 1000 mbar (~100 m) on
1) the global scale
2) and US only
3) U.S.+ surrounding
for the year of 2000
2. Time series of all time available (01/1948 to ) using data
   at 100 m and averaged in x and y (US only)
'''

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap

#def draw_map(lat1=-90, lat2=90, lon1=0, lon2=360, dlat=30., dlon=60., lon0=180):
#    m = Basemap(llcrnrlon = lon1, urcrnrlon=lon2, llcrnrlat=lat1,urcrnrlat = lat2, projection='robin',lon_0=lon0,resolution='c')
#      # resolution c, l, i, h, f in that order
#
#    m.drawmapboundary(fill_color='white', zorder=-1)
#    m.fillcontinents(color='0.8', lake_color='white', zorder=0)
#
#    m.drawcoastlines(color='0.6', linewidth=0.5)
#    m.drawcountries(color='0.6', linewidth=0.5)
#
#    m.drawparallels(np.arange(lat1,lat2, dlat), labels=[1,0,0,1], dashes=[1,1], linewidth=0.25, color='0.5')
#    m.drawmeridians(np.arange(lon1, lon2, dlon), labels=[1,0,0,1], dashes=[1,1], linewidth=0.25, color='0.5')
#    return m
#
#def draw_map_merc(lat1=-90, lat2=90, lon1=0, lon2=360, dlat=30., dlon=60., lon0=180):
#    m = Basemap(llcrnrlon = lon1, urcrnrlon=lon2, llcrnrlat=lat1,urcrnrlat = lat2,projection='merc',lon_0=lon0,resolution='c')
#      # resolution c, l, i, h, f in that order
#
#    m.drawmapboundary(fill_color='white', zorder=-1)
#    m.fillcontinents(color='0.8', lake_color='white', zorder=0)
#
#    m.drawcoastlines(color='0.6', linewidth=0.5)
#    m.drawcountries(color='0.6', linewidth=0.5)
#
#    m.drawparallels(np.arange(lat1,lat2, dlat), labels=[1,0,0,1], dashes=[1,1], linewidth=0.25, color='0.5')
#    m.drawmeridians(np.arange(lon1, lon2, dlon), labels=[1,0,0,1], dashes=[1,1], linewidth=0.25, color='0.5')
#    return m


dir =  '../'  #'content/drive/MyDrive/CS231N-Project/'
data = Dataset(dir + 'air_final_1.nc','r')
wf_name = 'ncep_data/US_daily.nc'

T2 = data.variables['air'][:,0]-273  # data at the first pressure level, equivalent to 100 m
t = data.variables['time'][:]
print('t min=', t[0]/24/365+1800, 'max=', t[-1]/24/365+1800)
lat = data.variables['lat'][:]
lon = data.variables['lon'][:]
LON, LAT = np.meshgrid(lon, lat)
data.close()

data = Dataset(dir+'air_final.nc', 'r')
T1 = data.variables['air'][:,0]-273
data.close()

data = Dataset(dir+'uwnd_final_19xx.nc', 'r')
u1 = data.variables['uwnd'][:,0]
data.close()


data = Dataset(dir+'uwnd_final_20xx.nc', 'r')
u2 = data.variables['uwnd'][:,0]

data = Dataset(dir+'vwnd_final_19xx.nc', 'r')
v1 = data.variables['vwnd'][:,0]

data = Dataset(dir+'vwnd_final_20xx.nc', 'r')
v2 = data.variables['vwnd'][:,0]


# concatenate data along the time dimension
T = np.concatenate((T1, T2), axis=0)
u = np.concatenate((u1, u2), axis=0)
v = np.concatenate((v1, v2), axis=0)

T = T2

dt = t[1] - t[0]
print('dt=',dt)
t_ind = int((2001-1948)*24*365/dt)  # the time index for 01/2000
print('t=', t[t_ind]/24/365+1800)
print('T shape', T.shape[0])

# extract data of the US
lat_us = np.argwhere(np.logical_and(lat > 29.06, lat<48.97))[:,0]
lon_us = np.argwhere(np.logical_and(lon > np.mod(-123.3,360),lon < np.mod(-81.2, 360)))[:,0]
lon_ind, lat_ind = np.meshgrid(lon_us, lat_us)

T_US = T[:, lat_ind, lon_ind]
u_US = u[:, lat_ind, lon_ind]
v_US = v[:, lat_ind, lon_ind]

print('****nan?', np.any(np.isnan(T_US)))
print('****nan?', np.any(np.isnan(u_US)))
print('****nan?', np.any(np.isnan(v_US)))

# extract coordinates of the US
LON_US, LAT_US = np.meshgrid(lon[lon_us], lat[lat_us])

# extract data of the US plus two points outside its border
print('lat', lat_us)
print('lon', lon_us)
lat_us2 = np.insert(lat_us, 0, [np.min(lat_us)-2, np.min(lat_us)-1])
lat_us2 = np.append(lat_us2, [np.max(lat_us)+1, np.max(lat_us)+2])
lon_us2 = np.insert(lon_us, 0, [np.min(lon_us)-2, np.min(lon_us)-1])
lon_us2 = np.append(lon_us2, [np.max(lon_us)+1, np.max(lon_us)+2])
print('lat2', lat_us2)
print('lon2', lon_us2)
lon_ind2, lat_ind2 = np.meshgrid(lon_us2, lat_us2)

T_US2 = T[:, lat_ind2, lon_ind2]
u_US2 = u[:, lat_ind2, lon_ind2]
v_US2 = v[:, lat_ind2, lon_ind2]

print('****nan?', np.any(np.isnan(T_US2)))
print('****nan?', np.any(np.isnan(u_US2)))
print('****nan?', np.any(np.isnan(v_US2)))

# extract coordinates of the US plus surrounding 2 points
LON_US2, LAT_US2 = np.meshgrid(lon[lon_us2], lat[lat_us2])

# Write to file
wfile = Dataset(wf_name, "w", format="NETCDF4")
wfile.createDimension("time", None)
wfile.createDimension("lat", T_US.shape[1])
wfile.createDimension("lon", T_US.shape[2])
wfile.createDimension("lat2", len(lat_us2))
wfile.createDimension("lon2", len(lon_us2))

Temp = wfile.createVariable("T", float, ("time",'lat', "lon"))
U = wfile.createVariable("u", float, ("time",'lat', "lon"))
V= wfile.createVariable("v", float, ("time",'lat', "lon"))

Temp2 = wfile.createVariable("T2", float, ("time",'lat2', "lon2"))
U2 = wfile.createVariable("u2", float, ("time",'lat2', "lon2"))
V2= wfile.createVariable("v2", float, ("time",'lat2', "lon2"))

latitude = wfile.createVariable("lat", float, ('lat'))
longitude = wfile.createVariable("lon", float, ('lon2'))
latitude2 = wfile.createVariable("lat2", float, ('lat2'))
longitude2 = wfile.createVariable("lon2", float, ('lon2'))

wfile.close()

wfile = Dataset(wf_name, "r+")
wfile['T'][:] = T_US
wfile['u'][:] = u_US
wfile['v'][:] = v_US
wfile['lat'][:] = lat[lat_us]
wfile['lon'][:] = lon[lon_us]

wfile['T2'][:] = T_US2
wfile['u2'][:] = u_US2
wfile['v2'][:] = v_US2
wfile['lat2'][:] = lat[lat_us2]
wfile['lon2'][:] = lon[lon_us2]
wfile.close()

# read the second US data to check
wfile = Dataset(wf_name, "r")
T_US2 = data.variables['T2'][:]
u_US2 = data.variables['u2'][:]
v_US2 = data.variables['v2'][:]
wfile.close()

# plot global map
plt.figure()
plt.subplot(221)
plt.contourf(LON,LAT, T[t_ind], cmap ='Spectral_r')
plt.ylabel('lat', fontsize=16)
plt.xlabel('lon', fontsize=16)
cbar = plt.colorbar()
cbar.set_label(r'T ($^oC$)', fontsize=16)
plt.tight_layout()

plt.subplot(222)
plt.contourf(LON,LAT, u[t_ind], cmap ='Spectral_r')
plt.ylabel('lat', fontsize=16)
plt.xlabel('lon', fontsize=16)
cbar = plt.colorbar()
cbar.set_label(r'u (m/s)', fontsize=16)
plt.tight_layout()

plt.subplot(223)
plt.contourf(LON,LAT, v[t_ind], cmap ='Spectral_r')
plt.ylabel('lat', fontsize=16)
plt.xlabel('lon', fontsize=16)
cbar = plt.colorbar()
cbar.set_label(r'v (m/s)', fontsize=16)
plt.tight_layout()
plt.savefig('global_daily_map.png')


# plot x-y plane in 01/2000
plt.figure()
plt.subplot(221)
plt.contourf(LON_US2,LAT_US2, T_US2[t_ind], cmap ='Spectral_r')
plt.ylabel('lat', fontsize=16)
plt.xlabel('lon', fontsize=16)
cbar = plt.colorbar()
cbar.set_label(r'T ($^oC$)', fontsize=16)
plt.tight_layout()

plt.subplot(222)
plt.contourf(LON_US2,LAT_US2, u_US2[t_ind], cmap ='Spectral_r')
plt.ylabel('lat', fontsize=16)
plt.xlabel('lon', fontsize=16)
cbar = plt.colorbar()
cbar.set_label(r'u (m/s)', fontsize=16)
plt.tight_layout()

plt.subplot(223)
plt.contourf(LON_US2,LAT_US2, v_US2[t_ind], cmap ='Spectral_r')
plt.ylabel('lat', fontsize=16)
plt.xlabel('lon', fontsize=16)
cbar = plt.colorbar()
cbar.set_label(r'v (m/s)', fontsize=16)
plt.tight_layout()

plt.savefig('us_daily_map2.png')
#
# plot time series
# u_ts = np.mean(u_US, axis=(1,2))
# v_ts = np.mean(v_US, axis=(1,2))
# T_ts = np.mean(T_US, axis=(1,2))

# t_int = (t/365/24).astype(int) + 1800
# plt.figure()
# plt.subplot(311)
# plt.plot(t_int, u_ts)
# plt.legend()
# plt.xlabel('year', fontsize=16)
# plt.ylabel('u (m/s)', fontsize=16)

# plt.subplot(312)
# plt.plot(t_int, v_ts)
# plt.legend()
# plt.xlabel('year', fontsize=16)
# plt.ylabel('v (m/s)', fontsize=16)

# plt.subplot(313)
# plt.plot(t_int, T_ts)
# plt.xlabel('year', fontsize=16)
# plt.ylabel(r'T ($^oC$)', fontsize=16)
# plt.show()

## plot global data on a base map
#plt.figure()
#plt.subplot(221)
#m = draw_map()
#LON, LAT = m(LON, LAT)
#m.pcolormesh(LON, LAT, T[t_ind], cmap='Spectral_r', rasterized=False, edgecolor='0.6', linewidth=0)
#cbar = m.colorbar()
#cbar.set_label(r'T ($^oC$)', fontsize=16)
#cbar.solids.set_edgecolor("face")
##cbar.set_ticks([0,100])
#
#plt.subplot(222)
#m = draw_map()
#m.pcolormesh(LON, LAT, u[t_ind], cmap='Spectral_r', rasterized=False, edgecolor='0.6', linewidth=0)
#cbar = m.colorbar()
#cbar.set_label('u (m/s)', fontsize=16)
#cbar.solids.set_edgecolor("face")
#
#plt.subplot(223)
#m = draw_map()
#m.pcolormesh(LON, LAT, v[t_ind], cmap='Spectral_r', rasterized=False, edgecolor='0.6', linewidth=0)
#cbar = m.colorbar()
#cbar.set_label('v (m/s)', fontsize=16)
#cbar.solids.set_edgecolor("face")
#plt.tight_layout()
#plt.suptitle("Jan 2000", fontsize=16)
#
##plot US data on a map
#plt.figure()
#plt.subplot(221)
#m = draw_map_merc(29.06, 48.97, -123.3, -81.2,5,10, (-123.3-81.2)/2)
#LON, LAT = m(np.mod((LON_US+180),360)-180, LAT_US)
#print('LON', np.mod((LON_US+180),360)-180)
#m.pcolormesh(LON, LAT, T_US[t_ind], cmap='Spectral_r', rasterized=False, edgecolor='0.6', linewidth=0)
#cbar = m.colorbar()
#cbar.set_label(r'T ($^oC$)', fontsize=16)
#cbar.solids.set_edgecolor("face")
##cbar.set_ticks([0,100])
#
#plt.subplot(222)
#m = draw_map_merc(29.06, 48.97, -123.3, -81.2, 5,10, (-123.3-81.2)/2)
#m.pcolormesh(LON, LAT, u_US[t_ind], cmap='Spectral_r', rasterized=False, edgecolor='0.6', linewidth=0)
#cbar = m.colorbar()
#cbar.set_label('u (m/s)', fontsize=16)
#cbar.solids.set_edgecolor("face")
##cbar.set_ticks([0,100])
#
#plt.subplot(223)
#m = draw_map_merc(29.06, 48.97, -123.3, -81.2,5,10, (-123.3-81.2)/2)
#m.pcolormesh(LON, LAT, v_US[t_ind], cmap='Spectral_r', rasterized=False, edgecolor='0.6', linewidth=0)
#cbar = m.colorbar()
#cbar.set_label('v (m/s)', fontsize=16)
#cbar.solids.set_edgecolor("face")
#plt.tight_layout()
#plt.suptitle("Jan 2000", fontsize=16)
#
#
#
#plt.savefig('us_daily.png')
