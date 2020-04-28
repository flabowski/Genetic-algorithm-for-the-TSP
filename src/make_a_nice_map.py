# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 12:36:34 2020

@author: Florian
sources:
    https://stackoverflow.com/a/50907364/6747238
    https://stackoverflow.com/a/48056459/6747238
    https://simplemaps.com/data/world-cities
    https://www.schemecolor.com/google-map-basic-colors.php
"""
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
import pickle
from descartes import PolygonPatch
import numpy as np
#%matplotlib notebook


shapename = 'admin_0_countries'
countries_shp = shpreader.natural_earth(resolution='110m',
                                        category='cultural', name=shapename)
fig, ax = plt.subplots()
ax.set_facecolor('#aadaff')

for country in shpreader.Reader(countries_shp).records():
#    print(country.attributes['NAME_LONG'])
    poly = country.geometry.__geo_interface__
    ax.add_patch(PolygonPatch(poly, fc='#c3ecb2', ec='#000000',
                              alpha=1.0, zorder=2))
ax.axis('scaled')
plt.show()

with open('../doc/world_map.pkl', 'wb') as fid:
    pickle.dump(ax, fid)

#  make file with latitude and longitude
path = "../doc/"
city_names = ["Barcelona", "Belgrade", "Berlin", "Brussels", "Bucharest",
              "Budapest", "Copenhagen", "Dublin", "Hamburg", "Istanbul",
              "Kyiv", "London", "Madrid", "Milan", "Moscow", "Munich",
              "Paris", "Prague", "Rome", "Saint Petersburg", "Sofia",
              "Stockholm", "Vienna", "Warsaw"]
latlon = np.empty((len(city_names), 2))
latlon_txt = np.genfromtxt(path+"worldcities.txt", dtype="<U256",
                           delimiter="\t", skip_header=1)
for i, cn in enumerate(city_names):
    ll = latlon_txt[latlon_txt[:, 0] == cn][0]
    latlon[i] = [np.float64(ll[1]), np.float64(ll[2])]
np.savetxt("../doc/latitude_lonitude.txt", latlon, delimiter=";",
           header=';'.join(city_names))
