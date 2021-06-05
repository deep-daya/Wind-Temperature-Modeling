# Wind-Temperature-Modeling
CS 231N Project

This repository stores data files and scripts used for the final project of cs 231N at Stanford.
The final python scripts used to generate the results in the final report and presentation
are 
1. train_monthly.py: train over the monthly NCEP/NCAR data of u,v,T over the U.S. and make plots
2. train_daily.py: train over the daily NCEP/NCAR data of u,v,T over the U.S. and make plots
3. us2_train.py: train over the daily NCEP/NCAR data of u,v,T over the U.S. plus two points outside its border and make plots

extract_us_data.py extracts from global daily NCEP/NCAR data the data over U.S., and U.S. plus two points outside its border, and write the extracted data into a new netCDF file.
