; This file is part of the MJO_teleconnection module of the MDTF code package (see LICENSE.txt)

;-------------------------------------------------------------------------------------
;Calculate RMM indices for a given  model data
;Projects model data onto reanalysis MJO EOFs
;RMM indices based on Wheeler and Hendon (2004)
;EOF patterns downloaded from: http://cawcr.gov.au/staff/mwheeler/maproom/RMM/eof1and2.htm
;
;Model time domain: 1861-2005.  Only 1950 - 2005 used
;Regional domain: 15N - 15S 
;RMM indices use 250mb zonal wind and NOT 200mb like in reanalysis (200mb not available)
;
;June 22, 2015, updated from RMMcalc.ncl for Libby's data (extra models) March 16, 2016
;Her data has a separate file per level per variable of each model
;Written by Stephanie Henderson
;Jan 10 2018
;Modified by Bohar Singh for MDTF Diagnostic Package 
;Data units: u250,u850: m/s   olr: W/m2
;-------------------------------------------------------------------------------------

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl" 

setfileoption("nc", "Format", "LargeFile")

;--------------Choose Model--------------
;previously used models (already have RMM indices for all of these)
;As Multimodel daignostic the RMM following model are already calculated and will be used in diagnostic  
;Model = "CNRM-CM5"      
;Model = "MRI-CGCM3"     ;don't have omega or olr_250 for this one
;Model = "MIROC5"        ;don't have 250mb OLR for this one
;Model = "IPSL-CM5A-MR"  ;don't have olr_250 for this one
;Model = "GFDL-ESM2M"  
;Model = "GFDL-ESM2G"	 ;noleap
;Model = "GFDL-CM3"       ;noleap.  olr only goes through 2004
;Model = "NorESM1-M"	 ;noleap
;Model = "bcc-csm1-1"
;Model = "CanESM2"	 ;365_day
 Model = "ERA5"

print("---------------------------------------------------------------------------------")

print("Model: " + Model)

;------------ Read in zonal wind and olr --------------------

diri1   = "/global/homes/l/linyaoly/ERA5/reanalysis/"         ;Observed Input data directory
diro    = "/global/homes/l/linyaoly/ERA5/reanalysis/"
filolr  = "/pscratch/sd/l/linyaoly/ERA5/Unet/reanalysis/ERA5.olr.6hr1979to2022.nc"          ; input file
filu850 = "/pscratch/sd/l/linyaoly/ERA5/Unet/reanalysis/ERA5.u850.6hr1979to2022.nc"          ; input file
filu250 = "/pscratch/sd/l/linyaoly/ERA5/Unet/reanalysis/ERA5.u200.6hr1979to2022.nc"         ; input file
; filolr  = "/pscratch/sd/l/linyaoly/ERA5/Unet/reanalysis/ERA5.olr.day.nc"          ; input file
; filu850 = "/pscratch/sd/l/linyaoly/ERA5/Unet/reanalysis/ERA5.u850.day.nc"          ; input file
; filu250 = "/pscratch/sd/l/linyaoly/ERA5/Unet/reanalysis/ERA5.u200.day.nc"         ; input file

olrvar = "olr"
u250var = "u200"
u850var = "u850"

print("Input data path(obs) :" + diri1)
print("Output data path :" + diro)
print("Input data file for OLR :" + filolr)
print("Input data file for U850:" + filu850)
print("Input data file for U250:" + filu250)
print("---------------------------------------------------------------------------------")

   ymdStrt = tointeger("1979"+"0101")                         ; start yyyymmdd
   ymdLast = tointeger("2021"+"1231") 
   yrStrt  = ymdStrt/10000
   yrLast  = ymdLast/10000
;---------------------------------------------------------------------------------------
print("Input data starts:" + yrStrt)

;-- Reading U250 data --
    a  = addfile (filu250 , "r")      
  ;  print(a)
     latt = a->lat
    ; print(latt)
     lat_ind = ind(latt .le. 15.5 .and. latt .ge. -15.5)  ;22 latitudes

    if(typeof(latt) .ne. "float")
      lat = tofloat(latt(lat_ind))
    else
      lat = latt(lat_ind)
    end if

    ; print(lat)
    
    lont = a->lon
 
    if(typeof(lont) .ne. "float")
      lon = tofloat(lont)
    else
      lon = lont
    end if
 
   ;--- read in time ---
   temp_time = tointeger(a->time)
   temp_time@units = "hours since 1979-01-01 00:00:00"
  ;  temp_time@units = "days since 1979-01-01 00:00:0.0"
   temp_time@calendar = "proleptic_gregorian"
   print(temp_time(1:10))
   calend = temp_time@calendar
   print(calend)
  ;  print("here" + temp_time@calendar)

  if(calend .ne. "360_day")  ;HadGEM2-CC uses a 360 day calendar
    ; print("here1")
    utc_date = cd_calendar(temp_time, 0)
    ymd     = cd_calendar(temp_time, -2)            ; yyyymmdd
    ; print("ymd:" + ymd)
    yeart   = tointeger(utc_date(:,0))
    montht  = tointeger(utc_date(:,1))
    dayst   = tointeger(utc_date(:,2))
    iStrt   = min(ind(ymd.eq.ymdStrt))             ; index start
    iLast   = max(ind(ymd.eq.ymdLast))              ; index last 
    print("CALENDAR DEBUG")   
    printMinMax(ymd,0)
    print("ymdStrt: "+ymdStrt+" iStrt: "+iStrt+" ymdLast: "+ymdLast+" iLast: "+iLast)
    print("Date range from year "+yeart(iStrt)+" to "+yeart(iLast))

    u250t = short2flt(a->$u250var$(iStrt:iLast,lat_ind,:))     ;time,lev,lat,lon , 15N - 15S
    print("type of u250t: " + typeof(u250t))
    u250t@_FillValue = -999

    ; delete(temp_time)
    delete(yeart)
    delete(montht)
    delete(dayst)
  ;***********************************************************
  ; Read user specified time and create required yyyyddd                    
  ;***********************************************************
    time1    = temp_time(iStrt:iLast)             ; time:units = "hours since"
    delete(temp_time)
    TIME    = cd_calendar(time1, 0)             ; type float 

    year    = tointeger( TIME(:,0) )
    month   = tointeger( TIME(:,1) )
    days     = tointeger( TIME(:,2) ) 
    
  ;***********************************************************
    print("Starting date:"+" " +  "Year:"+ year(0) + "  " + "Month:"+ month(0) + "  " + "Day:"+ days(0))
    print("End date:"+"      " +  "Year:"+ year(dimsizes(year)-1) + "  " + "Month:"+ month(dimsizes(year)-1) + " " + "Day:"+ days(dimsizes(year)-1))
    
    ;clean up
    delete(utc_date)
  end if
   
  year@calendar = "gregorian"
  doy = day_of_year(year,month,days)
  time = tointeger(year * 1000) + tointeger(doy)

  ; print("doy: " + min(doy))

  print("First doy: " + min(time) + "  last doy: " + max(time))
  print("Total days: " + dimsizes(year))  ;20440 (4950 DJF) days
  print("-------------------------------------------")
  delete(year)
  delete(month)
  delete(days)
  delete(doy)
print("Reading data for 250mb wind")

  u250 = u250t(:,:,:); Reading data over the all avaliable time period
  u250@_FillValue = default_fillvalue("float")

delete(u250t)
print("U250 data has been read")
;----------- 850mb wind --------------
print("Reading data for 850mb wind")
;-------------------------------------

  b = addfile(filu850,"r")
 
 ;-check lats, lons, and time -
 
  latt2 = b->lat
  lont2 = b->lon
  temp_time = tointeger(b->time)
  temp_time@units = "hours since 1979-01-01 00:00:00"
  temp_time@calendar = "proleptic_gregorian"
  time2 = temp_time(iStrt:iLast)
;print(dimsizes(time2))
;print(dimsizes(time))
;print(time2(0))
;print(time(0))

  if(latt2(0) .eq. latt(0) .and. (dimsizes(latt2) .eq. dimsizes(latt)))
    print("850mb wind: latitudes match")
  else
    print("850mb wind: latitudes don't match")
    exit
  end if
  if(lont2(0) .eq. lont(0).and. (dimsizes(lont2) .eq. dimsizes(lont)))
    print("850mb wind: longitudes match")
  else
    print("850mb wind: longitudes don't match")
    exit
  end if
  if(time2(0) .eq. time1(0).and.(dimsizes(time2) .eq. dimsizes(time1)))
    print("850mb wind: times match")
  else
    print("850mb wind: times don't match")
    exit
  end if
  delete(latt2)
  delete(lont2)
  delete(time2)
 
 ;-read in data-
  u850 = short2flt(b->$u850var$(iStrt:iLast,lat_ind,:))     ;time,lat,lon , 15N - 15S
  u850@_FillValue = default_fillvalue("float")
print("U850 data has been read")
print("-------------------------------------------")

   delete(b)

;-- olr --
print("Now reading OLR data")

  
  b = addfile(filolr,"r")

 ;-check lats, lons, and time -
 
  latt2 = b->lat
  lont2 = b->lon
  temp_time = tointeger(b->time)
  temp_time@units = "hours since 1979-01-01 00:00:00"
  temp_time@calendar = "proleptic_gregorian"
  time2 = temp_time(iStrt:iLast)


  if(latt2(0) .eq. latt(0) .and. (dimsizes(latt2) .eq. dimsizes(latt)))
    print("olr: latitudes match")
  else
    print("olr: latitudes don't match")
    exit
  end if
  if(lont2(0) .eq. lont(0).and.(dimsizes(lont2) .eq. dimsizes(lont)))
    print("olr: longitudes match")
  else
    print("olr: longitudes don't match")
    exit
  end if
  
 if(time2(0) .eq. time1(0).and. (dimsizes(time2) .eq. dimsizes(time1)))
    print("olr: times match")
  else
    print("olr: times don't match")
  end if  

  delete(latt2)
  delete(lont2)
  delete(latt)
  delete(lont)
  delete(time2)
  delete(time1)

  
 ;-read in data-
    olr = short2flt(b->$olrvar$(iStrt:iLast,lat_ind,:))     ;time,lat,lon , 15N - 15S
    olr = -olr/3600.
    olr@_FillValue = default_fillvalue("float")

  ;clean up
  delete(b)
  delete(lat_ind)
print("OLR data has been read")
print("olr "+olr(0,0,0))
print("-------------------------------------------")

;-------------------------------------------------------------------------
;------ remove mean of previous 120 days ------
print("remove interannual variability")
;-approximately removes interannual variability
;-not calculated for first 120 days (which are not used so will be tossed)
;-------------------------------------------------------------------------
;first find the interannual mean.  start on day 120
 
 u200_int = new(dimsizes(u250),typeof(u250),u250@_FillValue)
 u850_int = new(dimsizes(u850),typeof(u850),u850@_FillValue)
 olr_int =  new(dimsizes(olr),typeof(olr),olr@_FillValue)

;printVarSummary(u200_int)

;prescribe an interannual mean of zero for first 120 days, which will get tossed
 
 u200_int(:120*4-1,:,:) = 0.0 
 u850_int(:120*4-1,:,:) = 0.0 
 olr_int(:120*4-1,:,:) = 0.0 

 u200_int@_FillValue = default_fillvalue("float")
 u850_int@_FillValue = default_fillvalue("float")
 olr_int@_FillValue = default_fillvalue("float")

 do t = 120*4,dimsizes(time)-1
  l = t - 120*4   ;beginning index of the previous 120 days
    u200_int(t,:,:) = dim_avg_n(u250(l:t,:,:),0)
    u850_int(t,:,:) = dim_avg_n(u850(l:t,:,:),0)
    olr_int(t,:,:) = dim_avg_n(olr(l:t,:,:),0)  
    delete(l)
 end do

;remove interannual
 u200_p = u250 - u200_int
 u850_p = u850 - u850_int
 olr_p = olr - olr_int

 u200_p@_FillValue = default_fillvalue("float")
 u850_p@_FillValue = default_fillvalue("float")
 olr_p@_FillValue = default_fillvalue("float")
 print("Interannual variability has been removed")


;clean up
 delete(u200_int)
 delete(u850_int)
 delete(olr_int)
 delete(u250)
 delete(u850)
 delete(olr)

 u200_p!0 = "time"
 u200_p!1 = "lat" 
 u200_p!2 = "lon" 

 u850_p!0 = "time"
 u850_p!1 = "lat" 
 u850_p!2 = "lon"

 olr_p!0 = "time"
 olr_p!1 = "lat" 
 olr_p!2 = "lon"


print("-------------------------------------------")
;------ remove seasonal cycle ------
print("remove seasonal cycle")
print("calculate raw climatology then smooth it")

if(any(ismissing(u200_p)))
  print("u200 missing values")
else
  print("no missing values")
end if

if(any(ismissing(u850_p)))
  print("u850 missing values")
else
  print("no missing values")
end if

if(any(ismissing(olr_p)))
  print("olr missing values")
else
  print("no missing values")
end if

;200mb wind
 rawclim = clmDayTLL(u200_p,time)   ; (120:dimsizes(time)-1)
 print("check rawclim wind: "+rawclim(0:4,0,0))
 print("rawclim size: " + dimsizes(rawclim))
 smoothclim = smthClmDayTLL(rawclim,3)  ;remove first 3 harmonics
 print("smoothclim for u200"+dimsizes(smoothclim)) 

;print(rawclim(:,10,10) + "     " + smoothclim(:,10,10))

;remove seasonal cycle
 u200_anom = calcDayAnomTLL(u200_p,time,smoothclim) ;remove seasonal 
 delete(rawclim) 
 delete(smoothclim)
 delete(u200_p)

;850mb wind
 rawclim = clmDayTLL(u850_p,time)
 smoothclim = smthClmDayTLL(rawclim,3)

 u850_anom = calcDayAnomTLL(u850_p,time,smoothclim)
 delete(rawclim) 
 delete(smoothclim)

;OLR
 rawclim = clmDayTLL(olr_p,time)
 smoothclim = smthClmDayTLL(rawclim,3)
 print("check rawclim olr: "+rawclim(0:4,0,0))
 print("rawclim size: " + dimsizes(rawclim))
 
 olr_anom = calcDayAnomTLL(olr_p,time,smoothclim)
 print("olr_anom size: " + dimsizes(olr_anom))
 
 delete(rawclim) 
 delete(smoothclim)
print("seasonal cycle has been removed")
print("-------------------------------------------")

;print(olr_p(150:200,10,10) + "    " + olr_anom(150:200,10,10))

;- average the data latitudes
print("averaging the data over 15S-15N")
u200_anom@_FillValue = default_fillvalue("float")
u850_anom@_FillValue = default_fillvalue("float")
olr_anom@_FillValue = default_fillvalue("float")

u200 = dim_avg_n(u200_anom,1)  ;time,lon
u850 = dim_avg_n(u850_anom,1)
olr = dim_avg_n(olr_anom,1)
print("Done with latitudinal averaging")
print("-------------------------------------------")
;clean up
delete(u200_anom)
delete(u850_anom)
delete(olr_anom)

;- interpolate longitudes to be of size 144 to match RMM EOF data
print("interpolate data to be of size 144 to match observed RMM EOF data")

u200@_FillValue = default_fillvalue("float")
u850@_FillValue = default_fillvalue("float")
olr@_FillValue = default_fillvalue("float")

u200!1 = "lon" 
u850!1 = "lon" 
olr!1 = "lon" 

u200&lon = lon
u850&lon = lon
olr&lon = lon

newlon = fspan(0,357.5,144)

u200_int = linint1(u200&lon,u200,True,newlon,0)
u850_int = linint1(u850&lon,u850,True,newlon,0)
olr_int = linint1(olr&lon,u200,True,newlon,0)
print("Interpolation completed")
print("-------------------------------------------")
;clean up
delete(u200)
delete(u850)
delete(olr)
delete(lon)

lon = newlon
delete(newlon)

dim_lon = 144
dl = dim_lon * 2   ;288
dl2 = dim_lon * 3  ;432

;- combine all three fields into one array
RMM_field = new((/dimsizes(time),dl2/),typeof(olr_int),olr_int@_FillValue)
RMM_field(:,:dim_lon-1) = olr_int
RMM_field(:,dim_lon:dl-1) = u850_int
RMM_field(:,dl:dl2-1) = u200_int

RMM_field@_FillValue = default_fillvalue("float")

;printVarSummary(RMM_field)
;clean up
delete(olr_int)
delete(u850_int)
delete(u200_int)
delete(dl)

;------ read in RMM EOF data ------
print("read in EOF data")
;Eigenvectors
ei_vectors  = readAsciiTable(diri1+"WH04_EOFstruc.txt",2,"float",(/9,434/)) ;ignore first 9 rows and last 434 rows (normalization numbers are at the end.  read in separately).

ei_vectors@_FillValue=default_fillvalue("float")  ;[432,2]
;print(ei_vectors)

;Eigenvalues: 55.4% and 52.6%
ei_valuest = readAsciiTable(diri1+"WH04_EOFstruc.txt",2,"float",(/4,870/))
ei_values = ei_valuest(0,:)

delete(ei_valuest)

;Normalization factor
Normt = readAsciiTable(diri1+"WH04_EOFstruc.txt",1,"float",(/443/))
Norm = Normt(:,0)
delete(Normt)

;Divide RMM field by the normalization factor (global std dev of each field)
do j = 0,dl2-1
  RMM_field(:,j) = RMM_field(:,j)/Norm(j)
end do

delete(Norm)

;------ compute RMM1 and RMM2 ------
print("compute RMMs")
;- Project model data onto reanalysis EOFs.  Projecting is simply multiplying the data by the modes and summing over time.

     pc = new((/dimsizes(time),2/),typeof(RMM_field),RMM_field@_FillValue)

pc@_FillValue = default_fillvalue("float")
ei_vectors@_FillValue = default_fillvalue("float")
     
do m = 0,1
 do t = 0,dimsizes(time)-1
   pc(t,m) = 0.0

   do j=0,dl2-1
      ;print(pc(t,m) + ei_vectors(j,m) * RMM_field(t,j))
      pc(t,m) = pc(t,m) + ei_vectors(j,m) * RMM_field(t,j)
   end do
  end do
end do

;- normalize by EOF-calculated std dev
 do m = 0,1
   pc(:,m) = pc(:,m)/sqrt(ei_values(m))
 end do

;--change due to Eric's suggestion --
;instead, normalize by the standard deviation of each model so that std dev of each RMM index is unity
do m = 0,1
   pc(:,m) = dim_standardize_n(pc(:,m),0,0)
end do


;------ calculate MJO phase ------
print("calculate MJO phase")
phase = (/ 202.5, 247.5, 292.5, 337.5,  22.5,  67.5, 112.5, 157.5 /) ; phi: Matt Wheeler's f77 code

  nPhase      = dimsizes(phase)
  phaBnd      = 180.0/nPhase
  angBnd      = new( (/2,nPhase/), "float")
  angBnd(0,:) = phase-phaBnd 
  angBnd(1,:) = phase+phaBnd
 ;print("phase: "+phase+"   angBnd: "+angBnd(0,:)+" - "+angBnd(1,:))

  r2d         = 180./(4.*atan(1.0))
  ang         = atan2(pc(:,1),pc(:,0))*r2d     ; phase space 
  nn          = ind(ang.lt.0)
  ang(nn)     = ang(nn) + 360          ; make 0 to 360
  ;print(ang)

delete(phase)
delete(nPhase)
delete(r2d)
delete(nn)
delete(phaBnd)

phase = new((/dimsizes(ang)/),integer)

do t = 0,dimsizes(ang)-1
 do i=0,7
  if(ang(t) .ge. angBnd(0,i) .and. ang(t) .le. angBnd(1,i))
    phase(t) = i+1
  end if
 end do
end do

;print("Angle: " + ang + "     Phase:" + phase)

delete(ang)
delete(angBnd)

;------calculate amplitude------
print("calculate amplitude")
amp = sqrt(pc(:,0)^2. + pc(:,1)^2.) 

;print("RMM1: " + pc(:,0) + "   RMM2: " + pc(:,1) + "   phase: " + phase + "   amplitude: " + amp)

;------write output file------

header = (/"Time    RMM1    RMM2    phase    amplitude"/)
hlist = [/header/]

alist = [/time,pc(:,0),pc(:,1),phase,amp/]


 write_table(diro+Model+"_RMMsv3.txt", "w", hlist, "%s ")
 write_table(diro+Model+"_RMMsv3.txt", "a", alist, "%07i, %7.7f, %7.7f, %01i, %7.7f")

print("Data is saved into txt format at:"+diro)
print("In following file:")
print(Model+"_RMMsv3.txt")
print("Process Completed")
print("==================================================")
