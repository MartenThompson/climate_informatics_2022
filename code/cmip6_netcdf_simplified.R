library(dplyr)
library(forecast)
library(ncdf4)
'%notin%' <- Negate('%in%')
#### Comments ####
# All data starts in 1850 Jan, ends in 2100 Dec, never skips a month

# near surface air temperature (TAS)

# inconsistent files
  # awi-cm-1-1-mr starts in 1851
  # cams-csm1-0 stops on 2099-10
  # fgoals-g3 jumps from 2016-11 back to 2014-12 and repeats all of 2015 and 2016
inconsistent_list <- c('awi-cm-1-1-mr', 'cams-csm1-0', 'fgoals-g3')
max_time <- 3012

#### CSV NetCDF Files ####
setwd('~/Git/climate_informatics_2022/data/')

#my_start_dt <- as.POSIXct('1850-01-01 00:00:00')
chop_off <- "_ssp585.nc"
nc_filenames = list.files('./CMIP6-tas/', "*.nc")

absolute_1961_90 <- read.csv('abs_glnhsh_reshape.txt')
kelvin_to_c <- -273.15

all_data <- data.frame()
for (f in nc_filenames) {
  name <- tolower(substr(f, start=1, stop=nchar(f)-nchar(chop_off)))
  
  if (name %notin% inconsistent_list) {
    data <- nc_open(paste0('CMIP6-tas/',f))
    nsa_temp <- ncvar_get(data, 'tas') + kelvin_to_c
    n <- length(nsa_temp)
    
    # Remove seasonality by subtracting 1961-90 monthly averages
    nsa_temp_ds <- nsa_temp # de-seasoned...
    for (i in 1:n) {
      m <- ifelse(i%%12 != 0, i%%12, 12)
      nsa_temp_ds[i] <- nsa_temp[i] - absolute_1961_90[absolute_1961_90$Month == m, 'Anom']
    }
    
    df <- data.frame(name = rep(name, n),
                     nsa_temp_raw = nsa_temp,
                     nsa_ds = nsa_temp_ds,
                     x = (1:n)/max_time
    )
    
    all_data <- rbind(all_data, df)
  } 
}

# confirm raw_has seasonality
dat.plot <- all_data[all_data$name == 'access-cm2',]
end <- 50
plot(dat.plot$x[1:end], dat.plot$nsa_temp_raw[1:end])
# and adj does not
plot(dat.plot$x[1:end], dat.plot$nsa_ds[1:end])


#### CSV Observed #####
# Note: HadCRUT5 is already adjusted for seasonality.
# Hence, we will make raw and adj temps equal for this.
observed <- read.csv('HadCRUT5.0Analysis_gl_reshape.txt')
observed_temps <- observed[seq(1,nrow(observed), by=2),] # every-other row
observed_temps <- observed_temps[,colnames(observed_temps) %notin% c('Annual')]
observed_temps_long <- reshape(observed_temps, idvar='Year',
                               varying=colnames(observed_temps)[colnames(observed_temps) %notin% c('Year')],
                               v.name=c('nsa_temp_raw'),
                               times=1:12,
                               direction = 'long')
observed_temps_long <- observed_temps_long[observed_temps_long$nsa_temp_raw != -9.999,] # remove 4 missing values

observed_df <- data.frame(name = rep('observed', nrow(observed_temps_long)),
                          nsa_temp_raw = observed_temps_long$nsa_temp_raw,
                          nsa_ds = observed_temps_long$nsa_temp_raw,
                          time_raw = paste(observed_temps_long$Year, observed_temps_long$time, '15 00:00:00', sep = '-'))
observed_df$time_posix <- as.POSIXct(observed_df$time_raw)
my_start_dt <- as.POSIXct('1850-01-01 00:00:00')
observed_df$time_hrs_since <- as.numeric(difftime(observed_df$time_posix, my_start_dt, units='hours'))
observed_df <- observed_df[order(observed_df$time_hrs_since),]

observed_keep <- observed_df[,colnames(observed_df) %in% c('name', 'nsa_temp_raw', 'nsa_ds')]
observed_keep$x <- (1:nrow(observed_keep))/max_time

# confirm order, no weird lines criss crossing
plot(observed_keep$x, observed_keep$nsa_temp_raw, type='n')
lines(observed_keep$x, observed_keep$nsa_temp_raw)
# confirm no seasonality
plot(observed_keep$x[1:24], observed_keep$nsa_ds[1:24], type='n')
lines(observed_keep$x[1:24], observed_keep$nsa_ds[1:24])



#### Saving ####
all_data <- rbind(observed_keep, all_data)

write.csv(all_data, "all_cmip6_simplified.csv", row.names = FALSE)

cmip6 <- read.csv('all_cmip6_simplified.csv')
table(cmip6$name)
