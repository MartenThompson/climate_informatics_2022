# Visuals and summary statistics for manuscript


empty_plot <- function(xlim, ylim, xlab, ylab, title=NA) {
  par(mgp=c(1.8,0.5,0),     # axis titles, tic labels, tic values
      mar = c(3, 3, 1, 1),  # b l t r
      las = 1,              # all horz axis tics
      xaxs = "i", yaxs = "i")
  plot(xlim, ylim, type='n', 
       xlab=xlab, ylab=ylab, main=title,
       bty = "l",          # remove bounding box
       tcl  = -0.2)
}

setwd('~/Git/climate_informatics_2022/code/')


##################
#### Observed ####
##################
observed <- read.csv('output/gpre_vis/preds_obs.csv')
x <- observed$x*(250) + 1850

png('images/observed_pred_nomuhat.png', width=540, height=360)
empty_plot(c(1850,2100), c(-1,6.5), '', paste('deviation ', expression('\u00B0C')))
lines(x, observed$observed, col=rgb(0,0,0,1))

stdev <- rep(sqrt(observed$obs_var_scalar[1]), length(x))
polygon(c(x, rev(x)), c(observed$obs_pred-2*stdev, rev(observed$obs_pred+2*stdev)),
        col = rgb(0,1,0,0.5), border = NA)
lines(x, observed$obs_pred, col=rgb(0,1,0,1), lwd=1)

legend(1856, 6.5, c('observed', 'predicted'), fill=c('black', 'green'), ncol=2,border=NA)
dev.off()


#############
#### LOO ####
#############
plot_full <- function(filepath, fn, a=1, b=length(x)) {
  dat <- read.csv(paste0(filepath,fn))
  x <- (dat$x-0.0003320053)*(250) + 1850
  x <- x[a:b]
  
  empty_plot(c(1850,2100), c(-2,8), '', paste('deviation ', expression('\u00B0C')))
  lines(x, dat$observed[a:b], col=rgb(0,0,0,1), lwd=1)
  
  stdev <- rep(sqrt(dat$obs_var_scalar[1]), length(x))
  
  lines(x, dat$obs_pred[a:b], col=rgb(0,1,0,1), lwd=1)
  
  lines(x, dat$mu_hat[a:b], col=rgb(1,0,0,1), lwd=2)
  
  legend(1856, 8, c(stringr::str_split(fn, "\\.")[[1]][1], expression(hat(mu)(t)), 'predicted'), fill=c('black', 'red', 'green'), ncol=2,border=NA)
  return(dat)
}

plot_zoom <- function(filepath, fn, a=1, b=length(x)) {
  dat <- read.csv(paste0(filepath,fn))
  x <- (dat$x-0.0003320053)*(250) + 1850
  x <- x[a:b]
  
  empty_plot(c(2021,2026), c(-1,4), '', paste('deviation ', expression('\u00B0C')))
  lines(x, dat$observed[a:b], col=rgb(0,0,0,1), lwd=2)
  
  stdev <- rep(sqrt(dat$obs_var_scalar[1]), length(x))
  
  polygon(c(x, rev(x)), c(dat$obs_pred[a:b]-2*stdev, rev(dat$obs_pred[a:b]+2*stdev)),
          col = rgb(0,1,0,0.25), border = NA)
  polygon(c(x, rev(x)), c(dat$obs_pred[a:b]-1*stdev, rev(dat$obs_pred[a:b]+1*stdev)),
          col = rgb(0,1,0,0.25), border = NA)
  lines(x, dat$obs_pred[a:b], col=rgb(0,1,0,1), lwd=2)
  
  legend(2021.1, 4, c(stringr::str_split(fn, "\\.")[[1]][1], 'predicted'), fill=c('black', 'green'), ncol=2, border=NA)
  
  return(dat)
}


dir <- 'output/gpre_vis/loo_mse/'
files <- list.files(dir)
for (f in files) {
  png(paste0('images/loo_', stringr::str_split(f, "\\.")[[1]][1], '.png' ), height=360, width=360)
  dat = plot_full(dir, f)#, 2062,2120)  
  
  dev.off()
  #break
  mse_model <- mean((dat$observed[2062:3012]-dat$obs_pred[2062:3012])^2)
  mse_muhat <- mean((dat$observed[2062:3012] - dat$mu_hat[2062:3012])^2)
  mse_ymean <- mean((dat$observed[2062:3012] - dat$y_mean[2062:3012])^2)
  cat(f, 'MSE model:', round(mse_model,2), ', muhat:', round(mse_muhat,2), ', ymean:', round(mse_ymean,2), '\n')
}


for (f in files) {
  png(paste0('images/zoomloo_', stringr::str_split(f, "\\.")[[1]][1], '.png' ), height=360, width=360)
  dat = plot_zoom(dir, f, 2062,2120)  
  
  dev.off()
  #break
}


############################
#### Conditional Effect ####
############################
dat <- read.csv(paste0(dir,'access-cm2.csv'))
cond.effect <-dat$obs_pred - dat$mu_hat
keep <- !is.na(cond.effect)
cond.effect <- cond.effect[keep]
x <- (dat$x-0.0003320053)*(250) + 1850
x <- x[keep]

png(paste0('images/condeff_', 'access-cm2', '.png' ), height=360, width=360)
empty_plot(c(2020,2100), c(-1.5,1.5), '', paste('deviation ', expression('\u00B0C')))
lines(x, cond.effect, col=rgb(0,1,0,1))
lines(c(2020,2100), c(0,0), col=rgb(1,0,0,1), lwd=2)
legend(2022, 1.5, expression('predicted'-hat(mu)(t)), fill=c('green'), border=NA)
dev.off()

