


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

setwd('~/Git/gp/grouped/climate/')

##################
#### Observed ####
##################


observed <- read.csv('output/gpre_vis/preds_obs.csv')
x <- observed$x*(250) + 1850

#png('images/gpre/observed_pred_nomuhat.png', width=540, height=360)
empty_plot(c(1850,2100), c(-1,6.5), '', paste('deviation ', expression('\u00B0C')))
lines(x, observed$observed, col=rgb(0,0,0,1))



stdev <- rep(sqrt(observed$obs_var_scalar[1]), length(x))
polygon(c(x, rev(x)), c(observed$obs_pred-2*stdev, rev(observed$obs_pred+2*stdev)),
        col = rgb(0,1,0,0.5), border = NA)
#polygon(c(x, rev(x)), c(observed$obs_pred-1*stdev, rev(observed$obs_pred+1*stdev)),
#        col = rgb(0,1,0,0.25), border = NA)
lines(x, observed$obs_pred, col=rgb(0,1,0,1), lwd=1)
#lines(x, observed$mu_hat, col=rgb(1,0,0), lwd=1.5)

#lines(c(0.68*250+1850, 0.68*250+1850), c(-100,100), lty=2)
#legend(1856, 6.5, c('observed', 'predicted', expression(hat(mu)(t))), fill=c('black', 'green', 'red'), ncol=3,border=NA)
legend(1856, 6.5, c('observed', 'predicted'), fill=c('black', 'green'), ncol=2,border=NA)
#dev.off()

#############
#### LOO ####
#############

my_plot <- function(filepath, fn, a=1, b=length(x)) {
  dat <- read.csv(paste0(filepath,fn))
  x <- (dat$x-0.0003320053)*(250) + 1850
  x <- x[a:b]
  
  empty_plot(c(1850,2100), c(-2,8), '', paste('deviation ', expression('\u00B0C')))
  lines(x, dat$observed[a:b], col=rgb(0,0,0,1), lwd=1)
  

  
  #stdev <- sqrt(dat$mu_hat_var + dat$obs_var_scalar[1])
  stdev <- rep(sqrt(dat$obs_var_scalar[1]), length(x))
  
  #polygon(c(x, rev(x)), c(dat$obs_pred[a:b]-2*stdev, rev(dat$obs_pred[a:b]+2*stdev)),
  #        col = rgb(0,1,0,0.25), border = NA)
  #polygon(c(x, rev(x)), c(dat$obs_pred[a:b]-1*stdev, rev(dat$obs_pred[a:b]+1*stdev)),
  #        col = rgb(0,1,0,0.25), border = NA)
  lines(x, dat$obs_pred[a:b], col=rgb(0,1,0,1), lwd=1)
  
  
  lines(x, dat$mu_hat[a:b], col=rgb(1,0,0,1), lwd=2)
  
  legend(1856, 8, c(stringr::str_split(fn, "\\.")[[1]][1], expression(hat(mu)(t)), 'predicted'), fill=c('black', 'red', 'green'), ncol=2,border=NA)
  #legend(2021.1, 4, c(stringr::str_split(fn, "\\.")[[1]][1], 'predicted'), fill=c('black', 'green'), ncol=2, border=NA)
  
  return(dat)
}


dir <- 'output/gpre_vis/loo_mse/'
files <- list.files(dir)
for (f in files) {
  # png(paste0('images/gpre/360loo_', stringr::str_split(f, "\\.")[[1]][1], '.png' ), height=360, width=360)
  dat = my_plot(dir, f)#, 2062,2120)  
  
  # dev.off()
  break
  # mse_model <- mean((dat$observed[2062:3012]-dat$obs_pred[2062:3012])^2)
  # mse_sgp <- mean((dat$observed[2062:3012] - dat$single_gp_theta[2062:3012])^2)
  # mse_muhat <- mean((dat$observed[2062:3012] - dat$mu_hat[2062:3012])^2)
  # mse_ymean <- mean((dat$observed[2062:3012] - dat$y_mean[2062:3012])^2)
  # cat(f, 'MSE model:', round(mse_model,2), ', single gp:', round(mse_sgp,2), ', muhat:', round(mse_muhat,2), ', ymean:', round(mse_ymean,2), '\n')
}
# note mu hat does not see all of grey data, and we want to see the green pulled from mu hat to grey.

#### cond eff ####
cond.effect <-dat$obs_pred - dat$mu_hat
keep <- !is.na(cond.effect)
cond.effect <- cond.effect[keep]
x <- (dat$x-0.0003320053)*(250) + 1850
x <- x[keep]

png(paste0('images/gpre/360condeff_', 'access-cm2', '.png' ), height=360, width=360)
empty_plot(c(2020,2100), c(-1.5,1.5), '', paste('deviation ', expression('\u00B0C')))
lines(x, cond.effect, col=rgb(0,1,0,1))
lines(c(2020,2100), c(0,0), col=rgb(1,0,0,1), lwd=2)
legend(2022, 1.5, expression('predicted'-hat(mu)(t)), fill=c('green'), border=NA)
dev.off()

#################
summary(dat$mu_hat_var)

dat <- read.csv(paste0('output/gpre_vis/loo2/', 'mpi-esm1-2-hr.csv'))
dat2 <- read.csv(paste0('output/gpre_vis/loo/', 'bcc-csm2-mr.csv'))
empty_plot(c(0.6,1), c(-1,8), '','')
lines(dat$x[2061:3012], dat$observed[2061:3012])
lines(dat$x[2061:3012], dat$obs_pred[2061:3012], col='red')
lines(dat$x[2061:3012], dat$single_gp_theta[2061:3012], col='blue')

my_plot('output/gpre_vis/loo/', 'bcc-csm2-mr.csv')

mean((dat$observed[2062:3012]-dat$mu_hat[2062:3012])^2)
mean((dat$observed[2062:3012]-dat$mu_hat[2062:3012])^2)
mean((dat$observed[2062:3012]-dat$mu_hat[2062:3012])^2)

#### Summary Stats ####

ss <- read.csv('output/gpre_vis/loo_output2.csv')
ss <- ss[order(ss$dataset_test),]
ss$mse_model <- round(ss$mse_model, 2)
ss$mse_single_gp <- round(ss$mse_single_gp, 2)
ss$mse_mu_hat <- rep(NA, nrow(ss))

files <- list.files('output/gpre_vis/loo')
itr <- 1
for (f in files) {
  dat <- read.csv(paste0('output/gpre_vis/loo/', f))
  ss$mse_mu_hat[itr] <- round(mean((dat$observed[2062:3012] - dat$mu_hat[2062:3012])^2), 2)
  itr <- itr+1
}


ss <- ss[,c(1,2,4,3)]
ss





# good 
ss[ss$mse_model <= ss$mse_single_gp,]

# bad 
# recall mse_single_gp knows all the data, which is not true in practice
# being beat by it isn't that bad.
ss[ss$mse_model > ss$mse_single_gp,]


# let's get something to punch up: a gp fit to 1850-2100 data (single)

files <- list.files('output/fits_preds_simplified/')



##################
#### Mat Norm ####
##################

setwd('~/Git/gp/grouped/climate/')
ks.means <- read.csv('output/matnorm_KS2061_mean.csv', header=FALSE)
ks.cov <- as.matrix(read.csv('output/matnorm_KS2061_cov.csv', header=FALSE))

plot(ts(ks.means))
summary(ks.means)
image(ks.cov)
ks.cov[1:4,1:4]

#K <- read.csv('output/K')

