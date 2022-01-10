# A Dependent Multi-model Approach to Climate Prediction with Gaussian Processes

Authors: Marten Thompson, Dr. Amy Braverman, Dr.Snigdhansu Chatterjee

Manuscript, data, and code associated with submission to Climate Informatics 2022. 

## Code
`cmip6_netcdf_simplified.R` creates the de-seasonalized time series. `analysis.py` estimates and examines our model to the observed data and CMIP6 simulations. `analysis_loo.py` performs the Leave-One-Out analysis described in the manuscript. These two scripts call methods defined in `cmip6_eb_funcs.py`. `gpre_vis.R` creates the figures.


## Data
This work would not be possible without free and ready access granted by many authors to their data. Please see the [University of East Anglia](https://crudata.uea.ac.uk/cru/data/temperature/#sciref) page for license and references pertaining to HadCRUT5 observed data and seasonal corrects, in particular the citations below:

* Jones, Phil D. et al.Surface air temperature and its changes over the past 150 years. https://crudata.uea.ac.uk/cru/data/temperature/abs_glnhsh.txt. Accessed 2021/11/29. 1999.

* Morice, Colin P. et al. “An Updated Assessment of Near-Surface Temperature Change From 1850: The HadCRUT5 DataSet”. Journal of Geophysical Research: Atmospheres 126.3 (2021), e2019JD032361.

We are similarly thankful for access to several CMIP6 simulations. Please see the manuscript for a full set of citations as well as

* Li, L. (2019).  Cas fgoals-g3 model output prepared for cmip6 scenariomip ssp585.
* Rong,  X.  (2019).   Cams  cams-csm1.0  model  output  prepared  for  cmip6  scenariomip470ssp585.
* Semmler, T., Danilov, S., Rackow, T., Sidorenko, D., Barbi, D., Hegewald, J., Pradhan,472H.  K.,  Sein,  D.,  Wang,  Q.,  and  Jung,  T.  (2019).   Awi  awi-cm1.1mr  model  outputprepared for cmip6 scenariomip ssp585.


