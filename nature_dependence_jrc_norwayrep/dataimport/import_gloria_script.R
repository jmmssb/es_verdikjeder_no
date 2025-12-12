################################################################################
### Script to get GLORIA IO data
################################################################################

### 1 Directories & packages ###################################################
library(hdf5r)
library(hdf5r.Extra)
library(dplyr)
library(data.table)
library(Rfast)
library(readxl)


### specify here the filepath where the data is located
fpe<-file.path(gfp,timestep)

fpemat<-file.path(gfp,"Gloria_satellites_20240725")

## notes:
# economic values in k USD basic price
# we have 120 sectors. Industries, products, industries, products. 
# We delete the product part, as we do not need it.
# we have 164 countries, and 120 sectors
# we only need markup001 (basic prices)


### 1 Import Satellite data ####################################################

# this just to create a vector of index numbers for those rows and columns we want to keep
# we want to keep only MRIO data, so remove the supply matrices within
# fread is a faster reader than read.csv

39360/(2*120)
sec<-seq(1:39360) # sequence of numbers (for columns)
ser<-seq(1:39360) # sequence of numbers (for rows)
ones<-rep(1,times=120)
zeros<-rep(0,times=120)
oz<-c(ones,zeros)
full<-rep(oz,times=164)
sec[full==0]<-NA
ser[full==1]<-NA
cselector<-sec[!is.na(sec)] # removes NA values (from columns sequence)
rselector<-ser[!is.na(ser)] # removes NA values (from rows sequence)
rm(full,ones,oz,ser,sec,zeros) # remove objects no longer needed
 

# satellites for transaction data (reads the satellite data from .mat files)

tqm<-h5Read(file.path(fpemat,paste0("QT_",timestep,".mat")),name="QT")
yqm<-h5Read(file.path(fpemat,paste0("QY_",timestep,".mat")),name="QY")


### 2 Import Intermediate transactions #########################################

t<-data.table::fread(file = file.path(fpe,
                                      paste0(mcode,
                                             "_120secMother_AllCountries_002_T-Results_",
                                             timestep,
                                             "_059_Markup001(full).csv")
                                      ),
                      header = F,
                      select = cselector,
                     data.table=F)


t<-t[rselector,]

tm<-as.matrix(t)
dim(tm)


### 3 Import demand

y<-data.table::fread(file = file.path(fpe,
                                      paste0(mcode,
                                             "_120secMother_AllCountries_002_Y-Results_",
                                             timestep,
                                             "_059_Markup001(full).csv")),
                     header = F,
                     data.table=F) # check if we can't just leave it as a data.table

y<-y[rselector,]
ym<-as.matrix(y)
dim(ym)

### 4 Import value added
v<-data.table::fread(file = file.path(fpe,
                                      paste0(mcode,
                                             "_120secMother_AllCountries_002_V-Results_",
                                             timestep,
                                             "_059_Markup001(full).csv")),
                     header = F,
                     data.table=F) # check if we can't just leave it as a data.table

v<-v[,cselector]
vm<-as.matrix(v)
dim(vm)

rm(cselector,rselector,t,v,y)


### 5 Cut down to the dimensions of interest ###################################
## Using Region and Sector aggregation created by MM

########## first require the index data
sector_ind <- read_excel(file.path(wfp,"dataimport","GLORIA_ReadMe_059_adj.xlsx"), 
                         sheet = "Sectors")
region_ind <- read_excel(file.path(wfp,"dataimport","GLORIA_ReadMe_059_adj.xlsx"),  
                         sheet = "Regions")

demand_ind <- read_excel(file.path(wfp,"dataimport","GLORIA_ReadMe_059_adj.xlsx"),  
                         sheet = "Value added and final demand")

satellites_ind <- read_excel(file.path(wfp,"dataimport","GLORIA_ReadMe_059_adj.xlsx"),
                         sheet = "Satellites")

newcat<-sector_ind$MM_sector_name
newcat<-newcat[!is.na(newcat)]

newreg<-region_ind$MM_region_name
newreg<-newreg[!is.na(newreg)]

nco<-length(unique(region_ind$Region_names))
ncn<-length(newreg)
nso<-length(unique(sector_ind$Sector_names))
nsn<-length(newcat)

nd<-length(demand_ind$Final_demand_names)



########## make aggregation function

aggregator<-function(op = "aggrows", # choose dimension to aggregate
                            om,             # matrix to aggregate
                            al){            # aggregation key list
  if(op == "aggrows"){
    nm<-matrix(data=NA,nrow=length(al),ncol=ncol(om))
    for(i in 1:nrow(nm)){
      nm[i,]<-colSums(om[al[[i]],,drop=F])
    }
  } else if (op == "aggcols"){
    nm<-matrix(data=NA,nrow=nrow(om),ncol=length(al))
    for(i in 1:ncol(nm)){
      nm[,i]<-rowSums(om[,al[[i]],drop=F])
    }
  }
  return(nm)
}

########## Aggregation of Sectors
# NB: sectors need to be aggregated across all matrices (demand, transaction, satellites)

# Create aggregation list

# step 1
agglist1<-list()
for(i in 1:length(newcat)){
  agglist1[[i]]<-as.vector(sector_ind$Lfd_Nr[sector_ind$MM_sector_match==sector_ind$MM_sector[i]])
}
names(agglist1)<-newcat
agglist<-rep(agglist1,times=nco)
rm(agglist1)

# step 2
addlist<-list()
for(i in 1:nco){
  addlist[[i]]<-rep(nso*(i-1),times=nsn)
}
addvect<-unlist(addlist)
addvect
rm(addlist)

# step 3
agglistf<-agglist
for(i in 1:length(agglist)){
  for(j in 1:length(agglist[[i]])){
    agglistf[[i]][j]<-agglist[[i]][j]+addvect[[i]]
  }
}
rm(agglist,addvect)


# now run aggregation process

# First demand and value added.
dim(ym)
yms<-aggregator(op="aggrows",ym,agglistf)
dim(yms)
dim(vm)
vms<-aggregator(op="aggcols",vm,agglistf)
dim(vms)
rm(ym,vm)


# Then transaction matrix
tmsr<-aggregator(op="aggrows",tm,agglistf)
tms<-aggregator(op="aggcols",tmsr,agglistf)
rm(tm,tmsr)

# Then satellites
dim(tqm)
tqms<-aggregator(op="aggcols",tqm,agglistf)
rm(tqm)


# clean up
rm(sector_ind,agglistf)


########## Aggregation of Countries

# create aggregation list
# step 1
agglist1<-list()
for(i in 1:length(newreg)){
  agglist1[[i]]<-as.vector(region_ind$Lfd_Nr[region_ind$MM_region_match==region_ind$MM_region[i]])
}
names(agglist1)<-newreg

# step 2
agglist<-list()
for(i in 1:length(newreg)){
  agglist[[i]]<-list()
  for(j in 1:length(newcat)){
    agglist[[i]][[j]]<-agglist1[[i]]*nsn-nsn+j
  }
  names(agglist[[i]])<-newcat
}
names(agglist)<-newreg

# Step 3
agglistf<-unlist(agglist, recursive=FALSE)
rm(agglist)

## Need a seperate agglist for demandcols
agglist<-list()
for(i in 1:length(newreg)){
  agglist[[i]]<-list()
  for(j in 1:6){
    agglist[[i]][[j]]<-agglist1[[i]]*nd-nd+j
  }
  names(agglist[[i]])<-demand_ind$Final_demand_names
}
names(agglist)<-newreg

agglistfd<-unlist(agglist, recursive=FALSE)
rm(agglist)

## Need a seperate agglist for v.a.rows
agglist<-list()
for(i in 1:length(newreg)){
  agglist[[i]]<-list()
  for(j in 1:6){
    agglist[[i]][[j]]<-agglist1[[i]]*nd-nd+j
  }
  names(agglist[[i]])<-demand_ind$Value_added_names
}
names(agglist)<-newreg

agglistfv<-unlist(agglist, recursive=FALSE)
rm(agglist)

# now run aggregation process

# First demand.
dim(yms)
ymscr<-aggregator(op="aggrows",yms,agglistf)
dim(ymscr)
ymsc<-aggregator(op="aggcols",ymscr,agglistfd)
dim(ymsc)
rm(yms,ymscr)

# then value added
dim(vms)
vmscr<-aggregator(op="aggcols",vms,agglistf)
dim(vmscr)
vmsc<-aggregator(op="aggrows",vmscr,agglistfv)
dim(vmsc)
rm(vms,vmscr)

# Then transaction matrix
dim(tms)
tmscr<-aggregator(op="aggrows",tms,agglistf)
tmsc<-aggregator(op="aggcols",tmscr,agglistf)
dim(tmsc)
rm(tms,tmscr)

# Then satellites
dim(tqms)
tqmsc<-aggregator(op="aggcols",tqms,agglistf)
dim(tqmsc)
rm(tqms)

# for demand satellites we need the short agglist
dim(yqm)
yqmc<-aggregator(op="aggcols",yqm,agglistfd)
dim(yqmc)
rm(yqm)

# clean up
rm(agglistfd,agglistfv,agglist1,region_ind,aggregator,nco,nso)


########## Rename and add index names

z<-tmsc # transaction matrix
dim(z)
colnames(z)<-names(agglistf)
rownames(z)<-names(agglistf)

y<-ymsc # demand matrix
dim(y)
di2<-rep(demand_ind$Final_demand_names,times=ncn)
sellist<-list()
for(i in 1:ncn){
  sellist[[i]]<-rep(i,times=nd)
}
selvect<-unlist(sellist)
di<-vector()
for(i in 1:length(di2)){
  di[i]<-paste(newreg[selvect[i]],di2[i],sep="_")
}
di
colnames(y)<-di
rownames(y)<-names(agglistf)

# add for value added
v<-vmsc # value added matrix
dim(v)
vi2<-rep(demand_ind$Value_added_names,times=ncn)
sellist<-list()
for(i in 1:ncn){
  sellist[[i]]<-rep(i,times=nd)
}
selvect<-unlist(sellist)
vi<-vector()
for(i in 1:length(vi2)){
  vi[i]<-paste(newreg[selvect[i]],vi2[i],sep="_")
}
vi
colnames(v)<-names(agglistf)
rownames(v)<-vi

# clean up
rm(demand_ind,sellist,di2,vi2,selvect,nd)

zq <-tqmsc # satellites of economy
dim(zq)
colnames(zq)<-names(agglistf)
zq<-zq[1:nrow(satellites_ind),]# this step is tmp needed because updated satellites are 6130 rows, while old labels only go until 5982
zq<-cbind(satellites_ind,zq)

yq <- yqmc # satellites of final demand
dim(yq)
colnames(yq)<-di
yq<-yq[1:nrow(satellites_ind),]# this step is tmp needed because updated satellites are 6130 rows, while old labels only go until 5982
yq<-cbind(satellites_ind,yq)
rm(satellites_ind)

# keep only satellites of interest
zq<-zq %>% 
  filter(Sat_head_indicator %in% c("Material","Land use","Energy",
                                   "Biodiversity loss","Water stress",
                                   "Blue_water_consumption")) %>%
  select(-Lfd_Nr)

yq<-yq %>% 
  filter(Sat_head_indicator %in% c("Material","Land use","Energy",
                                   "Biodiversity loss","Water stress",
                                   "Blue_water_consumption")) %>%
  select(-Lfd_Nr)


rm(tmsc,tqmsc,ymsc,vmsc,yqmc,agglistf)



### 7 Index Identifiers ########################################################

fsq<-rep(newcat,length(newreg))
fsq

fcq<-list()
for(i in 1:length(newreg)){
  fcq[[i]]<-rep(newreg[i],length(newcat))
}
fcq<-do.call("c",fcq)
fcq


### 8 Save result ##############################################################

save(v,y,z,zq,yq,fsq,fcq,
     file = file.path(wfp,"dataimport","tmpdir",paste0("gloria_",timestep,".RData")))

rm(list=ls()[! ls() %in% c("wfp","gfp","nfp")])
gc()

#############################################################################


