################################################################################
### Script to get EXIOBASE IO data
################################################################################

### 1 Directories & packages ###################################################
library(data.table)

### specify here the filepath where the data is located
fpe <- file.path(efp,paste0("IOT_",timestep,"_ixi"))

fpesat<-file.path(efp,"satellite")

## notes:
# economic values in million EUR
# we have 163 sectors. All industries
# we have 49 regions


### 1 Import Satellite data ####################################################

# no need to use satellite data here
 

### 2 Import Intermediate transactions #########################################

t<-data.table::fread(file = file.path(fpe,"Z.txt"
                                      ),
                      header = F,
                     data.table=F)

# save labels
fcq<-t[-c(1,2,3),1]
fsq<-t[-c(1,2,3),2]

# then remove labels (first three rows and first two columns)
t <- t[-c(1,2,3),-c(1,2)]
# save as matrix
dim(t)
tm<-as.matrix(t)
storage.mode(tm)<-"numeric"
dim(tm)
is.numeric(tm)
rm(t)
gc()

### 3 Import demand

y<-data.table::fread(file = file.path(fpe,"Y.txt"),header = F,data.table=F)

# save labels
fdq<-as.vector(as.matrix(y[2,-c(1,2)]))
dcq<-as.vector(as.matrix(y[1,-c(1,2)]))

# remove labels
y <- y[-c(1,2,3),-c(1,2)]

# save as matrix
ym<-as.matrix(y)
storage.mode(ym)<-"numeric"
dim(ym)
mode(ym)
rm(y)

### 4 Import value added

# not available and also not really needed, we can create it ourselves

### 5 Cut down to the dimensions of interest ###################################
## Using Region and Sector aggregation created by MM

########## first require the index data
sector_ind <- read_excel(file.path(wfp,"dataimport","EXIOBASE_Aggregation.xlsx"), 
                         sheet = "Sectors")
region_ind <- read_excel(file.path(wfp,"dataimport","EXIOBASE_Aggregation.xlsx"),  
                         sheet = "Regions")

newcat<-sector_ind$MM_sector_name
newcat<-newcat[!is.na(newcat)]

newreg<-region_ind$MM_region_name
newreg<-newreg[!is.na(newreg)]

nco<-length(unique(region_ind$Region_names))
ncn<-length(newreg)
nso<-length(unique(sector_ind$Sector_names))
nsn<-length(newcat)

nd<-length(unique(fdq))



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
rm(ym)

# Then transaction matrix
tmsr<-aggregator(op="aggrows",tm,agglistf)
tms<-aggregator(op="aggcols",tmsr,agglistf)
rm(tm,tmsr)

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
  for(j in 1:7){
    agglist[[i]][[j]]<-agglist1[[i]]*nd-nd+j
  }
  names(agglist[[i]])<-unique(fdq)
}
names(agglist)<-newreg

agglistfd<-unlist(agglist, recursive=FALSE)
rm(agglist)

# now run aggregation process

# First demand.
dim(yms)
ymscr<-aggregator(op="aggrows",yms,agglistf)
dim(ymscr)
ymsc<-aggregator(op="aggcols",ymscr,agglistfd)
dim(ymsc)
rm(yms,ymscr)

# Then transaction matrix
dim(tms)
tmscr<-aggregator(op="aggrows",tms,agglistf)
tmsc<-aggregator(op="aggcols",tmscr,agglistf)
dim(tmsc)
rm(tms,tmscr)


# clean up
rm(agglistfd,agglistfv,agglist1,region_ind,aggregator,nco,nso)


########## Rename and add index names

z<-tmsc # transaction matrix
dim(z)
colnames(z)<-names(agglistf)
rownames(z)<-names(agglistf)

y<-ymsc # demand matrix
dim(y)
di2<-rep(unique(fdq),times=ncn)
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

# clean up
rm(demand_ind,sellist,di2,selvect,nd)
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

save(y,z,fsq,fcq,di,
     file = file.path(wfp,"dataimport","tmpdir",paste0("exiobase_",timestep,".RData")))

rm(list=ls()[! ls() %in% c("wfp","efp","nfp")])
gc()

#############################################################################


