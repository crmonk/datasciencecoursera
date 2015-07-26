---
title: "Exploratory Data Analysis crib"
output: html_document
---

#Exploratory

##1D

#breaks specifies number bars # pm25 is level of small particle pollution
hist(pollution$pm25, col = "green", breaks = 100) 
#rug plots all points in row at bottom of histogram 
rug(pollution$pm25) 
#abline plots line vertical or horizontal, lwd = line width
abline(v=12, lwd =2)
abline(v=median(pollution$pm25), col="magenta", lwd =2)

boxplot(pollution$pm25,col = "blue")
abline(h=12)

barplot(table(pollution$region), col ="wheat", main = "Number of Counties in Each Region")


##2D
#Multiple boxplots
boxplot(pm25 ~ region, data = pollution, col="red")
#multi histogram
par(mfrow = c (2,1), mar = c(4,4,2,1))
hist(subset(pollution, region == "east")$pm25, col="green")
hist(subset(pollution, region == "west")$pm25, col="green")

#scatterplot
with(pollution, plot(latitude, pm25))
fit <- lm(pm25 ~ latitude)
abline(fit, lwd = 3, col="blue")

#lty line type, see par for details
abline(h=12, lwd =2, lty = 2)
# add colour per region
with(pollution, plot(latitude, pm25, col=region))

#Multi scatterplot
par(mfrow = c (1,2), mar = c(5,4,2,1))
with(subset(pollution, region == "west"), plot(latitude, pm25, Main = "West"))
with(subset(pollution, region == "east"), plot(latitude, pm25, Main = "East"))


#ONLINE RESOURCES - inspiration
R Graph Gallery
R Bloggers

#Base plot - build up layer upon layer with lines of R code. Cannot save 'new plot type' or remove part of the plot.

#phase1 inititalise
library(datasets)
data(cars)
with(cars, plot(speed,dist))

#pch = plotting symbol (default open circle)
example(points)
#lty = tine type (default solid, options dotted, dashed, ...)
#lwd = line width (integer multiple) 
#col = colour/ colors() = same but vector with entries specified by name
#xlab = x axis label
#ylab = y axis label
# par() GLOBAL parameters (all plots)
#las orientation of axis on labels
#bg background colour
# -- the following CANNOT be overriden in individual plots
#mar margin size (par("mar") = default = 5.1 4.1 4.1 2.1. 4 sides, starting at BOTTOM and going CLOCKWISE)
#oma outer margin size (default 0, to see, call par("oma"))
#mfrow multiple plots per row (rows, cols)
#mfcol multiple plots per column (rows, cols)


#phase2 annotate (sometimes option inbuilt in plot params, or added later in seperate line)
#lines - add lines
#points - add points
#test 
#title
#mtext text in the margins
#axis axis ticks/labels

#can use plot parameter to not plot points/ line so you can add later. Plus legend!
with (airquality, plot(Wind, Ozone, main = "Ozone and Wind", type="n"))
with (subset(airquality, Month == 5), points(Wind, Ozone, col= "blue"))
with (subset(airquality, Month != 5), points(Wind, Ozone, col= "red"))
legend("topright", pch = 1, col = c("blue", "red"), legend = c("May","Other Months"))



# leave 2 lines of space on top and add outer title
par(mfrow = c(1,3), mar = c(4,4,2,1), oma = c(0,0,2,0))
with (airquality, {
  plot(Wind, Ozone, main = "Ozone and Wind")
  plot(Solar.R, Ozone, main = "Ozone and solar radiation")
  plot(Temp, Ozone, main = Ozone and Temperature)
  mtext("Ozone and weather in New York city", outer = TRUE)
})

#Lattice plots - 1 function call - useful for coplots = conditioning plots - rel between x and y 
# over various values of z. margins, spacing automatically set. Cannot add to plot once created.
# CAVEAT - lattice generates an object of type trellis which needs to be PRINTED 
#(unlike plot which generates straight onto device). Thankfully R normally auto-prints to current device. 
#xyplot - scatterplots
#bwplot - "box and whiskers plot" = boxplot
#histogram
#stripplot - boxplot but with actual points
#dotplot - plot dots on violin strings
#splom - scatterplot matrix (like pairs in base plotting system)
#levelplot, contourplot: plotting image data

#xyplot( y ~ x | f * g, data)
state <- data.frame(state.x77, region = state.region)
xyplot(Life.Exp ~ Income | region, data = state, layout = c(4,1))

library(lattice)
library(datasets)
#basic xyplot
xyplot(Ozone ~ Wind, data = airquality)
##convert 'Month' to a factor variable
airquality <- transform(airquality, Month = factor(Month))
xyplot(Ozone ~ Wind | Month, data = airquality, layout = c(5,1))

# CAVEAT part 2. If assign lattice result to variable, then need to call print manually
p <- xyplot(Ozone ~ Wind, data = airquality)
print(p)

set.seed(10)
x <- rnorm(100)
f <- rep(0:1, each = 50)
y <- x + f - f* x = rnorm(100, sd = 0.5)
f <- factor(f, labels = c("Group 1", "Group 2"))
xyplot( y ~ x | f, layout = c(2, 1)) ##Plot with 2 panels
# Panel function - Can customise (differently to defaults)
xyplot( y ~ x | f, panel = function(x, y, ...) {
  panel.xyplot(x, y, ...) ## First call the default panel function for 'xyplot'
  panel.abline(h = median(y), lty = 2) ## Add a horizontal line at the median
})
# Custom Panel - example 2
xyplot( y ~ x | f, panel = function(x, y, ...) {
  panel.xyplot(x, y, ...) ## First call default panel function
  panel.lmline(x, y, col = 2) ## Overlay a simple linear regression line
})


#ggplot2 "grammar of graphics" - build plot up one by one. but labels, spacing automatically put in 
#correct place. A lot of defaults, but they are customisable.
# "Shorten the distance from the mind to the page"
# mapping from data to aesthetic attributes = aesthetics (colour, shape, size) 
# of geometric objects = geoms (points, lines, bars) 
# (+ statistical info) on a coordinate system
library(ggplot2)
data(mpg)
#"quick plot" - like plot in base plotting. Must use data.frame
# qplot( <x coord>, <y coord>, <data definition>)
qplot(displ, hwy, data = mpg)
# NOTE: Factors should be properly labelled (not just default index 1,2,3,...)
qplot(displ, hwy, data = mpg, color = drv) ## Colours defined automatically
qplot(displ, hwy, data = mpg, geom = c("point", "smooth")) ## Colours defined automatically
#HISTOGRAM (ONLY PROVIDE ONE VARIABLE)
qplot(hwy, data = mpg, fill = drv)
# Facets  = <rows>~<columns> 
# Facets equivalent to lattice multiple panels
#eg1
qplot(displ, hwy, data = mpg, facets = .~drv) 
#eg2
qplot(displ, hwy, data = mpg, facets = drv~., binwidth = 2) 

#1d data example (not got dataset => just examine code)
qplot(log(eno), data = maacs, fill = mopos)
qplot(log(eno), data = maacs, geom = "density") 
qplot(log(eno), data = maacs, geom = "density", color = mopos)
#scatterplots (relations)
qplot(log(pm25), log(eno), data = maacs)
qplot(log(pm25), log(eno), data = maacs, shape = mopos) ## difficult to see
qplot(log(pm25), log(eno), data = maacs, color = mopos) ## easier to see by color
# DIFF COLOURS - Want smoothed linear regression line per 'mopos group'= is allergic to mice
qplot(log(pm25), log(eno), data = maacs, color = mopos, geom = c("point", "smooth"), method = "lm")
# DIFF FACETS - Want smoothed linear regression line per 'mopos group'= is allergic to mice
qplot(log(pm25), log(eno), data = maacs, geom = c("point", "smooth"), method = "lm", facets= .~mopos)

# More flexible ggplot2 - PLOTS are built up in layers
# data.frame
# aesthetic mappings (how data mapped to color/ size )
# geoms
# facets - for conditional plots
# stats e.g. smoothing, binning, quantiles, smoothing
# scales (e.g. male = red, female = blue)
# coordinate system (numerical representations -> plot)

# basic version split by bmicat:
qplot(logpm25, NocturnalSympt, data = maacs, geom = c("point", "smooth"), method = "lm", facets= .~bmicat)

#detailed version
g <- ggplot(maacs, aes(logpm25, NocturnalSympt))
# Can always call summary on a ggplot object (shows data, mappping, facets)
summary(gg)
# Not ready to print yet (no layers!)
p <- g + geom_point()
print(p) ## OK now
g + geom_point() ## auto-print plot without saving 
g + geom_point() + geom_smooth() ## + smoothing lowse smoothing by default
g + geom_point() + geom_smooth(method = "lm") ## + define linear model smoothing 
g + geom_point() + facet_grid(. ~ bmicat) + geom_smooth(method = "lm") ## + facets per weight factor

# ANNOTATION
# LABELS: xlab(), ylab(), labs(), ggtitle()
# geom functions have parameters
# Global changes: theme(legend.position = "none") # also used for background colour
# standard themes: theme_gray(), theme_bw()
# modifying aesthetics:
g + geom_point(color = "steelblue", size = 4, alpha = 1/2) ## color constant changed
g + geom_point(aes(color = bmicat), size = 4, alpha = 1/2) ## color per factor
g + geom_point(aes(color = bmicat)) + labs(title = "MAACS Cohort") 
  + labs( x = expression("log "*PM[2.5]), y = "Nocturnal Symptoms"))  ##PM2.5 subscript
g + geom_point(aes(color = bmicat), size = 4, alpha = 1/2) 
  + geom_smooth(size = 4, linetype = 3, method = "lm", se = FALSE)## se controls confidence intervals
# Changing theme
g + geom_point(aes(color = bmicat)) + theme_bw(base_family = "Times") ##changed font and background


# Axis limits 
# in plot, can set with ylim
testdat <- data.frame(x = 1:100, y = rnorm(100))
testdat[50,2] <- 100 ## Outlier!
plot(testdat$x, testdat$y, type = "l", ylim = c(-3,3))

# without limits in ggplot2
g <- ggplot(testdat, aes(x = x, y = y))
g + geom_line()
#WRONG
g + geom_line() + ylim(-3,3) ## subsets out outlier points (MISSING), so lines wrong
#CORRECT
g + geom_line() + coord_cartesian(ylim(-3,3))

# "Conditional" plotting on continuous variable using cut()
# This cuts continuous variable into reasonable ranges = new categories
# Note this is not runnable as dataset missing...
cutpoints <- quantile(maacs$logno2_new, seq(0, 1, length = 4), na.rm = TRUE)
maacs$no2dec <- cut(maacs$logno2_new, cutpoints) ## cuts data at quantiles
levels(maacs$no2dec)

#FINAL example
# setup with data frame
g <- ggplot2(maacs, aes(logpm25, NocturnalSympt))

## Add layers
g + geom_point(alpha = 1/3)
  + facet_wrap(bmicat ~ no2dec, nrow = 2, ncol = 4)
  + geom_smooth(method="lm", se=FALSE, col="steelblue") # turn off standard error = confidence intervals
  + theme_bw(base_family = "Avenir", base_size = 10) ## default 12
  + labs(x = expression("log " * PM[2.5]))
  + labs(y = "Nocturnal Symptoms")
  + labs(title = "MAACS Cohort")


#Graphics Devices
#OSx = quartz
# Windows = windows()
# Unix/ Linux = x11()

##Vector devices (file formats) - line type, details scalable (not good for plots with many points - file size huge as all point represented by information)
#pdf
#svg - XML based and good for animation, interactivity, web based plots
#win.metafile - v. old
#postscript - older, not so used on windows

##bitmap devices (file formats) - pixel based - good for plots with many different points - Do not resize well - distorts images
#png - portable network graphics (- good for line graphs of images with solid lossless compression algorithm, file sizes small. good for web based plots)
#jpeg (good for natural scenes and gradients - bad for line graphs as you see aliasing)
#tiff (old but still used)
#bmp (for windows, e.g. icons)

#file open and close device
pdf(file = "myPlot.pdf")
#...plot ...
dev.off()
# Can open multiple devices
windows()
#current device
dev.cur
windows()
dev.cur
#set the active device (input param = integer >=2) 
dev.set(2)
with(faitful, plot(eruptions, waiting))
title(main = "Old faithful Geyser Data")
dev.copy(png, file = "geyserplot.png")
dev.copy2pdf()
dev.off()

#Cluster analysis - define close(distance metric), how to group, how to visualise, how to interpret.
# agglomerative approach = 2 closest things, put together, find next closest
# requires defined distance (distance metric) and merging approach
# result - tree showing how close things are to each other

#Distance metrics 
# 1. continuous (euclidean)
# 2. continuous (correlation similarity)
# 3. binary - manhattan distance - absolute sum of distances in each dimension.  

set.seed(1234)
par(mar = c(0, 0, 0, 0))
x <- rnorm(12, mean = rep(1:3, each = 4), sd = 0.2)
y <- rnorm(12, mean = rep(c(1, 2, 1), each = 4), sd = 0.2)
plot(x, y, col = "blue", pch = 19, cex = 2)
text(x + 0.05, y + 0.05, labels = as.character(1:12))

#Calculate distance between points of dataframe
dataFrame <- data.frame(x=x,y=y)
distxy <- dist(dataFrame)

# dendrograms = hierarchical clustering
hClustering <- hclust(distxy)
plot(hClustering)

# prettier dendrograms

myplclust <- function( hclust, lab=hclust$labels, lab.col=rep(1,length(hclust$labels)), hang=0.1,...){
  ## modifiction of plclust for plotting hclust objects *in colour*!
  ## Copyright Eva KF Chan 2009
  ## Arguments:
  ##    hclust:    hclust object
  ##    lab:        a character vector of labels of the leaves of the tree
  ##    lab.col:    colour for the labels; NA=default device foreground colour
  ##    hang:     as in hclust & plclust
  ## Side effect:
  ##    A display of hierarchical cluster with coloured leaf labels.
  y <- rep(hclust$height,2)
  x <- as.numeric(hclust$merge)
  y <- y[which(x<0)]
  x <- x[which(x<0)]
  x <- abs(x)
  y <- y[order(x)]
  x <- x[order(x)]
  plot( hclust, labels=FALSE, hang=hang, ... )
  text( x=x, y=y[hclust$order]-(max(hclust$height)*hang), labels=lab[hclust$order], col=lab.col[hclust$order], srt=90, adj=c(1,0.5), xpd=NA, ... )
}

myplclust(hClustering, lab = rep(1:5, each = 4), lab.col = rep(1:5, each = 4))

#Prettier dendrograms
#http://rpubs.com/gaston/dendrograms

#Merging points (useful to try both methods)
#1. Averaging (distance between centre of gravity of two clusters)
#2. Complete distance (max distance of point in cluster 1 to a point in cluster 2)

#Heatmap (organise columns - normally as they are observations, even useful if they are variables)
dataFrame <- data.frame(x=x,y=y)
set.seed(143)
dataMatrix <- as.matrix(dataFrame)[sample(1:12), ]
heatmap(dataMatrix)


# Clustering primarily for exploration
# downsides: 
# 1. sensitive to outliers
# 2. sensitive toscalings of variables
# NOTE: Can mitigate by checking different distance metrics

# Clustering positives
# 1. repeatable (deterministic)

#Resources:
# Rafa's Distances and clustering video
# Elements of statistical learning

# K-Means Clustering
# distance metric?

# Partitioning approach - fix 'n' clusters - guess centroids of each cluster

set.seed(1234)
par(mar = c(0, 0, 0, 0))
x <- rnorm(12, mean = rep(1:3, each = 4), sd = 0.2)
y <- rnorm(12, mean = rep(c(1, 2, 1), each = 4), sd = 0.2)
plot(x, y, col = "blue", pch = 19, cex = 2)
text(x + 0.05, y + 0.05, labels = as.character(1:12))

# Algorithm
# assign points to closest centroid
# take new centroid of the cluster (average of points)
# repeat
dataFrame <- data.frame(x, y)
kmeansObj <- kmeans(dataFrame, centers = 3)
names(kmeansObj)
kmeansObj$centers

# plot resulting clusters (centroid as +, datapoints coloured by cluster)
par(mar = rep(0.2, 4))
plot(x, y, col = kmeansObj$cluster, pch = 19, cex = 2)
points(kmeansObj$centers, col = 1:3, pch = 3, cex = 3, lwd = 3)

# plot using heatmap
set.seed(1234)
dataMatrix <- as.matrix(dataFrame)[sample(1:12), ]
kmeansObj2 <- kmeans(dataMatrix, centers = 3)
par(mfrow = c(1,2), mar = c(2, 4, 0.1, 0.1))

image(t(dataMatrix)[, nrow(dataMatrix):1], yaxt = "n")

image(t(dataMatrix)[, order(kmeansObj2$cluster)], yaxt = "n")
image(t(dataMatrix)[, order(kmeansObj2$cluster)], yaxt = "n")

# How to find optimum number of clusters?
# Can use cross-valiation or information theory methods

heatmap(dataMatrix)

#PRINCIPAL COMPONENTS ANALYSIS
#and
#SINGULAR VALUE DECOMPOSITION (ALMOST IDENTICAL RESULTS)

set.seed(12345)
par(mar = rep(0.2, 4))
dataMatrix <- matrix(rnorm(400), nrow = 40)
image(1:10, 1:40, t(dataMatrix)[, nrow(dataMatrix):1])

#heatmap adds hierarchical clustering on columns and rows
heatmap(dataMatrix)

# Add pattern to data
set.seed(678910)
for (i in 1:40) {
  #flip a coin
  coinFlip <- rbinom(1, size = 1, prob = 0.5)
  #if coin is heads add a common pattern to that row
  if (coinFlip) {
    dataMatrix[i, ] <- dataMatrix[i, ] + rep(c(0, 3), each = 5)
  }
}

# now there is a pattern
image(1:10, 1:40, t(dataMatrix)[, nrow(dataMatrix):1])
#pattern is refined
heatmap(dataMatrix)

#Patterns within rows and columns
hh <- hclust(dist(dataMatrix))
dataMatrixOrdered <- dataMatrix[hh$order, ]
par(mfrow = c(1, 3))
image(t(dataMatrixOrdered)[, nrow(dataMatrixOrdered):1])
plot(rowMeans(dataMatrixOrdered), 40:1, , xlab = "Row Mean", ylab = "Row", pch = 19)
plot(colMeans(dataMatrixOrdered), xlab = "Column", ylab = "Column Means", pch = 19)

# Principal components analysis- for mulivariate variables X1,...Xn, so X1 = (X11, ... X1m)
# Data Compression = find lower rank matrix (fewer variables)
# Statistical consistency = find subset of variables which are uncorrelated 
# ... and explain as much of the variance of the original matrix as possible

# One way = SVD
# If col = variable, row = observation
# X = U D V^T   (Transpose of V)
# U orthogonal (left singular values) D (Diagonal) V orthogonal (right singular values)
# principal componenets = V (right singular values) after you scale (- mean & divide by SD)

svd1 <- svd(scale(dataMatrixOrdered))
par(mfrow = c(1, 3))
image(t(dataMatrixOrdered)[, nrow(dataMatrixOrdered):1])
plot(svd1$u[, 1], 40:1, , xlab = "Row", ylab = "First left singular vector", pch = 19)
plot(svd1$v[, 1], xlab = "Column", ylab = "First right singular vector", pch = 19)

# SVD 1st component has highest D value (descending order of "variance explained")
par(mfrow = c(1, 2))
plot(svd1$d, xlab = "Column", ylab = "Singular value", pch = 19)
plot(svd1$d^2/sum(svd1$d^2), xlab = "Column", ylab = "Prop. of variance explained", pch = 19)

#PCA = principal component analysis almost identical to SVD
par(mar = c(5,5,2,1), mfrow = c(1, 1))
svd1 <- svd(scale(dataMatrixOrdered))
pcal1 <- prcomp(dataMatrixOrdered, scale = TRUE)
plot(pcal1$rotation[, 1], svd1$v[, 1], pch = 19, xlab = "Principal Component 1", ylab = "Right Singular Vector 1")
abline(c(0,1))

# Sometimes all variance can be explained with a single variable (vector)
constantMatrix <- dataMatrixOrdered*0
for(i in 1:dim(dataMatrixOrdered)[1]){constantMatrix[i,] <- rep(c(0,1), each = 5)}
svd1 <- svd(constantMatrix)
par(mfrow = c(1,3))
image(t(constantMatrix)[,nrow(constantMatrix):1])
plot(svd1$d, xlab = "Column", ylab="Singular value", pch=19)
plot(svd1$d^2/sum(svd1$d^2), xlab = "Column", ylab="Prop. of variance explained", pch=19)

#Second pattern
set.seed(678910)
for (i in 1:40) {
  #flip a coin
  coinFlip1 <- rbinom(1, size = 1, prob = 0.5)
  coinFlip2 <- rbinom(1, size = 1, prob = 0.5)
  #if coin is heads add a common pattern to that row
  if (coinFlip1) {
    dataMatrix[i, ] <- dataMatrix[i, ] + rep(c(0, 5), each = 5)
  }
  if (coinFlip2) {
    dataMatrix[i, ] <- dataMatrix[i, ] + rep(c(0, 5), times = 5)
  }
}
hh <- hclust(dist(dataMatrix))
dataMatrixOrdered <- dataMatrix[hh$order, ]

# SVD - true patterns
par(mfrow = c(1, 3))
# TRUTH
image(t(dataMatrixOrdered)[, nrow(dataMatrixOrdered):1])
plot(rep(c(0,1), each = 5), pch = 19, xlab = "Column", ylab = "Pattern 1")
plot(rep(c(0,1), times = 5), pch = 19, xlab = "Column", ylab = "Pattern 2")

# SVD Trying to recover 2 patterns 
svd2 <- svd(scale(dataMatrixOrdered))
par(mfrow = c(1,3))
image(t(dataMatrixOrdered)[, nrow(dataMatrixOrdered):1])
plot(svd2$v[, 1], xlab = "Column", ylab = "First right singular vector", pch = 19)
plot(svd2$v[, 2], xlab = "Column", ylab = "Second right singular vector", pch = 19)

# D and variance explained
svd1 <- svd(scale(dataMatrixOrdered))
par(mfrow = c(1, 2))
plot(svd1$d, xlab = "Column", ylab = "Singular value", pch = 19)
plot(svd1$d^2/sum(svd1$d^2), xlab = "Column", ylab = "Percent of variance explained", pch = 19)
# SVD struggles to pick up the alternating pattern, so does not well narrow down to 2 variables, instead finding 5

# MISSING VALUES - prevents running SVD or PCA (prcomp)
dataMatrix2 <- dataMatrixOrdered
# Randomly insert some missing data
dataMatrix2[sample(1:100, size = 40, replace = FALSE)] <- NA
svd1 <- svd(scale(dataMatrix2)) ## NOW SVD DOESN'T WORK

# Impute
#source("http://bioconductor.org/biocLite.R")
#biocLite("impute")
#library(impute) ## Available from http://bioconductor.org
dataMatrix2 <- dataMatrixOrdered
dataMatrix2[sample(1:100, size = 40, replace = FALSE)] <- NA
dataMatrix2 <- impute.knn(dataMatrix2)$data
svd1 <- svd(scale(dataMatrixOrdered)); svd2 <- svd(scale(dataMatrix2))
par(mfrow= c(1,2)); plot(svd1$v[,1],pch=19); plot(svd2$v[,1],pch=19)

# Face data example
load("face.rda")
image(t(faceData)[, nrow(faceData):1])

svd1 <- svd(scale(faceData))
plot(svd1$d^2/sum(svd1$d^2), xlab = "Singular vector", ylab = "Variance explained", pch = 19)

## Note that %*% is matrix multiplication

#Here svd1$d[1] is a constant
approx1 <- svd1$u[, 1] %*% t(svd1$v[, 1]) * svd1$d[1]

# in these examples we need to make the diagonal matrix out of d
approx5 <- svd1$u[, 1:5] %*% diag(svd1$d[1:5]) %*% t(svd1$v[, 1:5])
approx10 <- svd1$u[, 1:10] %*% diag(svd1$d[1:10]) %*% t(svd1$v[, 1:10])

par(mfrow = c(1,4))
image(t(approx1)[, nrow(approx1):1], main = "(a)")
image(t(approx5)[, nrow(approx5):1], main = "(b)")
image(t(approx10)[, nrow(approx10):1], main = "(c)")
image(t(faceData)[, nrow(faceData):1], main = "(d)") ## Original data

#Notes for PCA/ SVD
# Scale matters
# May mix real patterns (not fully separate)
# Can be computationally intensive
# OTHER APPROACHES - Factor Analysis, Independent components analysis, Latent Semantic analysis
# references


# Changing color schemes in R plots
# default col [1,2,3] = [black, red, green]
# heat.colors() palette = red to white via yellow
# topo.colors() blue to green to yellow - topology

#grDevices package
# - colorRamp         (takes palette and returns values between 0 and 1 indicating extremes of palette see 'gray' )
# - colorRampPalette  (similar but returns a vecotr of colours interpolating the palette - like heat.colors() or topo.colors())
# interpolates between primary colours which are inputted in list of names using colors()

pal <- colorRamp(c("red", "blue")) ## (RGB) 1 = red, 2 = green, 3 = blue 
pal(0)   ## 255 0 0 
pal(1)   ## 0 0 255
pal(0.5) ## 127.5 0 127.5

pal(seq(0,1, len = 10))

# colorRampPalette - take INTEGER inputs - returns hexadecimal character representation evenly spaced (like inputting seq above)
pal <- colorRampPalette(c("red","yellow"))
pal(2)
pal(10)

#FF = 255 = max number

# RColorBrewer Package creates palettes for you
# 3 types: 
# sequential (ordered) - e.g. Blues, BuPu, OrRd, YIOrRd
# diverging (deviate from a mean/ 0/ ...) - light in middle - e.g. Spectral, PiYG, RdBu, RdYIGn
# Qualitative (non ordered (categorical/ factors)) - e.g. Set3, Set1, Dark2, Accent

# Palettes can then be used as input to colorRamp() and colorRampPalette()

library(RColorBrewer)
?brewer.pal
cols <- brewer.pal(3, "BuGn")
cols
pal <- colorRampPalette(cols)
image(volcano, col = pal(20))

# SmoothScatter function - useful if LOT (10k+) of points
# 2D histogram - default "Blues" palette
x <- rnorm(10000)
y <- rnorm(10000)
smoothScatter(x,y)

# rgb function created manual hexadecimal string
# alpha parameter adds transparency # effect like 1D histogram "per pixel"
# colorspace pacakge can give different control over colours - NOT TALKED ABOUT HERE

plot(x,y, pch=19)
plot(x,y, col = rgb(0,0,0,0.2),pch=19)
