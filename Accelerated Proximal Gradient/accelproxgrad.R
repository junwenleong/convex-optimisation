library(gradDescent)
library(pracma)
library(rje)
library(R.matlab)

data <- readMat("APGdata.mat") #load Matlab data file

delf <- function(w){ #method to compose gradient vector of objective function
  grad <- 0
  for (i in 1:3065){
    yi <- data$ytrain[i,]
    xi <- data$Xtrain[i,]
    expo <- exp(-yi*w%*%xi)
    expo2 <- expo[1,1]
    p1 <- 1 + expo
    pa <- p1[1,1]
    p2 <- -(yi*xi)
    p3 <- (expo2/pa)*p2
    grad <- grad + p3
  }
  return (grad)
}

originalf <- function(w){
  outf <- 0
  for (i in 1:3065){
    xi <- data$Xtrain[i,]
    yi <- data$ytrain[i,]
    p1 <- 1 + exp(-yi * w% * %xi)
    p2 <- log(p1)
    outf <- outf + p2
  }
  return (outf[1,1])
}

accproxgrad <- function(par, L, e, iterbound = 10000) {  # accelerated proximal gradient projection
  oldx <- par
  X <- par
  oldt <- 1
  newt <- 1
  beta <- (oldt - 1)/newt
  gradvec <- delf(X)
  newx <- sign( X - (1/L) * gradvec) * pmax(Wvec, abs(X - (1/L) * gradvec) - (0.05/L)) 
  i <- 1
  while (norm(newx - oldx, type = "2") > e && i < iterbound){ # abs val wrt stopping criteria epsilon not reached
    oldt <- newt
    newt <- 0.5 * (1 + sqrt(1 + 4 * (oldt^2)))
    beta <- (oldt - 1)/newt
    X <- newx + beta * (newx-oldx)
    oldx <- newx
    gradvec <- delf(X)
    newx <- sign(X - (1/L)*gradvec)*pmax(Wvec, abs(X - (1/L)*gradvec) - (0.05/L))
    i <- i + 1
  }
  return (newx)
}

# Lcon <- 0.5 * (norm(data$Xtrain, type = c("F")))^2
# Wvec <- rep(0, 57)
# accproxgrad(Wvec, Lcon, 0.001)


