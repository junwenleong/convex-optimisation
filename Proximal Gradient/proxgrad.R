library(gradDescent)
library(pracma)
library(rje)
library(R.matlab)

data <- readMat("proxgraddata.mat") #load Matlab data file

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

proxgrad <- function(par, L, e, maxiter = 10000) { # unaccelerated proximal gradient projection
  numiter <- 1
  X <- par
  oldx <- par
  delfvalue <- delf(X)
  newx <- sign(X - (1/L)*delfvalue)*pmax(Wvec, abs(X - (1/L)*delfvalue) - (0.05/L))
  while (norm(newx - oldx, type = "2") > e && numiter < maxiter) { # abs val wrt stopping criteria epsilon not reached
    numiter <- numiter + 1
    X <- newx
    delfvalue <- delf(X)
    oldx <- newx
    newx <- sign(X - (1/L)*delfvalue)*pmax(Wvec, abs(X - (1/L)*delfvalue) - (0.05/L))
  }
  return (newx)
}

# Lcon <- 0.5 * (norm(data$Xtrain, type = c("F")))^2
# Wvec <- rep(0, 57)
# proxgrad(Wvec, Lcon, 0.001)

