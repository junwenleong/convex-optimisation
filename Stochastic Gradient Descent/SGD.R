library(rje)

LLHgrad <- function(w, X, y) { #gradient of loglikelihood fn for logreg
  n <- dim(X)[1] 
  m <- dim(X)[2] 
  outf <- rep(0, m) 
  for (i in 1:n) { 
    yi <- as.vector(y[i]) 
    Xi <- as.vector(X[i,]) 
    denom <- 1 + exp(yi * (w %*% Xi))
    numer <- yi * Xi 
    outf <- outf - (numer/denom) }
  return(outf) } 

SGD <- function(funct, delf, X, y, xnot, epsil = 0.01, batch_size = 10, steps = 50, max_iter = 500, miu = 0.25, beta = 0.5) {
  w <- xnot 
  iter <- 1 
  while(iter < max_iter) { 
    z <- sample.int(nrow(X), batch_size) 
    Xsam <- X[z,] 
    ysam <- y[z] 
    del <- -1 * delf(w, Xsam, ysam) 
    if (norm(del) < epsil) {break}
    alpha <- armijo(w, del, funct, miu, beta, Xsam, ysam) #armijo line search chosen
    w <- w + alpha * del
    iter <- iter + 1 
    }
  return(w) }
