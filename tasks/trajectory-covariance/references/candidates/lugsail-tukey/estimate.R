args <- as.integer(commandArgs(trailingOnly=TRUE))
stopifnot(packageVersion("mcmcse") == "1.5.0")
input <- file("stdin", "rb")
x <- matrix(readBin(input, "double", n=prod(args), endian="little"), nrow=args[1], ncol=args[2], byrow=TRUE)
close(input)
estimate <- mcmcse::mcse.multi(x, method="tukey", r=3)$cov
output <- file("/dev/stdout", "wb")
writeBin(as.double(t(estimate)), output, endian="little")
close(output)
