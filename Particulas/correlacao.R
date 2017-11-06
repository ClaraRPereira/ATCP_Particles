train_data <- read.csv(file="training.csv")
train_data$EventId <- NULL
train_data$Label <- NULL
train_data$Weight <- NULL
library(corrplot)
cMatrix <- cor(train_data)
pdf(file="matriz_correlacao__.pdf")
corrplot(cMatrix, method="square",type="upper",tl.col="black",diag=FALSE)
dev.off()

