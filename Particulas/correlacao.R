
train_data <- read.csv(file="training.csv")
train_data$EventId <- NULL
train_data$Label <- NULL
train_data$Weight <- NULL
library(corrplot)
cMatrix <- cor(train_data)
corrplot(cMatrix, method="color",type="upper",tl.col="black",diag=FALSE)
dev.print(pdf, 'matriz_correlacao.pdf')
dev.off()
corrplot(cMatrix, method="color",type="upper", order="hclust",tl.col="black",diag=FALSE)
dev.print(pdf, 'matriz_correlacao_hierarquia.pdf')
dev.off()


