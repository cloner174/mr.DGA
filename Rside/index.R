data = read.csv('dataset.csv')
dim(data)
data1 <- data[1:17943,]
data2 <- data[17944:35887,]
write.csv(data1,'data1.csv')
write.csv(data2,'data2.csv')


data1 <- read.csv('data1.csv')

data2 <- read.csv('data2.csv')

head(data1)

class(data1)

unique(data1$emotion)
unique(data2$emotion)

library(ggplot2)

ggplot(data,aes(x = emotion)) +
  geom_histogram(bins = 100)
