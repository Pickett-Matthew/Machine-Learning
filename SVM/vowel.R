library(e1071)

vowel <- read.csv("vowel.csv", header=TRUE)

allRows <- 1:nrow(vowel)
testRows <- sample(allRows, trunc(length(allRows) * 0.3))

vowelTest <- vowel[testRows,]
vowelTrain <- vowel[-testRows,]


model <- svm(Class~., data = vowelTrain, kernal = "radial", gamma = 0.01, cost = 1000, C = 3)

prediction <- predict(model, vowelTest[-13])
confusionMatrix <-table(pred = prediction, true = vowelTest$Class)

agreement <- prediction == vowelTest$Class
accuracy <- prop.table(table(agreement))

print(confusionMatrix)
print(accuracy)


