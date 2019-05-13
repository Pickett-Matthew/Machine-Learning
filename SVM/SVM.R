library(e1071)

letters <- read.csv("letters.csv", header=TRUE)

allRows <- 1:nrow(letters)
testRows <- sample(allRows, trunc(length(allRows) * 0.3))

lettersTest <- letters[testRows,]
lettersTrain <- letters[-testRows,]


model <- svm(letter~., data = lettersTrain, kernal = "radial", gamma = 0.01, cost = 1000, C = 3)

prediction <- predict(model, lettersTest[-1])
confusionMatrix <-table(pred = prediction, true = lettersTest$letter)

agreement <- prediction == lettersTest$letter
accuracy <- prop.table(table(agreement))

print(confusionMatrix)
print(accuracy)


