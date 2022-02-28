Assignment 1 Example Code
================

We first read in train and test data

    train = read.csv("census-income-training.csv")
    test = read.csv("census-income-test.csv")

And check dimension of data

    print (dim(train))

    ## [1] 142863     40

    print (dim(test))

    ## [1] 71739    40

We train our simplest model. Here I am just using age as the single
predictor, and logistic regression as the model.

The outcome is `income_morethan_50K`

    m <- glm(income_morethan_50K ~AAGE, train, family = "binomial")

Then we make predictions on test data

    # get predicted probability
    predict_probability = predict(m, newdata = test, type = "response")

turn predicted probabilities into predicted categories, if predicted
probability is bigger than a threshold *ϕ*. Below I set *ϕ* = 0.5

    predict_Y = ifelse(predict_probability>0.5, a,0)
    print (table(predict_Y))

    ## predict_Y
    ##     0 
    ## 71739

As you can see, every data point’s predicted category is “0”. That’s a
*trivial* solution by predicting every data point belongs to the same
majority class. Your prediction should be better than my simplest
prediction.

output
------

You should save your output, Your data should contain 71740 rows

-   The first row is the header, with two columns, one call “Id” and the
    other called “income\_morethan\_50K”.
-   From the second row to the 71739 row,

<!-- -->

    output= data.frame(Id = test$Id, income_morethan_50K = predict_Y)
    write.csv(output, "predictions.csv", row.names = FALSE, quote = F)
