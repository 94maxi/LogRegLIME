# LogRegLIME
Logistic Regression Sentiment Analysis With Lime Analysis
To build the logistic regression , firstly a dataframe is created with binary labels -1
and 1, labelling all negative reviews as -1 and all positive reviews as 1. Next to each label
is its respective the text review data. The text passes through a pre-processing function
and lemmatizer. The pre-processing function cleans the text, removing punctuation and
turning all letters to lowercase. The lemmatizer function used the WordNetLemmatizer function from the nltk package. The dataframe is then split into two, a training
dataframe with 80% of data, and a test dataframe with the remaining 20%. A count
vectorizer from the Sci-Kit Learn package is applied within the data frame . It takes
the documents within the dataframe, applies a word count and tokenizes them into
vectors. At this point the logistic regression is ran with the training data. The logistic
regression is ran with Sci-Kit Learn package. Limited-memory BFGS (L-FBGS) is used
as the optimization algorithm. I suggesst L-FBGS for problems with high
dimensional vectors since it only uses approximations on a select quantity of historical
states to determine the step directions. It is also a quasi-Newtonian function, meaning
that it approximates the full Hessian matrix, making it less computationally expensive.
There is a tendency for logistic regressions to overfit high-dimensional vectors, however
aggressive regularization functions can harm the predictive capacities of a regression
using a L-FBGS optimization algorithm. It is crucial to find the right regularization term, so a quick search is ran to find the difference in between training and
testing accuracy for l1 and l2 regularization terms at levels (10,1, 0.1,0.001).
