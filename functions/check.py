import tensorflow as tf

#this function is to evaluate the performance of the model.

#define the dataset used in data arg
#define the model used in model arg
#define the number of verbose desired to be used in verbose arg

def score(data, model, verbose):
    score = model.evaluate(data, verbose=verbose)
    print("Loss:", score[0], "\n", "Accuracy:", score[1])
