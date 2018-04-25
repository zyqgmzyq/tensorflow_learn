import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.enable_eager_execution()

# 1.Write one or more dataset importing functions.
def input_fn(dataset):
   ...  # manipulate dataset, extracting feature names and the label
   return feature_dict, label


# 2.Define the feature columns
# Define three numeric feature columns.
population = tf.feature_column.numeric_column('population')
crime_rate = tf.feature_column.numeric_column('crime_rate')
median_education = tf.feature_column.numeric_column('median_education',
                    normalizer_fn='lambda x: x - global_education_mean')


# 3.Instantiate the relevant pre-made Estimator
# Instantiate an estimator, passing the feature columns.
estimator = tf.estimator.Estimator.LinearClassifier(
    feature_columns=[population, crime_rate, median_education],
    )


# 4.Call a training, evaluation, or inference method.
# my_training_set is the function created in Step 1
estimator.train(input_fn=my_training_set, steps=2000)

