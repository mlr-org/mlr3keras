% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/DataGenerators.R
\name{make_generator_from_xy}
\alias{make_generator_from_xy}
\title{Make a DataGenerator from a x,y matrices}
\usage{
make_generator_from_xy(
  x,
  y,
  generator = keras::image_data_generator(),
  batch_size = 32L,
  shuffle = TRUE,
  seed = 1L,
  subset = NULL,
  ignore_class_split = FALSE
)
}
\arguments{
\item{x}{\code{\link{matrix}} \cr
Numeric matrix of features.}

\item{y}{\code{\link{matrix}} \cr
Numeric matrix of target, already one-hot transformed.}

\item{generator}{\verb{Python Object} \cr
A generator as e.g. obtained from \code{keras::image_data_generator}.
Used for consistent train-test splits.}

\item{batch_size}{\code{\link{integer}} \cr
Batch size.}

\item{shuffle}{\code{\link{logical}} \cr
Should data be shuffled?}

\item{seed}{\code{\link{integer}} \cr
Set a seed for shuffling data.}

\item{subset}{\code{\link{character}} \cr
Should samples be generated from 'training' or 'validation' set?
Only applicable together with a 'generator'.}

\item{ignore_class_split}{\code{\link{logical}} \cr
Test whether all class labels appear in all splits.}
}
\description{
Creates a Python Class that internally iterates over the data.
}
