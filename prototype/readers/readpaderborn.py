"""
  This script processes data files that were made available from the Paderborn repository:
  https://en.cs.uni-paderborn.de/de/is/research/research-projects/software/label-ranking-datasets

  The Paderborn repository contains the standard label ranking benchmark datasets, both
  real and semi-synthetic, as proposed in:

  [1] Cheng, Weiwei, and Eyke Hüllermeier. "Instance-Based Label Ranking using the
      Mallows Model." In ECCBR workshops, pp. 143-157. 2008.

  Most Paderborn datasets have been derived from datasets widely known by the machine
  learning community. To the best of our knowledge, the process of derivation has
  not been documented, but it can be shown that at least some Paderborn datasets:
  1) have probably been derived by applyng min-max[-1, 1] to the original dataset
     (e.g., iris and wine datasets), while others have been derived by applying
     z-score normalisation (e.g., bodyfat);
  2) include all the features from the original dataset (e.g., iris and wine
     datasets), while others include just a subset of the original features (e.g.,
     bodyfat);
  3) have feature values expressed with a precision that, although compatible with
     a normalisation procedure, differs significantly from the precision with which
     the original data were reported. For example, feature values of the original
     Iris dataset are reported in centimeters with single decimal precision, but
     its Paderborn version reports adimensional feature values with six decimals.
     It must be said that it is not possible to determine the original values of
     each feature after normalisation is applied (and, thus, neither is it
     possible to determine their original precision) unless additional metadata
     is included to the dataset -- and this absence leads us to the next point;
  4) miss significant metadata, such as feature and target names, description and
     traceable history of the dataset (e.g., citation of the original article),
     and the minimal data required to reproduce the process that was used to
     transform the original dataset into its corresponding version in the
     Paderborn repository.

  These facts make it difficult to train the Polygrid model with data from the
  Paderborn datasets, since Polygrid expects counting data. For this reason, this
  script converts some data files into the format required by Polygrid CLI (i.e.,
  pickled Bunch object, the standard used in sklearn.datasets). It looks for data
  files in a local filesystem indicated in the config file for reader scripts
  (e.g., readers_T01_C0.cfg). Selected data files are recovered from
  [basepath]/[repository_i]/[sourcefolder]/*.txt, the original feature matrix of
  the dataset is recovered, and then results are saved to
  [basepath]/[repository_i]/[targetfolder] as a pickle file. Only files with
  parameters defined in filename2pars are considered.

"""

import sys
import traceback
import openml
import numpy as np

from os            import listdir, makedirs, remove
from os.path       import join, isfile, exists
from customDefs    import setupEssayConfig, getEssayParameter, listEssayConfig
from customDefs    import tsprint, loadAsText, saveLog, resetLog, serialise
from polygrid      import ECO_CAPACITY, ECO_DEFICIT
from sklearn.utils import Bunch

# This 'file to parameters' map has been manually set for each Paderborn dataset because
# they require comparing sizing and identity of different objects across both the Paderborn
# and the OpenML repositories

filename2pars = {
  # filename -> (#rows, #features, #labels, openmlid, target_attr, measurand_type)
  #'analcatdata-authorship_dense.txt':
  #                        (841,   70,  4, 42834,               '___', ECO_CAPACITY),
  #'bodyfat_dense.txt':    (252,    7,  7,   560,              'class', ECO_CAPACITY),
  #'calhousing_dense.txt': (20640,  4,  4,   537, 'median_house_value', ECO_CAPACITY),
  #'cpu-small_dense.txt':  (8192,   6,  5,     0,               '___', ECO_CAPACITY),
  #'elevators_dense.txt':  (16599,  9,  9,     0,               '___', ECO_CAPACITY),
  #'fried_dense.txt':      (40768,  9,  5,     0,               '___', ECO_CAPACITY),
  #'glass_dense.txt':      (214,    9,  6,     0,               '___', ECO_CAPACITY),
  #'housing_dense.txt':    (506,    6,  6,     0,               '___', ECO_CAPACITY),
  'iris_dense.txt':       (150,    4,  3,    61,              'class', ECO_CAPACITY),
  #'pendigits_dense.txt':  (10992, 16, 10,     0,               '___', ECO_CAPACITY),
  #'segment_dense.txt':    (2310,  18,  7,     0,               '___', ECO_CAPACITY),
  #'stock_dense.txt':      (950,    5,  5,     0,               '___', ECO_CAPACITY),
  #'vehicle_dense.txt':    (846,   18,  4,     0,               '___', ECO_CAPACITY),
  'vowel_dense.txt':      (528,   10, 11,  42865,                None, ECO_CAPACITY),
  'wine_dense.txt':       (178,   13,  3,    187,             'class', ECO_CAPACITY),
  #'wisconsin_dense.txt':  (194,   16, 16,     0,               '___', ECO_CAPACITY),
  }

"""
  In the following code, we adopted some conventions:

  Z represents the feature matrix recovered from a Paderborn data file
  X represents the feature matrix recovered from the OpenML repository (using openmlid)
  W is computed by applying a normalisation scheme over X (plus some additional changes)

"""


def get_float_precision(x):
  s = str(x)
  return len(s) - (s.index('.') + 1)

def get_feature_matrix_stats(Z):
  (_,d) = Z.shape
  z_min = Z.min(axis=0)
  z_max = Z.max(axis=0)
  p_z = int(np.median([get_float_precision(Z[0,i]) for i in range(d)]))
  z_mu = np.round(Z.mean(axis=0), decimals=p_z)
  return (z_min, z_max, z_mu, p_z)

def guess_applied_normalisation(Z, z_min, z_max, z_mu, p_z):
  mrd = 10**(-p_z) # assumes a minimal representable difference
  if(np.all(z_min == -1.) and np.all(z_max == 1.)):
    normalisation = 'min-max [-1, 1]'
  elif(np.all(z_min < 0.) and np.all(z_max > 0.) and np.all(abs(z_mu) <= mrd)):
    normalisation = 'z-score'
  else:
    normalisation = 'unknown'
  return normalisation

def apply_normalization(X, normalisation):
  if(normalisation == 'min-max [-1, 1]'):
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    W = (X - x_min)/(x_max - x_min) * 2 - 1
  elif(normalisation == 'z-score'):
    x_mu  = X.mean(axis=0)
    x_std = X.std(axis=0, ddof=1)
    W = (X-x_mu)/x_std
  else:
    W = X
  return W

def map_Zcols_to_Xcols(Z, X, normalisation):
  # Z is the feature matrix from a Paderborn data file
  # X is the feature matrix recovered from OpenML repository

  W = apply_normalization(X, normalisation)
  if(normalisation == 'min-max [-1, 1]'):
    # after min-max normalisation,
    # - the min/max statistics are uninformative, but the mean statistic is informative
    # so we resort to the mean statistic
    w = W.mean(axis=0)
    z = Z.mean(axis=0)

  elif(normalisation == 'z-score'):
    # after z-score normalisation,
    # - the mean statistic is uninformative, but min/max statistics are informative
    # so we resort to the one of them (min statistic)
    w = W.min(axis=0)
    z = Z.min(axis=0)

  cols  = []
  (_,d) = Z.shape
  (_,D) = X.shape
  (success, errmsg) = (True, '')

  # for each column of Z, finds the column of X with smaller difference in the statistic
  for i in range(d):
    L = sorted([(j, (w[j] - z[i])**2) for j in range(D)], key=lambda e:e[1])
    cols.append(L[0][0])
  if(len(set(cols)) != d):
    success = False
    errmsg = 'Unable to map columns of Z to columns of X'

  return (success, errmsg, cols, W[:,cols])

def load_Paderborn_datafile(sourcepath, filename, n_rows, n_features, n_labels):

  try:
    raw = loadAsText(join(*sourcepath, filename)).strip().split('\r\n')
    (Z, Y, row_counter) = ([], [], 3)
    for line in raw[3:]: # assumes first 3 rows of Paderborn dataset files are header rows
      buffer = line.split('\t')
      # assumes all feature columns precede the target columns
      # assumes all features are encoded as float values
      # assumes all targets  are encoded as integer values
      features = [float(val) for val in buffer[0:n_features]]
      labels   = [int(val)   for val in buffer[n_features:]]
      Z.append(features)
      Y.append(labels)
      row_counter += 1
  except:
    raise ValueError(f'Failed parsing file {filename} at row #{row_counter}: {line}')

  # creates the feature matrix X and the target matrix Y from the recovered data
  Z = np.array(Z, dtype=np.float64)
  Y = np.array(Y, dtype=int)

  # checks if recovered target labels match the expected set of labels
  # assumes all targets are encoded as integet values starting at 1
  # assumes targets encode complete label rankings (i.e., symbol for absence is not required)
  all_labels = sorted(set(Y.flatten().tolist()))
  is_integer_sequence = True
  for i in range(n_labels):
    if(i+1 not in all_labels):
      is_integer_sequence = False
      break
  if(is_integer_sequence):
    # adopts the sklearn.datasets standard and recodes targets as integers starting at zero
    Y = Y - 1
    all_labels = sorted(set(Y.flatten().tolist()))
  else:
    raise ValueError('Class indicators could not be safely translated to begin at zero')

  # checks if recovered feature matrix matches the expected dimensions
  (m,d) = Z.shape
  if(m != n_rows):
    raise ValueError(f'Inconsistent number of rows recovered: expected {n_rows}, found {m}')
  elif(d != n_features):
    raise ValueError(f'Inconsistent number of features recovered: expected {n_features}, found {d}')

  # checks if recovered target matrix matches the expected dimensions
  (m,n) = Y.shape
  if(m != n_rows):
    raise ValueError(f'Inconsistent number of rows recovered: expected {n_rows}, found {m}')
  elif(n != n_labels):
    raise ValueError(f'Inconsistent number of labels recovered: expected {n_features}, found {n}')

  return (Z, Y, all_labels)

def get_original_data(openmlid, target_attr):
  """
  Recovers an image of the current dataset from the OpenML repository, and returns
  the original level of the features of the dataset (i.e., matrix X), as well as the
  available metadata (i.e., description, feature and target names)
  Parameters:
  - openmlid is the dataset id in openml.org
  - target_attr is the name of the target field/column
  Returns:
  - success, if the required dataset is recovered
  - errmsg, in case of failure, with a string describing the issue
  - descr, a string describing the dataset
  - X, the feature matrix with features expressed in their original levels
  - feature_names, and
  - target_names.
  NOTE: the original targets are not recovered because we are interested in using the
        rankings in the Paderborn datasets
  """

  (success, errmsg) = (True, '')
  try:

    mirror = openml.datasets.get_dataset(openmlid,
                                         download_data = True,
                                         download_qualities = True,
                                         download_features_meta_data = True,
                                        )

    if(target_attr is None):
      (X_, y, u, v) = mirror.get_data()
      (col_features, feature_names) = list(zip(*[(i, feature_name) for (i, feature_name) in enumerate(v) if u[i] == False]))
      (col_targets, target_names)  = list(zip( *[(i, target_name)  for (i, target_name)  in enumerate(v) if u[i] == True]))
      X_ = X_.to_numpy()
      X = X_[:,col_features].astype(float)
      X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
      y = X_[:,col_targets].astype(int)
      descr = mirror.description

    else:
      (X, y, u, v) = mirror.get_data(target=target_attr)
      X = X.to_numpy()  # from pandas dataframe to numpy array
      y = y.to_numpy()  # from pandas dataframe to numpy array
      feature_names = v # already a list of strings
      target_names  = mirror.retrieve_class_labels()
      descr = mirror.description

  except Exception as e:
    errmsg = traceback.format_exc()
    success = False
    descr = None
    X = None
    feature_names = None
    target_names = None

  return (success, errmsg, descr, X, feature_names, target_names)

def main(configfile):

  setupEssayConfig(configfile)
  basepath     = getEssayParameter('PARAM_PADERBORN_BASEPATH')
  sourcefolder = getEssayParameter('PARAM_PADERBORN_SOURCEFOLDER')
  targetfolder = getEssayParameter('PARAM_PADERBORN_TARGETFOLDER')
  repositories = getEssayParameter('PARAM_PADERBORN_REPOSITORIES')

  for repository in repositories:

    tsprint(__doc__)
    tsprint(f'Parameters loaded from {configfile}:\n{listEssayConfig()}')

    sourcepath = basepath + [repository, sourcefolder]
    targetpath = basepath + [repository, targetfolder]

    # ensures the target folder is available and empty
    if(exists(join(*targetpath))):
      for f in listdir(join(*targetpath)):
        remove(join(*targetpath, f))
    else:
      makedirs(join(*targetpath))

    for filename in filename2pars:
      if(isfile(join(*sourcepath, filename))):

        tsprint( '--------------------------------------------------------------------------------')
        tsprint(f'Processing Paderborn data file {filename}')
        tsprint( '--------------------------------------------------------------------------------')

        # recovers data from the Paderborn data file
        (m, d, n, openmlid, target_attr, measurand_type) = filename2pars[filename]
        (Z, Y, all_labels) = load_Paderborn_datafile(sourcepath, filename, m, d, n)
        (z_min, z_max, z_mu, p_z) = get_feature_matrix_stats(Z)
        np.set_printoptions(precision=p_z, suppress=True)

        tsprint( '-- feature matrix Z is {0} by {1}, precision {2}'.format(m,d,p_z))
        tsprint(f'   z_min {z_min.shape}: {z_min}')
        tsprint(f'   z_max {z_max.shape}: {z_max}')
        tsprint(f'   z_mu  {z_mu.shape}: {z_mu}')
        tsprint(f'   set of labels: {all_labels}')

        # identifies the type of normalisation scheme that was applied to the original data
        # to produce its Paderborn version (by means of a heuristics)
        normalisation = guess_applied_normalisation(Z, z_min, z_max, z_mu, p_z)
        tsprint(f'   the dataset seems to have been normalised by {normalisation}')

        # recovers an image of the dataset that is taken as the original data
        tsprint(f'-- recovering alternative dataset image (openmlid = {openmlid})')
        (success, errmsg, descr, features, feature_names, target_names) = get_original_data(openmlid, target_attr)
        if(success):

          X = features

          if(feature_names is None or len(feature_names) == 0):
            feature_names = ['A{0}'.format(k+1) for k in range(d)]
            tsprint('** WARNING - original feature names were not recovered, resorting to default')
          tsprint(f'   feature names: {feature_names}')

          if(target_names is None or len(target_names) == 0):
            target_names  = ['C{0}'.format(j+1) for j in range(n)]
            tsprint('** WARNING - original target names were not recovered, resorting to default')
          tsprint(f'   target names: {target_names}')

          (M,D) = X.shape
          (x_min, x_max, x_mu, p_x) = get_feature_matrix_stats(X)
          np.set_printoptions(precision=p_x, suppress=True)
          tsprint(f'   feature matrix X is {M} by {D}, precision {p_x}')
          tsprint(f'   x_min {x_min.shape}: {x_min}')
          tsprint(f'   x_max {x_max.shape}: {x_max}')
          tsprint(f'   x_mu  {x_mu.shape}: {x_mu}')

        else:
          tsprint( '   --------------------------------------------------------------------------------')
          tsprint(f'   WARNING - original data could not be retrieved')
          tsprint(errmsg)
          tsprint( '   --------------------------------------------------------------------------------')

          X             = Z
          descr         = filename
          feature_names = ['A{0}'.format(k+1) for k in range(d)]
          target_names  = ['C{0}'.format(j+1) for j in range(n)]

        # if mirror image was successfully recovered, maps each column of the Paderborn
        # feature matrix Z to a column of the recovered feature matrix X
        if(success):

          tsprint(f'-- applying {normalisation} normalisation, column matching, and precision adjustment to the recovered data')
          (success, errmsg, cols, W) = map_Zcols_to_Xcols(Z, X, normalisation)
          if(success):
            X = X[:, cols]
            (M,D) = X.shape

            # applying precision adjustment to the recovered data
            # (matrix W is an attempted reconstruction of Z by applying normalisation to X)
            (w_min, w_max, w_mu, p_w) = get_feature_matrix_stats(W)
            W = np.round(W, decimals=p_z)
            #w_mu = np.round(W.mean(axis=0), decimals=p_z)
            np.set_printoptions(precision=p_z, suppress=True)

            tsprint(f'   the following columns of X were selected: {cols}')
            tsprint(f'   feature matrix W is {M} by {D}, precision {p_z}')
            tsprint(f'   w_min {w_min.shape}: {w_min}')
            tsprint(f'   w_max {w_max.shape}: {w_max}')
            tsprint(f'   w_mu  {w_mu.shape}: {w_mu}')

            # lists diverging pairs of instances in terms of feature levels
            mrd = 10**(-p_z) # minimal representable difference (at precision p_z)
            tsprint('-- comparing feature matrices recovered from Paderborn data file (Z) and OpenML (W)')
            D = Z - W
            diff = np.linalg.norm(D)
            tsprint(f'   Frobenius norm: {diff}')
            idxs = np.where((D*D).sum(axis=1) > mrd)[0]
            tsprint(f'   {len(idxs)} exceptions found:')
            for i in idxs:
              tsprint(f'   row {i}:')
              tsprint(f'      Z[{i}] = {Z[i]}')
              tsprint(f'      W[{i}] = {W[i]}')

          else:
            raise ValueError(errmsg)

        # collects the gathered data into a bunch object and serialises it
        x_min = X.min(axis=0)
        #assert np.all(x_min >= 0.)
        caseIDs       = ['P{0}'.format(i)   for i in range(m)]
        newBunch = Bunch(
            DESCR=filename,
            feature_names = feature_names,
            target_names  = target_names,
            #factor_names  = [],
            data    = X,
            #factors = None,
            target  = Y,
            caseIDs = caseIDs,
            symbol  = measurand_type,
        )

        basename = filename.split('.')[0]
        basename = basename.replace('_dense', '@pb')
        serialise(newBunch, join(*targetpath, basename))

    # saves the execution log
    logfile = 'readpaderborn.log'
    tsprint('')
    tsprint('This report was saved to {0}'.format(join(*targetpath, logfile)))
    tsprint('Done.')
    saveLog(join(*targetpath, logfile))
    resetLog()

if(__name__ == '__main__'):

  main(sys.argv[1])
