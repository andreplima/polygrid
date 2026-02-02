import numpy as np

from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.metrics import mean_squared_error, roc_auc_score
from scipy.stats     import kendalltau

#-------------------------------------------------------------------------------------------------------------------------------------------
# Problem-specific definitions: performance metrics
#-------------------------------------------------------------------------------------------------------------------------------------------

def accuracy(Y_real, Y_pred, YorU_hat=None):
  """
  In multilabel tasks, this metric corresponds to subset accuracy.
  In scikit-learn 1.2.2:
  -- Accuracy classification score.
  -- In multilabel classification, this function computes subset accuracy:
     the set of labels predicted for a sample must exactly match the
     corresponding set of labels in y_true.
  """
  return accuracy_score(Y_real, Y_pred)

def f1_micro(Y_real, Y_pred, YorU_hat=None):
  # calculates the metrics globally by counting the total TP, FN, and FP
  # returns (2*TP)/( (TP+FP) + (TP+FN) )
  return f1_score(Y_real, Y_pred, average='micro')

def f1_macro(Y_real, Y_pred, YorU_hat=None):
  # calculates F1 score for each label, and find their unweighted mean.
  # This does not take label imbalance into account.
  return f1_score(Y_real, Y_pred, average='macro', zero_division=0.0)

def f1_weigh(Y_real, Y_pred, YorU_hat=None):
  # calculates metrics for each label, and find their average weighted by support
  # (the number of true instances for each label).
  # This alters ‘macro’ to account for label imbalance;
  # it can result in an F-score that is not between precision and recall.
  return f1_score(Y_real, Y_pred, average='weighted', zero_division=0.0)

def hammingl(Y_real, Y_pred, YorU_hat=None):
  return hamming_loss(Y_real, Y_pred)

def mse(Y_real, Y_pred, YorU_hat=None):
  return mean_squared_error(Y_real, Y_pred)

def ktau(Y_real, Y_pred, YorU_hat=None):
  Y_real_ = Y_real.astype(np.float64)
  Y_pred_ = Y_pred.astype(np.float64)
  Y_real_[Y_real_ == -1.] = np.nan
  Y_pred_[Y_pred_ == -1.] = np.nan
  return kendalltau(Y_real_, Y_pred_, nan_policy='omit').statistic

def lracc(Y_real, Y_pred, YorU_hat=None):
  # computes the fraction of predicted labelsets that are equal to the ground truth
  (m,n) = Y_real.shape
  matches_per_position = (Y_real == Y_pred).astype(int)
  matches_per_sample = (matches_per_position.sum(axis=1) == n).astype(int)
  res = matches_per_sample.sum()/m
  return res

def lrloss(Y_real, Y_pred, YorU_hat=None):
  # computes the fraction of predicted label/position that are equal to the ground truth
  (m,n) = Y_real.shape
  mismatches_per_position = (Y_real != Y_pred).astype(int)
  res = mismatches_per_position.sum()/(m*n)
  return res

#p np.hstack((self.Y_real, self.Y_pred, (self.Y_real == self.Y_pred).astype(int).sum(axis=1)[:,None]))
#p np.hstack((Y_real, Y_pred, (Y_real == Y_pred).astype(int).sum(axis=1)[:,None]))

def mapk(Y_real, Y_pred, YorU_hat=None, k=None):
  # a Kaggle implementation for the "H&M Personalized Fashion Recommendations" challenge
  # https://www.kaggle.com/code/debarshichanda/understanding-mean-average-precision
  def precision_at_k(y_real, y_pred, k):
    intersection = np.intersect1d(y_real, y_pred[:k])
    return len(intersection) / k

  def rel_at_k(y_real, y_pred, k):
    return 1 if y_pred[k-1] in y_real else 0

  def average_precision_at_k(y_real, y_pred, k):
    ap = 0.0
    for i in range(1, k+1):
      ap += precision_at_k(y_real, y_pred, i) * rel_at_k(y_real, y_pred, i)
    return ap / min(k, len(y_real))

  if(k is None):
    (_,k) = Y_real.shape
  vals = [average_precision_at_k(gt, pred, k) for (gt, pred) in zip(Y_real, Y_pred)]
  return np.mean(vals)

def auroc(Y_real, Y_pred, YorU_hat):
  return roc_auc_score(Y_real, Y_pred, average='weighted')

class ClosedRealInterval:

  def __init__(self, lb, ub):

    if(lb > ub):
      raise ValueError(f'ClosedRealInterval expects lb <= ub, but {lb} > {ub}')

    self.lb = lb # lower boundary
    self.ub = ub # upper boundary

  def __str__(self):
    return f'[{self.lb}, {self.ub}]'

  def isempty(self):
    return (np.isnan(self.lb) or np.isnan(self.ub))

  def issingleton(self):
    return (self.lb == self.ub)

  def mu(self):
    return 0. if self.isempty() else (self.ub - self.lb)

  def intersection(self, other):
    if(self.isempty() or other.isempty()):
      # empty intersection
      new_interval = ClosedRealInterval(np.nan, np.nan)
    else:
      (lb, ub) = (max(self.lb, other.lb), min(self.ub, other.ub))
      if(lb > ub):
        # empty intersection
        new_interval = ClosedRealInterval(np.nan, np.nan)
      else:
        new_interval = ClosedRealInterval(lb, ub)
    return new_interval

  def union(self, other):
    if(self.isempty()):
      if(other.isempty()):
        # empty union
        new_interval = ClosedRealInterval(np.nan, np.nan)
      else:
        new_interval = ClosedRealInterval(other.lb, other.ub)
    else:
      if(other.isempty()):
        new_interval = ClosedRealInterval(self.lb, self.ub)
      else:
        new_interval = ClosedRealInterval(min(self.lb, other.lb), max(self.ub, other.ub))
    return new_interval

def cl(s):
  # computes the closure of a finite, discrete set of numbers on the real line (s)
  if(len(s) > 0):
    (lb, ub) = (min(s), max(s))
  else:
    (lb, ub) = (np.nan, np.nan)
  return ClosedRealInterval(lb, ub)

def jaccsim(Y_real, Y_pred, YorU_hat):
  """
  This function computes the average Jaccard index between two discrete sets of scores.
  The two sets refer to the true positive (TP) and true negative (TN) cases for each label.
  The two sets are exclusively informed by the argument Y_real, and the corresponding score
  for each case the argument YorU_hat is informed; Thus, Y_pred is not used.
  It assumes score as being real numbers.
  """
  (_,n) = Y_real.shape
  vals = []
  for j in range(n):

    # identifies the two sets of cases (TP and TN) by their positions in the Y_real matrix
    rows_TP = np.where(Y_real[:,j] == 1)[0]
    rows_TN = np.where(Y_real[:,j] == 0)[0]

    # determines the discrete sets of scores corresponding to each set of cases
    set_TP = set(YorU_hat[rows_TP, j])
    set_TN = set(YorU_hat[rows_TN, j])

    # transforms each discrete set into its closure, here represented by the extremes
    # of the resulting closed real interval
    int_TP = cl(set_TP)
    int_TN = cl(set_TN)

    # computes the Jaccard (similarity) index between the two intervals representing each
    # a closure of the (finite and discrete) set of scores of either the TP or TN cases.
    # more precisely, \frac{\mu(cl(s_{TN}) \cap cl(s_{TP}))}{\mu(cl(s_{TN}) \cup cl(s_{TP}))}
    int_TP_cap_TN = int_TP.intersection(int_TN)
    int_TP_cup_TN = int_TP.union(int_TN)

    if(int_TP_cup_TN.mu() == 0.):
      # the union can only have measure zero if it is a singleton or an empty set
      if(int_TP_cup_TN.issingleton()):
        # in this case, int_TP and int_TN are necessarily the same singleton
        # a reasonable convention is to define the result as the limiting value obtained
        # when two intersecting sets have their measures going to zero
        vals.append(1.)
      else:
        # in this case, both int_TP and int_TN are necesssarily empty sets.
        # this situation is not supposed to occur.
        raise ValueError('jaccsim cannot handle TP and TN when both are empty sets')
    else:
      if(int_TP_cap_TN.mu() == 0.):
        # the union has non-zero measure, but the intersection has zero measure
        # in this case, the intersection is either an empty set or a singleton
        if(int_TP_cap_TN.isempty()):
          # the intersection is an empty set, and the index is zero
          vals.append(0.)
        else:
          # the intersection is a singleton
          # the index is zero if both intervals have non-zero measure
          # (the intervals share a boundary)
          if(int_TP.mu() > 0 and int_TN.mu() > 0):
            vals.append(0.)
          else:
            # the intersection is a singleton because one of the intervals is a
            # singleton which is necessarily a subset of the other interval
            vals.append(1.)

      else:
        # both the union and intersection have non-zero measure
        vals.append(int_TP_cap_TN.mu()/int_TP_cup_TN.mu())

  return np.mean(vals)

