import os
import numpy as np

from math                   import ceil, floor, log
from copy                   import copy
from collections            import defaultdict

from metrics                import f1_micro, f1_macro, f1_weigh, hammingl, mse
from datasets               import rank2presence, prune_rank
from layoutman              import LayoutManager

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model   import LinearRegression, RidgeCV
from sklearn.tree           import DecisionTreeRegressor
from sklearn.ensemble       import RandomForestRegressor
from matplotlib             import pyplot as plt, colors
from matplotlib.backend_bases import MouseEvent
from skmultilearn.problem_transform import BinaryRelevance

from customDefs import ECO_THRSHLDLVLS
from customDefs import ECO_CUTOFF_SINGLE, ECO_CUTOFF_MULTIPLE
from customDefs import ECO_HIT, ECO_MISS
from datasets   import ECO_DB_MULTICLASS, ECO_DB_MULTILABEL, ECO_DB_LABELRANK

ECO_SPLITNODE2WEIGHT = 2
ECO_LEAFNODE2WEIGHT  = 1
ECO_MAX_DT_RF        = 10

#-------------------------------------------------------------------------------------------------------------------------------------------
# Problem-specific definitions: competing models
#-------------------------------------------------------------------------------------------------------------------------------------------

short_names = {'Polygrid': 'Ply',
               'Linear':   'Lnr',
               'Ridge':    'Rg',
               'MLP':      'MLP',
               'DT':       'DT',
               'RF':       'RF',
               'BRDT':     'BDT',
               'BRRF':     'BRF',
               'Random':   'Rd',
              }

class BaseCompetitor:
  """
  Base class for wrapping linear models from the scikit-learn library to be used
  in the CLI environment.

  -- maxsize is a tuple (nw, n, d) that reflects how the reference model was
     trained: nw is the number of weights used up by the reference model, while n
     and d are related to characteristics of the training dataset. More precisely,
     n is the number of labels, and d is the number of features of the dataset
     used in the training of the reference model.

  -- seed is an integer informed by the CLI environment, and is used as a seed for
     the random number generator instantiated by this model.

  -- sizes is a list of the number of weights that previously fitted instances
     of this model have allocated. In standalone evaluations (e.g., when the user
     issues a 'fit MLP' command in CLI), this list is empty, while in robust evals
     (e.g., when the user issues a 'assess' command), this list is iteratively
     updated after an instance of this model is fitted and evaluated.

  """

  def __init__(self, maxsize, seed=None, sizes=[]):

    # stores parameters and sets the default attribute values
    self.maxsize      = maxsize
    self.rng          = np.random.default_rng(seed)

    self.scenario     = ECO_DB_MULTICLASS
    self.cutoff       = ECO_CUTOFF_MULTIPLE
    self.metric       = f1_micro
    self.avoid_FP     = False # if True, avoids FP during learning_threshold, and avoids
                              # FN otherwise. Maybe one hurts trust more than the other
    self.dataset_name = None
    self.target_names = None
    self.model_name   = None
    self.model        = None
    self.size         = None
    self.tw           = None

    self.m  = None # (int)  number of instances
    self.n  = None # (int)  number of classes/labels
    self.d  = None # (int)  number of features
    self.P  = None # class prototypes
    self.vo = None

    #+++ implement additional code needed to:
    #  + determine a config of this model that would satisfy this.size() ~ nw
    #  + instantiate a model of size just determined
    pass

  def set_scenario(self, value):
    if(value in [ECO_DB_MULTICLASS, ECO_DB_MULTILABEL, ECO_DB_LABELRANK]):
      self.scenario = value
    else:
      raise NotImplementedError(f'Evaluation scenario {self.scenario} not implemented')
    return None

  def set_cutoff(self, value):
    if(value in [ECO_CUTOFF_SINGLE, ECO_CUTOFF_MULTIPLE]):
      self.cutoff = value
    else:
      raise NotImplementedError(f'Threshold mode {value} not implemented')
    return None

  def set_metric(self, fn):
    self.metric = fn

  def set_avoid_FP(self, value):
    self.avoid_FP = value

  def set_dataset_name(self, value):
    self.dataset_name = value
    return None

  def set_target_names(self, value):
    self.target_names = value
    return None

  def learn_thresholds(self, Y, YorU_hat):

    metric = self.metric

    (m,n) = Y.shape
    if(self.cutoff == ECO_CUTOFF_SINGLE):

      if(self.scenario == ECO_DB_MULTICLASS):
        tw = None

      elif(self.scenario in [ECO_DB_MULTILABEL, ECO_DB_LABELRANK]):
        thresholds = np.linspace(YorU_hat.min(), YorU_hat.max(), ECO_THRSHLDLVLS)
        best = np.nan
        watermark = -np.inf

        if(self.scenario == ECO_DB_MULTILABEL):
          Y_real = Y                # Y is already encoded as label presence
        elif(self.scenario == ECO_DB_LABELRANK):
          Y_real = rank2presence(Y) # Y is recoded from label ranking to label presence
        else:
          raise ValueError(f'Support for {self.scenario} not implemented in learningThreshold')

        for threshold in thresholds:
          Y_pred = (YorU_hat >= threshold).astype(int)
          score = metric(Y_real, Y_pred)

          if(self.avoid_FP):
            if(score >= watermark):
              (best, watermark) = (threshold, score)
          else:
            if(score > watermark):
              (best, watermark) = (threshold, score)

        # in single threshold mode, all interventions get the same threshold
        tw = best * np.ones(n)

      else:
        raise ValueError(f'Threshold learning for {self.scenario} tasks is not implemented.')

    else: # in multiple thresholds mode

      if(self.scenario == ECO_DB_MULTICLASS):
        tw = None

      elif(self.scenario in [ECO_DB_MULTILABEL, ECO_DB_LABELRANK]):
        thresholds = np.linspace(YorU_hat.min(), YorU_hat.max(), ECO_THRSHLDLVLS)
        cutoffs = []

        if(self.scenario == ECO_DB_MULTILABEL):
          Y_real = Y                # Y is already encoded as label presence
        elif(self.scenario == ECO_DB_LABELRANK):
          Y_real = rank2presence(Y) # Y is recoded from label ranking to label presence
        else:
          raise ValueError(f'Support for {self.scenario} not implemented in learningThreshold')

        for j in range(n):
          best = np.nan
          watermark = -np.inf
          for threshold in thresholds:
            Y_pred = (YorU_hat[:,j] >= threshold).astype(int)
            score = metric(Y_real[:,j], Y_pred)

            if(self.avoid_FP):
              if(score >= watermark):
                (best, watermark) = (threshold, score)
            else:
              if(score > watermark):
                (best, watermark) = (threshold, score)

          cutoffs.append(best)

        # in multiple thresholds mode, each intervention gets its own threshold
        # (cursor is an analogy to mechanical balances)
        tw = np.array(cutoffs)

      else:
        raise ValueError(f'Not implemented threshold learning for {self.scenario}')

    return tw

  def get_reverse_vorder(self):
    return list(range(self.d))

  def init_weights(self, X, Y, U = None):

    # obtains the average levels of each feature for each class
    # (used in inspect)
    (n,d) = (self.n, self.d)
    self.vo = list(range(d))
    P = np.zeros((n, d))
    if(self.scenario in [ECO_DB_MULTICLASS, ECO_DB_MULTILABEL]):
      for j in range(n):
        idxs = np.where(Y[:, j] == 1)
        P[j] = np.mean(X[idxs], axis=0)

    elif(self.scenario in [ECO_DB_LABELRANK]):
      # -- proposal #2 - each offer equals the average of demands assigned to it
      # -- strategy: uses membership data to deduce label presence
      # -- pros: produces visual content the user can rely on to simulate the task
      #          most similar with multilabel classification scenario
      # -- cons: the more complete the order, the more this scenario converges to proposal #1
      _Y = rank2presence(Y)
      for j in range(n):
        idxs = np.where(_Y[:, j] == 1)
        P[j] = np.mean(X[idxs], axis=0)

    else:
      raise NotImplementedError(f'Scenario {self.scenario} not implemented')

    return P

  def fit(self, X, Y, U=None):

    (m,d) = X.shape
    (m,n) = Y.shape
    (self.m, self.d, self.n) = (m, d, n)

    self.P = self.init_weights(X, Y, U)

    if(self.scenario == ECO_DB_MULTICLASS):
      # learns a regression model to solve f(X) = Y approximately
      self.model.fit(X, Y)
      self.tw = None

    elif(self.scenario == ECO_DB_MULTILABEL):
      # learns a regression model to solve f(X) = Y approximately
      self.model.fit(X, Y)
      Y_hat = self.model.predict(X)
      self.tw = self.learn_thresholds(Y, Y_hat)

    elif(self.scenario == ECO_DB_LABELRANK):
      # learns a regression model to solve f(X) = U approximately
      self.model.fit(X, U)
      U_hat = self.model.predict(X)
      self.tw = self.learn_thresholds(Y, U_hat)

    return self

  def predict(self, X, raw=False, return_scores=False):
    YorU_hat = self.model.predict(X)
    Y_pred = self.post_predict(YorU_hat, raw=raw)
    if(return_scores):
      res = (Y_pred, YorU_hat)
    else:
      res = Y_pred
    return res

  def post_predict(self, YorU_hat, raw=False):
    if(raw):
      Y_pred = YorU_hat
    else:
      (m,n) = YorU_hat.shape

      if(self.scenario == ECO_DB_MULTICLASS):
        Y_pred = (YorU_hat == YorU_hat.max(axis=1)[:,None]).astype(int)

      elif(self.scenario == ECO_DB_MULTILABEL):
        Y_pred = np.zeros(YorU_hat.shape)
        for j in range(n):
          Y_pred[:,j] = (YorU_hat[:,j] >= self.tw[j]) #.astype(int)
        Y_pred = Y_pred.astype(int)

      elif(self.scenario == ECO_DB_LABELRANK):
        Y_pred = (-YorU_hat).argsort()
        Y_pred = prune_rank(Y_pred, YorU_hat, self.tw).astype(int)

    return Y_pred

  def get_size(self):
    return self.size

  def get_config_summary(self, include_model_name=False, details=[]):
    summary = []
    summary.append(f'dataset: {self.dataset_name}')
    summary.append(f'task: {self.scenario}')
    summary.append(f'model: {self.model_name}, with {self.get_size()} weights')
    for other in details:
      summary.append(other)
    if(self.scenario in [ECO_DB_MULTILABEL, ECO_DB_LABELRANK]):
      summary.append(f'cutoff: {self.cutoff}')
    return(summary)

  def inspect(self, X, Y, idxs, tr_idxs):

    # prepares the data
    (m,d) = X.shape
    (m,n) = Y.shape
    coords  = defaultdict(lambda: defaultdict(list)) # coords[j][ECO_HIT or ECO_MISS]
    names   = defaultdict(lambda: defaultdict(list)) # holds the IDs of the datapoints
    cutoffs = []

    # obtains the raw predictions for each sample
    (Y_pred, YorU_hat) = self.predict(X, return_scores = True)
    (xmin, xmax) = (YorU_hat.min(), YorU_hat.max())
    xwidth = xmax - xmin
    thresholds = np.linspace(xmin, xmax, ECO_THRSHLDLVLS).tolist()
    dt = thresholds[1] - thresholds[0]

    """
    This block determines the coordinates/color of each datapoint on the scale for offer j
    --  xs holds the x-coordinates, which are taken as the scores of the respective cases
    -- y1s holds the factors that indicate if a case has beeen allocated to
       -1: the test partition
       +1: the training partition
    -- y2s holds the factors that indicate if a case is a
       +1: a true negative (TN) case for the class j
       +2: a true positive (TP) case for the class j
    -- ys = ys1 * ys2 combines factors to determine the y-coordidate of each datapoint:
       ys \\in {-2, -1, +1, +2} \\mapsto {te/TP, te/TN, tr/TN, tr/TP}

    Works for multilabel classification, but y2s has a different implementation
    for supporting label ranking
    """
    y1s = np.array([1 if i in tr_idxs else -1 for i in idxs])
    yticklabels = ['te/TP', 'te/TN', 'scale', 'tr/TN', 'tr/TP']
    for j in range(n):

      # determines the x-coordinates of the datapoints for the current class j
      xs  = YorU_hat[:,j]

      if(self.scenario in [ECO_DB_MULTICLASS, ECO_DB_MULTILABEL]):

        # determines the y-coordinates of each datapoint for the current class j
        y2s = np.array([2 if Y[i,j] == 1 else 1 for i in range(m)]) # TP -> 2, TN -> 1
        ys  = y1s * y2s

        # summarises the description of the datapoints (position, colour) and cutoffs
        for i in range(m):
          if(Y[i,j] == Y_pred[i,j]):
            # instance i has been correcly predicted (TP/PP or TN/PN)
            case = ECO_HIT
          else:
            # instance i has been wrongly predicted (TP/PN or TN/PP)
            case = ECO_MISS
          names[j][case].append(idxs[i])
          coords[j][case].append((xs[i], ys[i]))

        if(self.scenario in [ECO_DB_MULTILABEL]):
          cutoffs.append(self.tw[j])

      elif(self.scenario == ECO_DB_LABELRANK):

        # determines the y-coordinates of each datapoint for the current class j
        # also prepares two arrays used to decide the colour its datapoints:
        # -- pos_real: the position of the label j in the real ranking
        # -- pos_pred: the position of the label j in the predicted ranking
        y2s      = np.empty(m)
        pos_real = np.empty(m)
        pos_pred = np.empty(m)
        for i in range(m):

          # determines the real and predicted ranks of the current label
          r_real = np.where(Y[i]                       == j)[0].tolist()
          r_pred = np.where(Y_pred[i, Y_pred[i] != -1] == j)[0].tolist()

          # pos_real[i] is the position of the current label in the i-th sample
          # if the current sample is negative for the current label, it stores -1
          if(len(r_real) == 1):
            y2s[i] = 2                # this case is TP -> 2
            pos_real[i] = r_real[0]
          elif(len(r_real) == 0):
            y2s[i] = 1                # this case is TN -> 1
            pos_real[i] = -1
          else:
            raise ValueError(f'Singleton or empty set expected, but got {r_real}')

          # pos_pred mirrors pos_real:
          # pos_pred[i] is the predicted position of the current label in the i-th
          # sample; if it stores -1 if the sample is not predicted as a TP case
          if(len(r_pred) == 1):
            pos_pred[i] = r_pred[0]
          elif(len(r_pred) == 0):
            pos_pred[i] = -1
          else:
            raise ValueError(f'Singleton or empty set expected, but got {r_real}')

        ys = y1s * y2s

        # summarises the description of the datapoints (position, colour) and cutoffs
        for i in range(m):
          if(pos_pred[i] == pos_real[i]):
            case = ECO_HIT
          else:
            case = ECO_MISS
          names[j][case].append(idxs[i])
          coords[j][case].append((xs[i], ys[i]))

        cutoffs.append(self.tw[j])

      else:
        raise ValueError(f'Support for {self.scenario} tasks not implemented')

    return (Y_pred, YorU_hat, names, coords, cutoffs, yticklabels)

  def show_scales(self, X, Y, idxs, tr_idxs, perf_params, show_params, scaleCols=1):

    # unpacks the parameters
    (scenario, popup,) = perf_params
    (gridCols, transpose, column2label, layout_data, hide_tags, filename) = show_params
    interactive = (filename is None)

    # prepares data
    (m,d) = X.shape
    (m,n) = Y.shape

    # maps each case to a datapoint in scale diagram (finds coordinates, chooses colour)
    (Y_pred, YorU_hat, names, coords, cutoffs, yticklabels) = self.inspect(X, Y, idxs, tr_idxs)
    (xmin, xmax) = (YorU_hat.min(), YorU_hat.max())
    xwidth = xmax - xmin
    thresholds = np.linspace(xmin, xmax, ECO_THRSHLDLVLS).tolist()
    dt = thresholds[1] - thresholds[0]

    # recovers data to organise the layout of the diagram
    layout = LayoutManager(layout_data)
    (nrows, ncols) = (ceil(n/scaleCols), scaleCols) # defines the diagram structure
    (sizew, sizeh, adjusts) = layout('scales.figure_sizes', nrows=nrows, ncols=ncols)

    # plots the scales diagram
    sc = defaultdict(lambda: defaultdict(list)) # sc[j][ECO_HIT or ECO_MISS], the same as datapoints
    tags = [] # one annotation for each subplot

    font_properties = layout('scales.title')
    fig, axes = plt.subplots(nrows, ncols, figsize=(sizew, sizeh))
    plt.suptitle('; '.join(self.get_config_summary(include_model_name=True)), fontsize=13) #**font_properties['fontdict'])
    for j in range(nrows*ncols):

      # sets up the layout of the current subplot
      plt.subplot(nrows, ncols, j + 1)
      plt.subplots_adjust(**adjusts)
      ax = plt.gca()

      if(j > n-1):
        # ensures an empty subplot is shown in case n is not a multiple of gridCols
        ax.axis('off')
        ax.autoscale()
        ax.set_box_aspect(1)

      else:
        plt.title(column2label[self.target_names[j]], **font_properties, loc='left')
        ax.set_xlim(xmin - dt, xmax + dt)
        ax.set_ylim(-3, 3)
        ax.margins(x=0.3, y=0.3)
        ax.use_sticky_edges = False
        ax.hlines(y=0, xmin=xmin, xmax=xmax, lw=5, color='k')
        ax.set_yticks(range(-2,3))
        ax.set_yticklabels(yticklabels)

        # plots the datapoints
        for case in coords[j]:
          # reorganises a list of (x,y) pairs in two lists xs,ys
          (xs,ys) = zip(*coords[j][case])
          if(case == ECO_HIT):
            sc[j][case] = ax.scatter(xs, ys, marker='o', c='darkgrey', zorder=1)
          elif(case == ECO_MISS):
            if(scenario in [ECO_DB_MULTICLASS, ECO_DB_MULTILABEL,]):
              sc[j][case] = ax.scatter(xs, ys, marker='o', c='red', zorder=2)
            else:
              point_colors = ['magenta' if (abs(ys[ii]) == 2 and xs[ii] > cutoffs[j]) else 'red' for ii in range(len(xs))]
              #sc[j][case] = ax.scatter(xs, ys, marker='o', c='magenta', zorder=2)
              sc[j][case] = ax.scatter(xs, ys, marker='o', c=point_colors, zorder=2)
          else:
            raise ValueError(f'Plotting cases {case} not implemented.')

        # plots the cutoff
        if(len(cutoffs) > 0):
          ax.axvline(x=cutoffs[j], lw=1, color='green')
          ax.axvline(x=cutoffs[j]+dt, lw=1, color='green', ls='--')
          ax.axvline(x=cutoffs[j]-dt, lw=1, color='green', ls='--')

        # sets up the resources to enable the interactive mode
        tag = ax.annotate('', xy=(0,0), xytext=(0,20), textcoords='offset points',
                              bbox=dict(boxstyle='round', fc='w'),
                              arrowprops=dict(arrowstyle='-'))
        tag.set_visible(False)
        tags.append(tag)

    def update_tag(j, case, idx_scatter):
      # recovers the (x,y) position of datapoint indicated by idx within the
      # PathCollection that is returned by plt.scatter
      pos = sc[j][case].get_offsets()[idx_scatter]

      # recovers the "name" of the datapoint within the data matrix X
      idx_source = names[j][case][idx_scatter]
      content = str(idx_source)

      # updates the required tag, makes it visible
      tag = tags[j]
      tag.xy = pos
      tag.set_text(content)
      tag.set_backgroundcolor('lemonchiffon')
      tag.get_bbox_patch().set_alpha(1.0)
      tag.get_bbox_patch().set_zorder(50)

      return idx_source

    def onclick_scales(event):
      # process event 'mouse hovered on datapoint'
      contains = False
      active_tags = []
      for j in range(n): # each j indicates the subplot for the j-th offer
        if(contains):
          break
        for case in sc[j]:
          tag = tags[j]
          vis = tag.get_visible()
          listener = sc[j][case] # listens all points drawn by plot.scatter
          (contains, ind) = listener.contains(event)
          if(contains):
            idx_scatter = ind['ind'][0]
            idx_source = update_tag(j, case, idx_scatter)
            tag.set_visible(True)
            for jj in range(n):
              if(jj != j):
                for case in names[jj]:
                  try:
                    sibling_idx_scatter = names[jj][case].index(idx_source)
                    update_tag(jj, case, sibling_idx_scatter)
                    sibling_tag = tags[jj]
                    sibling_tag.set_visible(True)
                  except:
                    pass
            fig.canvas.draw_idle()
            break
          else:
            if(vis):
              # all active tags are eventually hidden
              tag.set_visible(False)
              fig.canvas.draw_idle()

      return None

    # pops up the case informed by the user
    if(popup is not None):
      for j in range(n):
        try:
          case = ECO_HIT
          id_scatter = names[j][case].index(popup)
        except ValueError:
          case = ECO_MISS
          id_scatter = names[j][case].index(popup)
        idx_source = update_tag(j, case, id_scatter)
        tags[j].set_visible(True)

    # shows or saves the figure
    if(interactive):
      cid = fig.canvas.mpl_connect("button_press_event", onclick_scales)
      plt.show()
      fig.canvas.mpl_disconnect(cid)
    else:
      plt.savefig(filename, dpi=self.DPI)
      print(f'-- sent to file {filename}')

    (fw, fh) = (fig.get_figwidth(), fig.get_figheight())
    print(f'-- gridspec has {nrows} rows {ncols} colums')
    print(f'-- figure width is {fw} and height is {fh}')
    plt.close(fig)

    return None

class BaseCompetitorNg(BaseCompetitor):
  """
  Base class for wrapping models from the scikit-multilearn-ng to be used
  in the CLI environment. We use their implementation of Binary Relevance
  to transform a multilabel classification task into multiple binary
  classification tasks.
  """

  def fit(self, X, Y, U=None):

    if(self.scenario == ECO_DB_MULTICLASS):
      # learns a regression model to solve f(X) = Y approximately
      self.model.fit(X, Y)
      self.tw = None

    elif(self.scenario in [ECO_DB_MULTILABEL]):
      # learns a regression model to solve f(X) = Y approximately
      self.model.fit(X, Y)
      Y_hat = self.model.predict(X).toarray()
      self.tw = self.learn_thresholds(Y, Y_hat)

    elif(self.scenario in [ECO_DB_LABELRANK]):
      # learns a regression model to solve f(X) = U approximately
      self.model.fit(X, U)
      U_hat = self.model.predict(X).toarray()
      self.tw = self.learn_thresholds(Y, U_hat)

    return self

  def predict(self, X, raw=False, return_scores=False):
    YorU_hat = self.model.predict(X).toarray()
    Y_pred = self.post_predict(YorU_hat, raw=raw)
    if(return_scores):
      res = (Y_pred, YorU_hat)
    else:
      res = Y_pred
    return res

class RandomModel:

  def __init__(self, seed=None, sizes=[]):
    # see https://numpy.org/doc/stable/reference/random/generator.html
    self.n   = None
    self.rng = np.random.default_rng(seed=seed)

  def fit(self, X, Y):
    (m,n) =  Y.shape
    self.n = n
    return self

  def predict(self, X):
    # returns a random real matrix with values in [0,1)
    # -- as a consequence, the weighted area score is a random variable 
    #    with uniform distribution. This is the pattern one sees in the
    #    scale diagram for instances of this model
    (m,d) = X.shape
    Y_pred = self.rng.random((m,self.n))
    return Y_pred

class RandomCompetitor(BaseCompetitor):

  def __init__(self, maxsize, seed=None, sizes=[]):
    super().__init__(maxsize, seed=seed, sizes=sizes)
    self.model_name='Random'
    self.model = RandomModel(seed)
    self.size = 1

class LinearCompetitor(BaseCompetitor):
  """
  This class implements a standard OLS linear regressor without a bias term.
  This choice of architecture was motivated by the fact that its prediction step
  is similar to Polygrid's with the lstsq solver, in that its output comes from
  an inner product. The difference is that, while Polygrid moves the input
  vector to a higher dimensional space, and then performs an inner product in that
  space, this model does not change the original dimensionality of the input data.
  Thus, when comparing the situated performance of same-sized instances of both
  models (i.e., nspd=1, na=1), we are warranted to ascribe the observed difference
  in performance to the Polygrid' strategy of transforming a d-dimensional input
  vector X_i into a n_{as}-dimensional segment vector S_i.
  """
  def __init__(self, maxsize, seed=None, sizes=[]):
    super().__init__(maxsize, seed=seed, sizes=sizes)
    self.model_name='Linear'

    # obtains the size of this model, which is fixed by the number of features (d)
    # and the number of targets (n)
    (nw, n, d) = maxsize
    self.size = n * d

    # instantiates a model of the determined size
    # NOTE: LinearRegression is a wrapping of scipy.linalg.lstsq, which is
    #       identical to np.linalg.lstsq since 2017, after scipy adopted the
    #       LAPACK xGELSD driver as default. Polygrid uses np.linalg.lstsq.
    #       (see https://stackoverflow.com/questions/29372559s)
    self.model = LinearRegression(fit_intercept=False,
                                  copy_X=True,
                                  n_jobs=None,
                                  positive=False)

  def get_size(self):
    # the model does have a self.model.intercept_ attribute, but this is always 0
    # thus, it is not accounted in the size
    return self.model.coef_.size

class RidgeCompetitor(BaseCompetitor):
  """
  This class implements a regularised ridge regressor with a bias term.
  This choice of architecture was motivated by the fact that its prediction step
  is similar to Polygrid's with the ridge solver, in that its output comes from
  an inner product. The difference is that, while Polygrid moves the input
  vector to a higher dimensional space, and then performs an inner product in that
  space, this model only minimally changes the dimensionality of the input vector.
  Thus, when comparing the situated performance of same-sized instances of both
  models (i.e., nspd=1, na=1), we are warranted to ascribe the observed difference
  in performance to the Polygrid' strategy of transforming a d-dimensional input
  vector X_i into a n_{as}-dimensional segment vector S_i.
  """

  def __init__(self, maxsize, seed=None, sizes=[]):
    super().__init__(maxsize, seed=seed, sizes=sizes)
    self.model_name='Ridge'

    # obtains the size of this model, which is fixed by the number of features (d)
    # and the number of targets (n)
    (nw, n, d) = maxsize
    self.size = n * (d + 1)

    # instantiates a model of the determined size
    # NOTE: this is the same model and config used by Polygrid when solver='ridge'
    self.model = RidgeCV(fit_intercept=True, scoring='r2')

  def get_size(self):
    return self.model.coef_.size + self.model.intercept_.size

class MLPCompetitor(BaseCompetitor):
  """
  This class implements an MLP model with a single hidden layer followed by
  a linear output layer. Its 'predict' operation can be described by 2 steps:
  (a) first, the input vector is projected onto a higher dimensional space, and
  (b) then its inner product with an output vector is computed.
  This choice of architecture was motivated by the fact that these two steps also
  describes how Polygrid performs its 'predict' operation. The difference is how
  the first step is done:
  -- MLP: the inner product between the extended (d+1)-dimensional input vector
     and h distinct (d+1)-dimensional basis vectors are computed and sigmoid-ed;
  -- Polygrid: the d-dimensional input vector is mapped to a polygon, segmented
     according to some arbitrary partitioning of the unit disc, and then discretised.
  Thus, when comparing the situated performance of same-sized instances of both
  models, we are warranted to ascribe the difference in performance to differences
  between the strategies to project the input vector to a higher dimension.
  """
  def __init__(self, maxsize, seed=None, sizes=[]):
    super().__init__(maxsize, seed=seed, sizes=sizes)
    self.model_name='MLP'

    # creates the attributes that describe the size of the instance
    # in terms of its architecture
    self.h = None # number of neurons in the hidden layer
    self.o = None # number of neurons in the output layer

    """
    This recovers the data used to compute the desired size for this instance:
    -- nw holds the number of weights used up by the reference Polygrid instance
    -- n and d refer to the training data used by the reference:
       n is the number of labels, and d is the number of features
    """
    (nw, n, d) = maxsize

    """
    This determines the parameters of an MLPCompetitor instance that would have
    its average #weights converge around nw. However, since a precise control is
    not possible, we adopt an adaptive approach:

    -- First, we determine if the model is being fit for a standalone evaluation or
       for a robust assessment. A standalone eval happens when the user issues a
       'fit MLP' command from the CLI. In this case, there is no history of
       previous evals, which implies that 'sizes' is empty. On the other hand, a
       robust eval happens when the user issues an 'assess' command. In this case,
       CLI stores data from previous evals of this model, and 'sizes' contains the
       recent history of the #weights used up by fitted instances of this class.

    -- Second, the running weight budget is calculated. This means that, if nw=12,
       but a previous execution used up sizes=[11,] weights, then the current
       budget=12*(1+1)-11=13. Assuming n=3, d=4 (iris dataset), this implies that
       h=(13-3)(4+3+1)=5/4=1.25.

    -- Since h must be an integer larger than zero, we may have to round it up or
       down. In a standalone eval, this is decided by a coin flip. In a robust
       eval, the #weights of previously trained instances is used to estimate a
       value for h such that the average #weights is near nw in the short run.
       Resuming the example, since h=1.25 > 1 and mean(sizes)=11<12=nw, the
       adaptive policy would select the 'round up' action, selecting h=2 and
       #weights=h(d+1)+n(h+1)=2*5+3*3=19. Then, in the next iteration, with
       sizes=[11,19], we would have a reduced budget=12(1+2)-30=6, and since
       h=(6-3)/(4+3+1)=3/8 <= 1, that would imply h=1 and #weighs=h(d+1)+n(h+1)=
       1*5+3*2=11<nw. The number of replications performed in robust evals takes
       this dynamics into account.
    """
    if(nw in ['free', 'narrow']):
      """
      The user has asked us to allow the model to grow freely. For a single
      hidden layer MLP, there is just one dimension to grow: the number of hidden
      neurons h. In this case, we set h to the default for MLPRegressor. In
      scikit-learn 1.2.2, we have:
      -- hidden_layer_sizes: array-like of shape(n_layers - 2,), default=(100,)
         The ith element represents the number of neurons in the ith hidden layer.
      """
      # the user asked to allow the model to grow freely
      # for MLP, we set h to the default for MLPRegressor, scikit-learn 1.2.2
      h = 100

    else:
      # computes the number of hidden neurons that can be afforded by the budget
      standalone = (len(sizes) == 0)
      budget = nw if standalone else nw*(1 + len(sizes)) - sum(sizes)
      h = (budget - n)/(d + n + 1)
      if(h <= 1):
        """
        There must be at least one neuron in the hidden layer. As an example of a
        situation when h <= 1 happens, assume minimal values for n and d, namely
        n=2, d=3. In this case, the minimal Polygrid config would use up 6 weigts.
        In standalone mode, having nw=6, n=2, d=3 implies h=(6-2)/(3+2+1)=4/6=2/3.
        """
        h = 1
      else:
        """
        Since h must be an integer larger than 0, it is rounded it up or down based
        on the running budget. When running avits, the policy selects a round up
        action, and when running deficits, the opposite action is taken.
        """
        do_ceil = (self.rng.random() >= .5) if standalone else (np.mean(sizes) < nw)
        h = ceil(h) if do_ceil else floor(h)

    # instantiates a model of the determined size
    self.h = h
    self.o = n
    self.model = MLPRegressor(
                      hidden_layer_sizes = (h,),
                      activation = 'logistic',  # allows for analytic treatment
                      solver    = 'lbfgs',       # better for small datasets
                      max_iter  = 30000,         # enough to avoid 'no convergence'
                      max_fun   = 30000,         # lbfgs only; needs tol
                      tol       = 1E-4,
                      alpha     = 0.0001,
                      verbose   = False,
                      warm_start = False,
                      # not used in lbfgs: batch_size
                      # only sgd/adam: early_stopping, validation_fraction,
                      # n_iter_no_change, # learning_rate, learning_rate_init,
                      # beta_1, beta_2, epsilon, power_t, shuffle, momentum,
                      # nesterovs_momentum
                      random_state = seed)

    # model size cannot be updated here because weight matrices are not available
    # at this point
    pass

  def get_size(self):
    if(self.size is None):
      self.size = get_size_MLPRegressor(self.model)
    return self.size

  def get_config_summary(self, include_model_name=False, details=[]):
    buffer = f'neurons: {self.h + self.o}, {self.h} hidden, {self.o} output'
    details = [buffer]
    summary = super().get_config_summary(include_model_name=include_model_name, details=details)
    return summary

class mlpCompetitor(MLPCompetitor):
  """
  Experimental -- the same as MLP, except for the activation function,
                  which is now linear
  """
  def __init__(self, maxsize, seed=None, sizes=[]):
    super().__init__(maxsize, seed=seed, sizes=sizes)
    self.model_name='mlp'

    # creates the attributes that describe the size of the instance
    # in terms of its architecture
    self.h = None # number of neurons in the hidden layer
    self.o = None # number of neurons in the output layer

    """
    This recovers the data used to compute the desired size for this instance:
    -- nw holds the number of weights used up by the reference Polygrid instance
    -- n and d refer to the training data used by the reference:
       n is the number of labels, and d is the number of features
    """
    (nw, n, d) = maxsize

    """
    This determines the parameters of an MLPCompetitor instance that would have
    its average #weights converge around nw. However, since a precise control is
    not possible, we adopt an adaptive approach:

    -- First, we determine if the model is being fit for a standalone evaluation or
       for a robust assessment. A standalone eval happens when the user issues a
       'fit MLP' command from the CLI. In this case, there is no history of
       previous evals, which implies that 'sizes' is empty. On the other hand, a
       robust eval happens when the user issues an 'assess' command. In this case,
       CLI stores data from previous evals of this model, and 'sizes' contains the
       recent history of the #weights used up by fitted instances of this class.

    -- Second, the running weight budget is calculated. This means that, if nw=12,
       but a previous execution used up sizes=[11,] weights, then the current
       budget=12*(1+1)-11=13. Assuming n=3, d=4 (iris dataset), this implies that
       h=(13-3)(4+3+1)=5/4=1.25.

    -- Since h must be an integer larger than zero, we may have to round it up or
       down. In a standalone eval, this is decided by a coin flip. In a robust
       eval, the #weights of previously trained instances is used to estimate a
       value for h such that the average #weights is near nw in the short run.
       Resuming the example, since h=1.25 > 1 and mean(sizes)=11<12=nw, the
       adaptive policy would select the 'round up' action, selecting h=2 and
       #weights=h(d+1)+n(h+1)=2*5+3*3=19. Then, in the next iteration, with
       sizes=[11,19], we would have a reduced budget=12(1+2)-30=6, and since
       h=(6-3)/(4+3+1)=3/8 <= 1, that would imply h=1 and #weighs=h(d+1)+n(h+1)=
       1*5+3*2=11<nw. The number of replications performed in robust evals takes
       this dynamics into account.
    """
    if(nw in ['free', 'narrow']):
      """
      The user has asked us to allow the model to grow freely. For a single
      hidden layer MLP, there is just one dimension to grow: the number of hidden
      neurons h. In this case, we set h to the default for MLPRegressor. In
      scikit-learn 1.2.2, we have:
      -- hidden_layer_sizes: array-like of shape(n_layers - 2,), default=(100,)
         The ith element represents the number of neurons in the ith hidden layer.
      """
      # the user asked to allow the model to grow freely
      # for MLP, we set h to the default for MLPRegressor, scikit-learn 1.2.2
      h = 100

    else:
      # computes the number of hidden neurons that can be afforded by the budget
      standalone = (len(sizes) == 0)
      budget = nw if standalone else nw*(1 + len(sizes)) - sum(sizes)
      h = (budget - n)/(d + n + 1)
      if(h <= 1):
        """
        There must be at least one neuron in the hidden layer. As an example of a
        situation when h <= 1 happens, assume minimal values for n and d, namely
        n=2, d=3. In this case, the minimal Polygrid config would use up 6 weigts.
        In standalone mode, having nw=6, n=2, d=3 implies h=(6-2)/(3+2+1)=4/6=2/3.
        """
        h = 1
      else:
        """
        Since h must be an integer larger than 0, it is rounded it up or down based
        on the running budget. When running avits, the policy selects a round up
        action, and when running deficits, the opposite action is taken.
        """
        do_ceil = (self.rng.random() >= .5) if standalone else (np.mean(sizes) < nw)
        h = ceil(h) if do_ceil else floor(h)

    # instantiates a model of the determined size
    self.h = h
    self.o = n
    self.model = MLPRegressor(
                      hidden_layer_sizes = (h,),
                      activation = 'tanh',   # allows for analytic treatment
                      solver     = 'lbfgs',      # better for small datasets
                      max_iter   = 30000,        # enough to avoid 'no convergence'
                      max_fun    = 30000,        # lbfgs only; needs tol
                      tol        = 1E-4,
                      alpha      = 0.0001,
                      verbose    = False,
                      warm_start = False,
                      # not used in lbfgs: batch_size
                      # only sgd/adam: early_stopping, validation_fraction,
                      # n_iter_no_change, # learning_rate, learning_rate_init,
                      # beta_1, beta_2, epsilon, power_t, shuffle, momentum,
                      # nesterovs_momentum
                      random_state = seed)

    # model size cannot be updated here because weight matrices are not available
    # at this point
    pass

class DTCompetitor(BaseCompetitor):
  """
  This class wraps the DecisionTreeRegressor model from the scikit-learn library,
  which implements a variant of the CART decision tree regressor [2]. The choice of
  architecture was motivated by the fact that multiple, massive evaluations of
  models that perform multilabel classification and label ranking tasks in the
  literature point out that the best performing models are usually based on DTs:

  [1] Bogatinovski, Jasmin, Ljupčo Todorovski, Sašo Džeroski, and Dragi Kocev.
      "Comprehensive comparative study of multi-label classification methods."
      Expert Systems with Applications 203 (2022): 117215.
      -- The authors report results from an evaluation of 26 multilabel
         classification methods vs. 42 benchmark datasets on 20 measures.
      -- They found that RFDTBR was among the best performing methods when
         example-based, label-based, or ranking-based measures were considered.
         In their nomenclature, RFDTBR stands for 'Binary Relevance with Random
         Forest of Decision Trees', which corresponds to our BRRFCompetitor model.

  [2] Fotakis, Dimitris, Alkis Kalavasis, and Eleni Psaroudaki.
      "Label ranking through nonparametric regression."
      In International Conference on Machine Learning, pp. 6622-6659. PMLR, 2022.
      -- The authors report results from an evaluation of 5 label ranking methods
         vs. 21 datasets on the Kendall's tau coefficient.
      -- They found that performance of RF and DT were mostly insignificant.
         In their work, RF stands for 'Random Forest of Decision Trees'. They
         implemented both RF and DT using scikit-learn classes, so this wrapper
         benefits from the code they shared (e.g., selected parameters).
         (see LabelRankingAlgorithms_SS_R.py)

  Thus, when comparing the situated performance of same-sized instances of both
  models, we are warranted to ascribe the difference in performance to differences
  xxx_____________________________ [2]. Besides, when the performance of the
  Polygrid model is lower (for same-sized models), we can argue that the
  difference is the price we pay for xxx________________________.

  --- read Psaroudaki, I think she describes CART trees in terms that involve inner
      products;

  """
  def __init__(self, maxsize, seed=None, sizes=[]):
    super().__init__(maxsize, seed=seed, sizes=sizes)
    self.model_name='DT'

    """
    This recovers the data used to compute the desired size for this instance:
    -- nw holds the number of weights used up by the reference Polygrid instance
    -- n and d refer to the training data used by the reference:
       n is the number of labels, and d is the number of features
    """
    (nw, n, d) = maxsize

    # creates the attributes that describe the size of the instance
    # in terms of its architecture
    self.n_trees  = 1    # number of trees
    self.n_nodes  = None # number of nodes
    self.n_splits = None # number of split nodes
    self.n_leaves = None # number of leaf  nodes
    self.target_k = None # (target) depth of the tree

    """
    This determines the parameters of a DTCompetitor instance that would have its
    average self.get_size() approximately near to nw. The idea is to explore a
    parameter that constrains the depth of an RDT (Regression Decision Tree)
    instance. However, this is an indirect, suboptimal control strategy, since we
    aim to control the number of weights of an RDT, but depth barely give us
    control over the number of nodes in the tree, e.g.:
    -- two RDTs of same depth k may have markedly distinct number of nodes,
       depending on how differently balanced they are;
    -- two RDTs with N nodes may not have the same number of weights, owing to
       differences in the number of splitting and leaf nodes.

    Thus, since a precise control is not possible, we adopt an adaptive approach
    that is based on 4 assumptions:
    (P1) The maximum number of weights an RDT of depth k can hold is attained by a
         balanced RDT;
    (P2) The minimum number of weights an RDT of depth k can hold is attained by a
         one-sided RDT (e.g., all left children are leaves);
    (P3) Each split node of an RDT corresponds to 2 weights.
         Evidence: the implementation we are using stores split nodes in 2 arrays:
            DecisionTreeRegressor.tree_.feature
            DecisionTreeRegressor.tree_.threshold
    (P4) Each leaf node of an RDT corresponds to 1 weight.
         Evidence: the implementation we are using stores leaf nodes in one array:
            DecisionTreeRegressor.tree_.threshold

    General strategy:
    -- First, we calculate the running weight budget. See details in MLPCompetitor.
    -- Second, if the budget is large enough, we compute the minimum and maximum
       estimates for the depth parameter. When running deficits, we set k=k_min,
       which produces shallower trees. When running avits, k moves towards k_max,
       fulfilling with the opposite goal.
    """
    alpha = ECO_SPLITNODE2WEIGHT # positive integer larger than 0
    beta  = ECO_LEAFNODE2WEIGHT  # positive integer larger than 0
    w_lb  = 1 * ECO_SPLITNODE2WEIGHT + 2 * ECO_LEAFNODE2WEIGHT # weight of RDT k=1

    if(nw in ['free', 'narrow']):
      """
      The user has asked us to allow the model to grow freely. For a CART DT, there
      is just one dimension we control: its maximum depth k. In this case, we set k
      to the default for DecisionTreeRegressor. In scikit-learn 1.2.2, we have:
      -- max_depth : int, default None
         The maximum depth of the tree. If None, then nodes are expanded until all
         leaves are pure or until all leaves contain less than min_samples_split
         samples (min_samples_split : int or float, default=2).
      """
      k = None

    else:
      # then nw >= 6, which is the size of the minimal Polygrid model (in #weights)
      """
      The reference model has used up nw weights, which is (more than) the number
      of weights needed to build a minimal RDT. Since DTCompetitor has a single DT,
      the next decision is to determine how to constrain its depth so that it uses
      up about the same number of weights as the reference model.

      In the code below, k_min represents the depth of a balanced RDT whose weight
      is equal to the current budget, and k_max is the depth of a one-sided RDT
      of weight equal to the current budget. An explanation about how the formulas
      for k_min and k_max were derived can be found in the notebook:
         'Strategy to control weights of a regression decision tree'
         https://colab.research.google.com/drive/1iI0f1XyvKO1Duw5yb711vc3K33hhj37O

      Finally, gamma determines a value k between the minimal and maximal estimated
      depths, following a budget-aware policy:

      -- running deficits:
           mean(sizes) > nw means using up more weights than allowed, which implies
             budget < nw, which implies
                gamma = 0.0, which implies
                   k = ceil(k_min), which aims to produce shallower trees
           budget < 0 < w_lb may happen; in this case, we set k=1

      -- running avits:
           mean(sizes) < nw means using up less weights than asked, which implies
             budget > nw, which implies
                gamma > 0, which implies
                   k > ceil(k_min)
                   as gamma approaches 1, k approaches ceil(k_max), which aims to
                   produce deeper trees
           k > ceil(k_max) may happen, in which case we set k=None to allow the tre
           to grow freely.
      """
      standalone = (len(sizes) == 0)
      budget = nw if standalone else nw*(1 + len(sizes)) - sum(sizes)
      if(budget >= w_lb):
        k_min = (log(alpha + budget) - log(alpha + beta)) / log(2)
        k_max = (budget - beta)/(alpha + beta)
        gamma = max(0.0, 2*(budget - nw)/budget) # assuredly budget >= w_lb > 0
        k     = ceil(k_min + gamma*(k_max - k_min))

      else:
        # budget < 0 < w_lb may happen, which indicates a systematic running deficit.
        # In this case, we strongly constrain the depth of this instance
        k = 1

    self.target_k = k
    self.model = DecisionTreeRegressor(max_depth=self.target_k,
                                       max_leaf_nodes=None,
                                       criterion='squared_error', # as in [2]
                                       splitter='best',           # as in [2]
                                       random_state=seed,
                                       )

    # model size cannot be updated now because tree structures are not available
    # at this point
    pass

  def get_size(self):
    # each split node is taken to correspond to ECO_SPLITNODE2WEIGHT weights
    # each leaf  node is taken to correspond to ECO_LEAFNODE2WEIGHT  weights
    (n_nodes, n_splits, n_leaves) = count_node_types(self.model)
    self.n_nodes  = n_nodes
    self.n_splits = n_splits
    self.n_leaves = n_leaves
    return n_splits * ECO_SPLITNODE2WEIGHT + n_leaves * ECO_LEAFNODE2WEIGHT

  def get_config_summary(self, include_model_name=False, details=[]):
    self.get_size()
    buffer = f'trees: {self.n_trees}, max depth {self.target_k}'
    details = [buffer]
    buffer = f'nodes: {self.n_nodes}, {self.n_splits} splits, {self.n_leaves} leaves'
    details.append(buffer)
    summary = super().get_config_summary(include_model_name=include_model_name, details=details)
    return summary

class RFCompetitor(BaseCompetitor):
  """
  This class wraps the RandomForestRegressor model from the scikit-learn library,
  which implements a variant of the Breiman's random forest with decision tree
  regressors. The choice of architecture was motivated by the fact that multiple,
  massive evaluations of models that perform multilabel classification and label
  ranking tasks in the literature point out that the best performing models are
  usually based on DTs, as mentioned earlier in the description of DTCompetitor.

  Thus, when comparing the situated performance of same-sized instances of both
  models, we are warranted to ascribe the difference in performance to differences
  xxx_____________________________ [2]. Besides, when the performance of the
  Polygrid model is lower (for same-sized models), we can argue that the
  difference is the price we pay for xxx________________________.

  xxx check Psaroudaki, I think she describes CART trees in terms that involve inner
      products;

  """
  def __init__(self, maxsize, seed=None, sizes=[]):
    super().__init__(maxsize, seed=seed, sizes=sizes)
    self.model_name='RF'

    """
    This recovers the data used to compute the desired size for this instance:
    -- nw holds the number of weights used up by the reference Polygrid instance
    -- n and d refer to the training data used by the reference:
       n is the number of labels, and d is the number of features
    """
    (nw, n, d) = maxsize

    # creates the attributes that describe the size of the instance
    # in terms of its architecture
    self.n_trees  = None # number of trees
    self.n_nodes  = None # number of nodes
    self.n_splits = None # number of split nodes
    self.n_leaves = None # number of leaf  nodes
    self.target_k = None # (target) depth of the tree

    """
    This determines the parameters of an RFCompetitor instance that would have its
    average self.get_size() approximately near to nw. The idea is to explore a
    parameter that constrains the depth of its underlying RDT instances, much the
    same way as DTCompetitor does. Again, this is an indirect, suboptimal control
    strategy, since we aim to control the number of weights of the whole instance.
    Thus, we resort to an adaptive control strategy. Please refer to DTCompetitor
    for details about this control strategy, since this is mostly similar to that.

    """
    alpha = ECO_SPLITNODE2WEIGHT # positive integer larger than 0
    beta  = ECO_LEAFNODE2WEIGHT  # positive integer larger than 0
    w_lb  = 1 * ECO_SPLITNODE2WEIGHT + 2 * ECO_LEAFNODE2WEIGHT # weight of RDT k=1

    if(nw == 'free'):
      """
      The user has asked us to allow the model to grow freely. For a RF, there are
      two dimensions it can grow: the number (dt_rf) and depth (k) of its trees.
      In this case, we set dt_rf to the reference value in [1], and allow the depth
      to grow freely.
      """
      k = None    # DTs with unconstrained depth
      dt_rf = 100 # set dt_rf = 100 to replicate [1] (Study details complement), or
                  # set dt_rf =  50 to replicate [2]

    elif(nw == 'narrow'):
      """
      The user has asked us to allow the model to grow in one dimension. For a RF,
      there are two dimensions, as we mentioned above. In this case, we set dt_rf
      to the number of labels of the problem, and allow the depth to grow freely.
      """
      k = None   # DTs with unconstrained depth
      dt_rf = n  # sets the number of DTs in the forest as the number of labels

    else:
      """
      The reference model has used up nw weights. If the reference was the Polygrid
      model, then assuredly nw >= 6 because this is the size of the model in the
      minimal setting (d = 3, n = 2, nspd = 1, na = 1). Then, nw weights is (more
      than) the number of weights needed to build a minimal RDT. Since RFCompetitor
      can have multiple DTs, the goal is to figure out how many DTs to build, and
      and then establish their maximum depth. For details about how k_min, kmax,
      and gamma are defined and used, please see DTCompetitor.
      """
      standalone = (len(sizes) == 0)
      budget = nw if standalone else nw*(1 + len(sizes)) - sum(sizes)

      if(budget >= w_lb):
        """
        For RF, the whole budget is allocated to a single random forest. The
        maximum number of decision trees per random forest is set to the number
        of existing labels. This decision makes it possible to compare RF and
        BRDT structurally: both are allowed to grow the same number of trees,
        and each tree is allowed to grow freely. Thus, any observed differences
        in performance can be predominantly ascribed to the stochastic policies
        introduced by random forests, namely sample bootstrapping and random
        feature selection.
        """
        nw_rf = budget             # the whole budget is allocated to a single RF
        dt_rf = n                  # the desired (maximum) number of DTs per RF

        # computes the share of the budget given to each DT
        nw_dt = nw_rf // dt_rf
        if(nw_dt < w_lb):
          # the budget per DT is not enough to build #dt_rf DTs
          # so, how many minimal DTs can be build with the available budget?
          nw_dt = w_lb
          dt_rf = nw_rf // nw_dt   # assuredly dt_rf >= 1 because
                                   # -- nw_rf = budget >= w_lb, and nw_dt = w_lb
          nw_dt = nw_rf // dt_rf   # revised budget for each DT

        # now that the budget per DT is known, estimates the corresponding depth
        k_min = (log(alpha + nw_dt) - log(alpha + beta)) / log(2)
        k_max = (nw_dt - beta)/(alpha + beta)
        gamma = max(0.0, 2*(budget - nw)/budget) # assuredly nw_dt >= w_lb > 0
        k     = ceil(k_min + gamma*(k_max - k_min))

      else:
        # budget < 0 < w_lb may happen, which indicates a systematic running deficit.
        # In this case, we strongly constrain the depth of this instance
        dt_rf = 1
        k = 1

    self.target_k = k
    self.n_trees  = dt_rf
    self.model = RandomForestRegressor(max_depth=self.target_k,
                                       max_leaf_nodes=None,
                                       n_estimators=self.n_trees,
                                       # DTs will use best split strategy
                                       criterion='squared_error', # as in [2]
                                       bootstrap=True,
                                       max_samples=0.66, #xxx Alpaydin...
                                       max_features='sqrt',  # explored in [1,2]
                                       random_state=seed,
                                       )

    # model size cannot be updated now because tree structures are not available
    # at this point
    pass

  def get_size(self):
    # each split node is taken to correspond to ECO_SPLITNODE2WEIGHT weights
    # each leaf  node is taken to correspond to ECO_LEAFNODE2WEIGHT  weights
    (n_nodes, n_splits, n_leaves) = count_node_types(self.model)
    self.n_nodes  = n_nodes
    self.n_splits = n_splits
    self.n_leaves = n_leaves
    return n_splits * ECO_SPLITNODE2WEIGHT + n_leaves * ECO_LEAFNODE2WEIGHT

  def get_config_summary(self, include_model_name=False, details=[]):
    self.get_size()
    buffer = f'trees: {self.n_trees}, max depth {self.target_k}'
    details = [buffer]
    buffer = f'nodes: {self.n_nodes}, {self.n_splits} splits, {self.n_leaves} leaves'
    details.append(buffer)
    summary = super().get_config_summary(include_model_name=include_model_name, details=details)
    return summary

class BRDTCompetitor(BaseCompetitorNg):
  """
  This class wraps the BinaryRelevance model from the scikit-multilearn-ng library
  equipped with the DecisionTreeRegressor model from the scikit-learn library as
  the base classifier. The BinaryRelevance model implements the strategy of
  transforming a multilabel classification task into n binary classification tasks,
  with n being the number of labels of the problem [1]. The choice of
  architecture was motivated by the fact that multiple, massive evaluations of
  models that perform multilabel classification tasks in the literature point out
  that the best performing models are usually based on a combination of BR and DTs:

  [1] Bogatinovski, Jasmin, Ljupčo Todorovski, Sašo Džeroski, and Dragi Kocev.
      "Comprehensive comparative study of multi-label classification methods."
      Expert Systems with Applications 203 (2022): 117215.
      -- The authors report results from an evaluation of 26 multilabel
         classification methods vs. 42 benchmark datasets on 20 measures.
      -- They found that RFDTBR was among the best performing methods when
         example-based, label-based, or ranking-based measures were considered.
         In their nomenclature, RFDTBR stands for 'Binary Relevance with Random
         Forest of Decision Trees', which corresponds to our BRRFCompetitor model.

  Thus, when comparing the situated performance of same-sized instances of both
  models, we are warranted to ascribe the difference in performance to differences
  xxx_____________________________ [1]. Besides, when the performance of the
  Polygrid model is lower (for same-sized models), we can argue that the
  difference is the price we pay for xxx________________________.
  """
  def __init__(self, maxsize, seed=None, sizes=[]):
    super().__init__(maxsize, seed=seed, sizes=sizes)
    self.model_name='BRDT'

    """
    This recovers the data used to compute the desired size for this instance:
    -- nw holds the number of weights used up by the reference Polygrid instance
    -- n and d refer to the training data used by the reference:
       n is the number of labels, and d is the number of features
    """
    (nw, n, d) = maxsize

    # creates the attributes that describe the size of the instance
    # in terms of its architecture
    self.n_trees  = n    # number of trees
    self.n_nodes  = None # number of nodes
    self.n_splits = None # number of split nodes
    self.n_leaves = None # number of leaf  nodes
    self.target_k = None # (target) depth of the tree

    """
    This determines the parameters of a BRDTCompetitor instance that would have its
    average self.get_size() approximately near to nw. The idea is to explore a
    parameter that constrains the depth of an RDT (Regression Decision Tree)
    instance. However, this is an indirect, suboptimal control strategy, since we
    aim to control the number of weights of an RDT, but depth barely give us
    control over the number of nodes in the tree. Thus, since a precise control is
    not possible, we adopt an adaptive approach. See DTCompetitor for more details.
    """
    alpha = ECO_SPLITNODE2WEIGHT
    beta  = ECO_LEAFNODE2WEIGHT
    w_lb  = 1 * ECO_SPLITNODE2WEIGHT + 2 * ECO_LEAFNODE2WEIGHT

    if(nw in ['free', 'narrow']):
      """
      The user has asked us to allow the model to grow freely. For BRDT, there are
      two dimensions it can grow: the number of and the depth of the trees.
      Since the BR strategy sets the number of trees as the number of labels, we
      are left to control the depth of the DTs, which we allow to grow freely.
      """
      k = None

    else:
      """
      The reference model has used up nw weights. If the reference was the Polygrid
      model, then assuredly nw >= 6 because this is the size of the model in the
      minimal setting (d = 3, n = 2, nspd = 1, na = 1). Since BRDTCompetitor has
      exactly n DTs, the goal is to figure out their maximum depth. For details
      about how k_min, kmax, and gamma are defined and used, see DTCompetitor.
      """
      standalone = (len(sizes) == 0)
      budget = nw if standalone else nw*(1 + len(sizes)) - sum(sizes)

      if(budget >= n * w_lb):
        """
        The current budget is large enough to build a minimal RDT for each label.
        Thus, the budget is equally divided among #n DTs.
        """
        nw_dt = budget // n # #weights per connected component of the forest (DTs)
        k_min = (log(alpha + nw_dt) - log(alpha + beta)) / log(2)
        k_max = (nw_dt - beta)/(alpha + beta)
        gamma = max(0.0, 2*(budget - nw)/budget)
        k     = ceil(k_min + gamma*(k_max - k_min))

      else:
        # budget < 0 < n*w_lb may happen, which indicates a systematic running
        # deficits. In this case, we strongly constrain the depth of this instance
        k = 1

    self.target_k = k
    self.model = BinaryRelevance(
            classifier = DecisionTreeRegressor(max_depth=self.target_k,
                                               max_leaf_nodes=None,
                                               criterion='squared_error', # as in [2]
                                               splitter='best',           # as in [2]
                                               random_state=seed,
                                               ),
            require_dense = [True, True],
        )

    # the size of the model cannot be updated here because tree structures are not
    # available at this point
    pass

  def get_size(self):
    # each split node is taken to correspond to ECO_SPLITNODE2WEIGHT weights
    # each leaf  node is taken to correspond to ECO_LEAFNODE2WEIGHT  weights
    (n_nodes, n_splits, n_leaves) = count_node_types(self.model)
    self.n_nodes  = n_nodes
    self.n_splits = n_splits
    self.n_leaves = n_leaves
    return n_splits * ECO_SPLITNODE2WEIGHT + n_leaves * ECO_LEAFNODE2WEIGHT

  def get_config_summary(self, include_model_name=False, details=[]):
    self.get_size()
    buffer = f'trees: {self.n_trees}, max depth {self.target_k}'
    details = [buffer]
    buffer = f'nodes: {self.n_nodes}, {self.n_splits} splits, {self.n_leaves} leaves'
    details.append(buffer)
    summary = super().get_config_summary(include_model_name=include_model_name, details=details)
    return summary

class BRRFCompetitor(BaseCompetitorNg):
  """
  This class wraps the BinaryRelevance model from the scikit-multilearn-ng library
  equipped with the RandomForestRegressor model from the scikit-learn library as
  the base classifier. The BinaryRelevance model implements the strategy of
  transforming a multilabel classification task into n binary classification tasks,
  with n being the number of labels of the problem [1]. The choice of
  architecture was motivated by the fact that multiple, massive evaluations of
  models that perform multilabel classification tasks in the literature point out
  that the best performing models are usually based on a combination of BR and DTs:

  [1] Bogatinovski, Jasmin, Ljupčo Todorovski, Sašo Džeroski, and Dragi Kocev.
      "Comprehensive comparative study of multi-label classification methods."
      Expert Systems with Applications 203 (2022): 117215.
      -- The authors report results from an evaluation of 26 multilabel
         classification methods vs. 42 benchmark datasets on 20 measures.
      -- They found that RFDTBR was among the best performing methods when
         example-based, label-based, or ranking-based measures were considered.
         In their nomenclature, RFDTBR stands for 'Binary Relevance with Random
         Forest of Decision Trees', which corresponds to our BRRFCompetitor model.

  Thus, when comparing the situated performance of same-sized instances of both
  models, we are warranted to ascribe the difference in performance to differences
  xxx_____________________________ [1]. Besides, when the performance of the
  Polygrid model is lower (for same-sized models), we can argue that the
  difference is the price we pay for xxx________________________.
  """

  def __init__(self, maxsize, seed=None, sizes=[]):
    super().__init__(maxsize, seed=seed, sizes=sizes)
    self.model_name='BRRF'

    """
    This recovers the data used to compute the desired size for this instance:
    -- nw holds the number of weights used up by the reference Polygrid instance
    -- n and d refer to the training data used by the reference:
       n is the number of labels, and d is the number of features
    """
    (nw, n, d) = maxsize

    # creates the attributes that describe the size of the instance
    # in terms of its architecture
    self.n_trees  = None # number of trees
    self.n_nodes  = None # number of nodes
    self.n_splits = None # number of split nodes
    self.n_leaves = None # number of leaf  nodes
    self.target_k = None # (target) depth of the tree

    """
    This determines the parameters of a BRRFCompetitor instance that would have its
    average self.get_size() approximately near to nw. The idea is to explore a
    parameter that constrains the depth of an RDT (Regression Decision Tree)
    instance. However, this is an indirect, suboptimal control strategy, since we
    aim to control the number of weights of an RDT, but depth barely give us
    control over the number of nodes in the tree. Thus, since a precise control is
    not possible, we adopt an adaptive approach. See DTCompetitor for more details.
    """
    alpha = ECO_SPLITNODE2WEIGHT
    beta  = ECO_LEAFNODE2WEIGHT
    w_lb  = 1 * ECO_SPLITNODE2WEIGHT + 2 * ECO_LEAFNODE2WEIGHT

    if(nw == 'free'):
      """
      The user has asked us to allow the model to grow freely. For a BRRF, there
      are three dimensions it can grow: the number of forests, the number of trees
      per forest, and the depth of these trees. Since the BR strategy sets the
      number of forests as the number of labels, we are left to control the number
      and the depth of the DTs. In this case, we set dt_rf to the reference value
      in [1] or [2], and allow the depth of the DTs to grow freely.
      """
      k = None    # DTs with unconstrained depth
      dt_rf = 100 # set dt_rf = 100 to replicate [1], or
                  # set dt_rf =  50 to replicate [2]

    elif(nw == 'narrow'):
      """
      The user has asked us to allow the model to grow in one dimension. For BRRF,
      there are three dimensions, as we mentioned above. One is constrained by the
      BR strategy, and the other two are free. In this case, we set dt_rf to the
      number of labels of the problem, and allow the depth to grow freely.
      """
      k = None   # DTs with unconstrained depth
      dt_rf = n  # sets the number of DTs in the forest as the number of labels

    else:
      """
      The reference model has used up nw weights. If the reference was the Polygrid
      model, then assuredly nw >= 6 because this is the size of the model in the
      minimal setting (d = 3, n = 2, nspd = 1, na = 1). Then, nw weights is (more
      than) the number of weights needed to build a minimal RDT. Since
      BRRFCompetitor can hold multiple DTs, the goal is to figure out how many DTs
      to build, and establish their maximum depth. For details about how k_min,
      kmax, and gamma are defined and used, please see DTCompetitor.
      """
      standalone = (len(sizes) == 0)
      budget = nw if standalone else nw*(1 + len(sizes)) - sum(sizes)

      if(budget >= n * w_lb):
        """
        The current budget is large enough to secure a minimal RDT for each label.
        In this arrangement, each forest would hold a single, minimal RDT.
        Thus, the budget is equally divided among #n RFs.
        """
        nw_rf = budget // n      # budget equally divided among n random forests
        dt_rf = ECO_MAX_DT_RF    # the desired (maximum) number of DTs per RF

        # computes the share of the budget given to each DT
        nw_dt = nw_rf // dt_rf
        if(nw_dt < w_lb):
          # the budget per DT is not enough to build n RFs with #rf_dt DTs
          # so, how many minimal DTs can be build with the available budget?
          nw_dt = w_lb
          dt_rf = nw_rf // nw_dt # surely nw_dt = w_lb > 0
          nw_dt = nw_rf // dt_rf # revised budget for each DT

        # now that the budget per DT is known, estimates the corresponding depth
        k_min = (log(alpha + nw_dt) - log(alpha + beta)) / log(2)
        k_max = (nw_dt - beta)/(alpha + beta)
        gamma = max(0.0, 2*(budget - nw)/budget)
        k     = ceil(k_min + gamma*(k_max - k_min))

      else:
        # budget < 0 < n*w_lb may happen, which indicates a systematic running
        # deficits. In this case, we strongly constrain the depth of this instance
        dt_rf = 1
        k = 1

    self.target_k = k
    self.n_trees  = n * dt_rf
    self.model = BinaryRelevance(
            classifier = RandomForestRegressor(max_depth=self.target_k,
                                               max_leaf_nodes=None,
                                               n_estimators=dt_rf,
                                               # DTs will use best split strategy
                                               criterion='squared_error', # as in [2]
                                               bootstrap=True,
                                               max_samples=0.66, #xxx Alpaydin...
                                               max_features='sqrt',  # explored in [2]
                                               random_state=seed,
                                               ),
            require_dense = [True, True],
        )

    # the size of the model cannot be updated here because tree structures are not
    # available at this point
    pass

  def get_size(self):
    # each split node is taken to correspond to ECO_SPLITNODE2WEIGHT weights
    # each leaf  node is taken to correspond to ECO_LEAFNODE2WEIGHT  weights
    (n_nodes, n_splits, n_leaves) = count_node_types(self.model)
    self.n_nodes  = n_nodes
    self.n_splits = n_splits
    self.n_leaves = n_leaves
    return n_splits * ECO_SPLITNODE2WEIGHT + n_leaves * ECO_LEAFNODE2WEIGHT

  def get_config_summary(self, include_model_name=False, details=[]):
    self.get_size()
    buffer = f'trees: {self.n_trees}, max depth {self.target_k}'
    details = [buffer]
    buffer = f'nodes: {self.n_nodes}, {self.n_splits} splits, {self.n_leaves} leaves'
    details.append(buffer)
    summary = super().get_config_summary(include_model_name=include_model_name, details=details)
    return summary

def get_size_MLPRegressor(model):
  return (sum([e.size for e in model.coefs_]) +
          sum([e.size for e in model.intercepts_]))

def count_node_types(model):

  def _count_dt_node_types(decision_tree):
    children_left  = decision_tree.tree_.children_left
    children_right = decision_tree.tree_.children_right
    n_nodes = decision_tree.tree_.node_count
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # starts with the root node id (0) and its depth (0)
    while len(stack) > 0:
        (node_id, depth) = stack.pop()
        if (children_left[node_id] != children_right[node_id]):
            # this is a split node; we will visit their children later
            stack.append((children_left[node_id],  depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    n_leaves = sum(is_leaves)
    n_splits = n_nodes - n_leaves
    return np.array([n_nodes, n_splits, n_leaves])

  if(type(model) in [DecisionTreeRegressor]):
    node_counts = _count_dt_node_types(model)
  elif(type(model) in [RandomForestRegressor]):
    node_counts = np.zeros(3, dtype=int)
    for dt in model.estimators_:
      node_counts += _count_dt_node_types(dt)
  elif(type(model) in [BinaryRelevance]):
    node_counts = np.zeros(3, dtype=int)
    for submodel in model.classifiers_:
      if(type(submodel) in [DecisionTreeRegressor]):
        aux = _count_dt_node_types(submodel)
        node_counts += aux
      elif(type(submodel) in [RandomForestRegressor]):
        for dt in submodel.estimators_:
          node_counts += _count_dt_node_types(dt)
      else:
        raise NotImplementedError(f'Node type counting not implemented for {type(model)}')
  elif(type(model) in [list, tuple]):
    node_counts = np.zeros(3, dtype=int)
    for submodel in model:
      node_counts += count_node_types(submodel)
  else:
    raise NotImplementedError(f'Node type counting not implemented for {type(model)}')

  return node_counts
