"""
Rationale for the selection of metrics implemented in this module.

(a) In our research project, we are modelling the task of indicating healthcare
    interventions to patients, in the context of gerontological primary care.
    -- The task consists in selecting a set/list of healthcare interventions based
       on the patient' scores obtained from a standardised instrument for
       comprehensive gerontological assessment (CGA).
    -- The task, thus, can be formalised either as a multilabel classification
       task or a label ranking task, depending on how the result of the task will
       be interpreted (i.e., the order of the labels being relevant or not).

(b) Thus, we selected metrics that are recurrently used in extensive evaluations of
    methods for multilabel classification and label ranking tasks:

  [1] Jasmin Bogatinovski, Ljupčo Todorovski, Sašo Džeroski, Dragi Kocev,
      "Comprehensive comparative study of multi-label classification methods.",
      Expert Systems with Applications, Volume 203, 2022,
      https://doi.org/10.1016/j.eswa.2022.117215.
      -- The authors report results from an evaluation of 26 multilabel
         classification methods vs. 42 benchmark datasets on 20 measures.
      -- Among these 20 measures, one finds: subset accuracy, and hamming loss
         (example-based), micro/macro F1 scores (label-based), one-error, ranking
         loss, and average precision (ranking-based).

  [2] Nicolás E. García-Pedrajas, José M. Cuevas-Muñoz, Gonzalo Cerruela-García,
      Aida de Haro-García, "A thorough experimental comparison of multilabel
      methods for classification performance.", Pattern Recognition, Vol. 151,
      2024, https://doi.org/10.1016/j.patcog.2024.110342.
      -- The authors report results from an evaluation of 62 multilabel
         classification methods vs. 65 benchmark datasets on 6 measures.
      -- Among the measures one finds: subset accuracy, and hamming loss
         (example-based), and the micro/macro F1 scores (label-based).

  [3] Nicolás E. García-Pedrajas, José M. Cuevas-Muñoz, Gonzalo Cerruela-García,
      Aida de Haro-García, "Extensive experimental comparison among multilabel
      methods focused on ranking performance.", Information Sciences, Vol. 679,
      2024, https://doi.org/10.1016/j.ins.2024.121074.
      -- The authors report results from an evaluation of 56 multilabel
         classification methods vs. 65 benchmark datasets on 6 measures.
      -- The measures are: one-error, coverage, ranking loss, average precision,
         and micro/macro-averaged AUC.

  [4] Fotakis, Dimitris, Alkis Kalavasis, and Eleni Psaroudaki.
      "Label ranking through nonparametric regression."
      In International Conference on Machine Learning, pp. 6622-6659. PMLR, 2022.
      -- The authors report results from an evaluation of 5 label ranking methods
         vs. 21 datasets on the Kendall's tau coefficient.

"""
import numpy as np
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

from collections import defaultdict
from scipy.stats import rankdata

from customDefs  import PrintBuffer
from polygrid    import Polygrid
from competitors import short_names
from competitors import LinearCompetitor, RidgeCompetitor, MLPCompetitor
from competitors import DTCompetitor, BRDTCompetitor, RFCompetitor, BRRFCompetitor
from competitors import RandomCompetitor
from competitors import mlpCompetitor
from metrics     import accuracy, f1_micro, f1_macro, f1_weigh, hammingl, jaccsim, auroc
from metrics     import ktau, lracc, lrloss
from datasets    import rank2presence, presence2rank

from datasets    import ECO_DB_UNLABELLED, ECO_DB_MULTICLASS
from datasets    import ECO_DB_MULTILABEL, ECO_DB_LABELRANK
from datasets    import ECO_PRESENCE, ECO_RANKING

ECO_REPORTKEYS = ['scenario', 'sizing', 'sizes', 'attc', 'cutoffs']

#--------------------------------------------------------------------------------------------------
# General purpose definitions - compute and handle bootstrapped confidence intervals
#--------------------------------------------------------------------------------------------------

_cimethod = 'bootstrap'
def CI(vals, alpha=0.05, minval=None, maxval=None, decimals=3):
  res = bs.bootstrap(np.array(vals), stat_func=bs_stats.mean, alpha=alpha, is_pivotal=True)
  lb = res.lower_bound if (minval is None) else max(res.lower_bound, minval)
  ub = res.upper_bound if (maxval is None) else min(res.upper_bound, maxval)
  return {'mean': np.round(res.value, decimals=decimals),
          'lb':   np.round(lb,        decimals=decimals),
          'ub':   np.round(ub,        decimals=decimals),
          'ss':   len(vals),
          'alpha': alpha,
         }

def overlap_CI(ci1, ci2):
  # returns true if two confidence intervals overlap
  lb = max(ci1['lb'], ci2['lb'])
  ub = min(ci1['ub'], ci2['ub'])
  return((ub - lb) >= 0.0)

#--------------------------------------------------------------------------------------------------
# General purpose definitions - define the set of metrics used in evaluation
#--------------------------------------------------------------------------------------------------

class MetricWrapper:
  def __init__(self, name, callable, watermark, lb, ub, rep):
    self.name      = name       # name of the metric (long name)
    self.callable  = callable   # the callable for an implementation of the metric
    self.watermark = watermark  # -np.inf if the metric encodes higher performance
                                #   with higher values; +np.inf otherwise
    self.lb        = lb         # the lower boundary of the range of the metric
    self.ub        = ub         # the upper boundary of the range of the metric
    self.rep       = rep        # ECO_PRESENCE if the metric expects to receive
                                # presence-encoded arguments (Y_real, Y_pred);
                                # ECO_RANKING otherwise

metricWrappers = {
    'accuracy': MetricWrapper('accuracy', accuracy, -np.inf,  0.0, 1.0, ECO_PRESENCE),
    'hammingl': MetricWrapper('hammingl', hammingl,  np.inf,  0.0, 1.0, ECO_PRESENCE),
    'f1.micro': MetricWrapper('f1.micro', f1_micro, -np.inf,  0.0, 1.0, ECO_PRESENCE),
    'f1.macro': MetricWrapper('f1.macro', f1_macro, -np.inf,  0.0, 1.0, ECO_PRESENCE),
    'f1.weigh': MetricWrapper('f1.weigh', f1_weigh, -np.inf,  0.0, 1.0, ECO_PRESENCE),
    'jaccsim' : MetricWrapper('jaccsim',  jaccsim,   np.inf,  0.0, 1.0, ECO_PRESENCE),
    'auroc'   : MetricWrapper('auroc',    auroc,    -np.inf,  0.0, 1.0, ECO_PRESENCE),
    'ktau'    : MetricWrapper('ktau',     ktau,     -np.inf, -1.0, 1.0, ECO_RANKING),
    'lracc'   : MetricWrapper('lracc',    lracc,    -np.inf,  0.0, 1.0, ECO_RANKING),
    'lrloss'  : MetricWrapper('lrloss',   lrloss,    np.inf,  0.0, 1.0, ECO_RANKING),
}

chosen_metrics = {
                  ECO_DB_MULTICLASS: [
                                      'accuracy',
                                      'hammingl',
                                      'f1.micro',
                                      'f1.macro',
                                      'f1.weigh',
                                      'jaccsim',
                                      #'auroc',
                                     ],
                  ECO_DB_MULTILABEL: [
                                      'accuracy',
                                      'hammingl',
                                      'f1.micro',
                                      'f1.macro',
                                      'f1.weigh',
                                      'jaccsim',
                                      #'auroc',
                                     ],
                  ECO_DB_LABELRANK:  [
                                      'lracc',
                                      'lrloss',
                                      'ktau',
                                     ],
                 }

def get_chosen_metrics(scenario):
  if(scenario not in chosen_metrics):
    raise NotImplementedError(f'No relevant metrics for {scenario} evaluation scenario')
  metrics = [metricWrappers[metric_name] for metric_name in chosen_metrics[scenario]]
  return metrics

def get_all_metric_names(scenarios):
  names = []
  for scenario in scenarios:
    if(scenario not in chosen_metrics):
      raise NotImplementedError(f'No relevant metrics for {scenario} evaluation scenario')
    for metric_name in chosen_metrics[scenario]:
      if(metric_name not in names):
        names.append(metric_name)
  return names

def has_usual_orientation(target_metric_name):
  # returns True if the metric indicates higher performance with larger values,
  #         False otherwise
  res = None
  for metric_name in metricWrappers:
    metric = metricWrappers[metric_name]
    if(metric.name == target_metric_name):
      res = (metric.watermark < 0)
      break
  if(res is None):
    raise ValueError(f'Support for {metric_name} metric not implemented')

  return res

#-------------------------------------------------------------------------------------------------------------------------------------------
# Problem-specific definitions: model instantiation
#-------------------------------------------------------------------------------------------------------------------------------------------

def get_model(model_type, model_config, model_hist=[]):

  (config, SEED, CORES) = model_config

  if(model_type == 'Polygrid'):
    model = Polygrid(**config)

  elif(model_type == 'MLP'):
    size = config
    model = MLPCompetitor(maxsize=size, seed=SEED, sizes=model_hist)

  elif(model_type == 'Linear'):
    size = config
    model = LinearCompetitor(maxsize=size, seed=SEED, sizes=model_hist)

  elif(model_type == 'Ridge'):
    size = config
    model = RidgeCompetitor(maxsize=size, seed=SEED, sizes=model_hist)

  elif(model_type == 'Random'):
    size = config
    model = RandomCompetitor(maxsize=size, seed=SEED, sizes=model_hist)

  elif(model_type == 'DT'):
    size = config
    model = DTCompetitor(maxsize=size, seed=SEED, sizes=model_hist)

  elif(model_type == 'RF'):
    size = config
    model = RFCompetitor(maxsize=size, seed=SEED, sizes=model_hist)

  elif(model_type == 'BRDT'):
    size = config
    model = BRDTCompetitor(maxsize=size, seed=SEED, sizes=model_hist)

  elif(model_type == 'BRRF'):
    size = config
    model = BRRFCompetitor(maxsize=size, seed=SEED, sizes=model_hist)

  elif(model_type == 'mlp'):
    size = config
    model = mlpCompetitor(maxsize=size, seed=SEED, sizes=model_hist)

  else:
    raise NotImplementedError(f'Model {model_type} not implemented')

  return model

#-------------------------------------------------------------------------------------------------------------------------------------------
# Problem-specific definitions: performance evaluation
#-------------------------------------------------------------------------------------------------------------------------------------------
def evaluate(Y_real, Y_pred, YorU_hat, scenario):

  metrics = get_chosen_metrics(scenario)
  evaluation = {'scenario': scenario}
  report = PrintBuffer()
  report.add(f'-- scenario: {scenario}')
  for metric in metrics:

    if(scenario == ECO_DB_LABELRANK and metric.rep == ECO_PRESENCE):
      # (Y_real, Y_pred) are raking-encoded, but metric expects presence-encoding
      Y_real_ = rank2presence(Y_real)
      Y_pred_ = rank2presence(Y_pred)
      evaluation[metric.name] = metric.callable(Y_real_, Y_pred_, YorU_hat)
    else:
      # assumes the metric is defined over whatever encoding Y is using
      evaluation[metric.name] = metric.callable(Y_real, Y_pred, YorU_hat)
    report.add(f'   {metric.name:<8}: {evaluation[metric.name]:5.3f}')

  return (evaluation, report)

def estimate(evaluations, scenario, alpha=0.05):

  sample = defaultdict(list)
  for evaluation in evaluations:
    for metric_name in evaluation:
      if(metric_name not in ECO_REPORTKEYS):
        sample[metric_name].append(evaluation[metric_name])

  new_evaluation = {'scenario': scenario}
  report = PrintBuffer()
  report.add(f'-- scenario: {scenario}')
  for metric_name in sample:
    metric = metricWrappers[metric_name]
    new_evaluation[metric_name] = CI(sample[metric_name],
                                     alpha=alpha,
                                     minval=metric.lb,
                                     maxval=metric.ub,
                                     decimals=3,)
    report.add(f'   {metric_name:<8}: {new_evaluation[metric_name]}')
    new_evaluation[metric_name]['sample'] = sample[metric_name]

  return (new_evaluation, report)

#-------------------------------------------------------------------------------------------------------------------------------------------
# Problem-specific definitions: detailed analysis of label-level errors
#-------------------------------------------------------------------------------------------------------------------------------------------

def analyse_errors(y_real, y_pred, scenario):

  if(scenario == ECO_DB_LABELRANK):

    # (y_real, y_pred) are ranking-encoded, so we just extract the labelseq
    y_real_seq = [e for e in y_real if e >= 0]
    y_pred_seq = [e for e in y_pred if e >= 0]

    # counts any bubbles in y_pred
    bb = 0
    state = 0 # starts at the reading state
    for e in y_pred:
      if(state == 0 and e == -1):
        # filler found during reading labels: goes to the closing state
        state = 1
      elif(state == 1 and e >= 0):
        # label found during closing state: goes back to reading state
        bb += 1
        state = 0

    # counts the number of edits
    permutation = []
    for e in y_pred_seq:
      try:
        pos = y_real_seq.index(e)
        permutation.append(pos)
      except ValueError:
        pass
    ne = len(permutation)
    ed = 0
    pivot = 0
    for pos in range(ne-1):
      if(permutation[pivot] > permutation[pos+1]):
        ed += 1
      else:
        pivot = pos+1

  else:
    # (y_real, y_pred) are presence-encoded, so we convert to ranking-encoding
    # and extract the labelseq
    y_real_seq = np.where(y_real==1)[0].tolist()
    y_pred_seq = np.where(y_pred==1)[0].tolist()

    bb = 0
    ed = 0

  # counts false negative and positive cases
  fn = sum([1 for e in y_real_seq if e not in y_pred_seq])
  fp = sum([1 for e in y_pred_seq if e not in y_real_seq])

  return (fn, fp, bb, ed)

def diagnose(Y_real, Y_pred, YorU_hat, scenario):

  header = ['FN', 'FP', 'BB', 'ED']

  (m,n) = Y_real.shape
  content = [analyse_errors(Y_real[i], Y_pred[i], scenario) for i in range(m)]
  Y_errstats = np.array(content, dtype=np.int64)
  report = ['\t'.join(header)]
  for i in range(m):
    buffer = [str(e) for e in Y_errstats[i]]
    rowbuffer = '\t'.join(buffer)
    report.append(rowbuffer)

  report.append('-' * 26)
  buffer = [str(e) for e in Y_errstats.sum(axis=0)]
  rowbuffer = '\t'.join(buffer)
  report.append(rowbuffer)
  report = '\n'.join(report)
  return (report, header, Y_errstats)

def get_tikz_diag(Y_real, Y_pred, header, Y_errstats, scenario):

  # template has three slots, [CONTENT1], [CONTENT2], and [CONTENT3]
  template = r'''
\documentclass{standalone}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, matrix, fit}
\begin{document}

    \begin{tikzpicture}[
        cell/.style={fill=red!20},
        filler/.style={fill=gray!30},
        ]
        \matrix [nodes={minimum width=20}]{
        [CONTENT1]
        };
        \node[above,font=\large\bfseries] at (current bounding box.north) {Target};
    \end{tikzpicture}

    \begin{tikzpicture}[
        cell/.style={fill=red!20},
        filler/.style={fill=gray!30},
        ]
        \matrix [nodes={minimum width=20}]{
        [CONTENT2]
        };
        \node[above,font=\large\bfseries] at (current bounding box.north) {Predicted};
    \end{tikzpicture}

    \begin{tikzpicture}[
        inner cells style/.style={fill=gray!20},
        first row style/.style={fill=white!100, rotate=90, anchor=west, font=\scriptsize},
        ]
        \matrix [
            nodes={inner cells style},
            row 1/.style={nodes={first row style}},
        ]
        {
        [CONTENT3]
        };
    \end{tikzpicture}

\end{document}
  '''

  def matrix2tikztab(Y, header=None, fmt=None):
    (m,n) = Y.shape

    if(header is not None):
      if(len(header) == n):
        buffer = [f"\\node{{{ colname }}};" for colname in header]
        rowbuffer = ' & '.join(buffer) + '\\\\'
        content = [rowbuffer]
      else:
        raise ValueError('Header has less elements than the matrix has columns')
    else:
      content = []

    if(fmt is None):
      for i in range(m):
        buffer = [f"\\node{{{ Y[i,j] }}};" for j in range(n)]
        rowbuffer = ' & '.join(buffer) + '\\\\'
        content.append(rowbuffer)
    else:
      for i in range(m):
        buffer = [f"\\node[style={'filler' if Y[i,j] < 0 else 'cell'}]{{{ Y[i,j] }}};" for j in range(n)]
        rowbuffer = ' & '.join(buffer) + '\\\\'
        content.append(rowbuffer)

    return content

  if(scenario == ECO_DB_LABELRANK):
    Y_real_seq = Y_real
    Y_pred_seq = Y_pred
  else:
    Y_real_seq = presence2rank(Y_real)
    Y_pred_seq = presence2rank(Y_pred)

  conditionalfmt = lambda e: 'filler' if e < 0 else 'cell'
  content1 = matrix2tikztab(Y_real_seq, header=None, fmt=conditionalfmt)
  content2 = matrix2tikztab(Y_pred_seq, header=None, fmt=conditionalfmt)
  content3 = matrix2tikztab(Y_errstats, header=header)

  content = template
  content = content.replace('[CONTENT1]', '\n'.join(content1))
  content = content.replace('[CONTENT2]', '\n'.join(content2))
  content = content.replace('[CONTENT3]', '\n'.join(content3))

  return content


#-------------------------------------------------------------------------------------------------------------------------------------------
# Problem-specific definitions: ranking models based on performance
#-------------------------------------------------------------------------------------------------------------------------------------------
# reusing the API designed by Mirko Bunse for a (Demsar, 2006)-compatible
# performance comparison framework. See https://github.com/mirkobunse/critdd
class Diagram:

  def __init__(self, X, X_lb=None, X_ub=None, treatment_names=None, maximize_outcome=True):
    """
    X is a (m,n)-real matrix, with the average performance data of n models on
    m datasets (for some metric). X_lb and X_ub have the same dimensions, but hold
    the lower and upper bounds of the confidence interval of the mean in X,
    respectively (3 sheets).

    treatment_names is a list with the names of the models.

    if maximize_outcome is True, then the metric used to assess performance encodes
    higher performance with larger numerical values, and the opposite otherwise.
    """
    (m,n) = X.shape
    self.m = m
    self.n = n

    if(treatment_names is None):
      treatment_names = [f'treatment {i}' for i in range(n)]
    elif(len(treatment_names) != n):
      raise ValueError(f'len(treatment_names) != {n}')

    self.competitors      = treatment_names
    self.short_names      = short_names
    self.maximize_outcome = maximize_outcome

    self.average_ranks    = self.compute_average_ranks(X)
    self.adjacency        = self.compute_dominance_DAG(X, X_lb, X_ub)
    self.echelons         = self.compute_echelons()

  @property
  def treatment_names(self):
    return self.competitors

  def compute_average_ranks(self, X, maximize_outcome=None):
    """
    X is a (m,n)-matrix; The columns of X must each correspond position-wise to the
    model in self.competitors. The rows of X must each correspond to some dataset,
    but they are not required to follow a specific order.
    """
    if(maximize_outcome is None):
      maximize_outcome = self.maximize_outcome
    # the results of this ranking method were manually compared to an Excel pivot table
    # [row=model, col=average(rank), filter=(metric,dataset)], built on the
    # data from the 'healthcare_evaluation_rank.csv'; same results obtained.
    ranks_per_dataset = rankdata(-X if self.maximize_outcome else X, method='average', axis=1)
    res = np.mean(ranks_per_dataset, axis=0).tolist()
    return res

  def compute_dominance_DAG(self, X, X_lb, X_ub, maximize_outcome=None):
    if(maximize_outcome is None):
      maximize_outcome = self.maximize_outcome

    (m,n) = (self.m, self.n)
    adjacency = np.zeros((n,n), dtype=int)
    for i in range(m):
      for j0 in range(n-1):
        ci0 = {'lb': X_lb[i,j0], 'ub': X_ub[i,j0]}
        for j1 in range(j0+1,n):
          ci1 = {'lb': X_lb[i,j1], 'ub': X_ub[i,j1]}
          if(not overlap_CI(ci0, ci1)):
            """
            1. Since the confidence intervals do not overlap, and the point estimates
               X[i,j0] and X[i,j1] sit each inside one of them, then it must be the
               case that they are significantly different (from a statistical PoV).
            2. The adjacency matrix encodes the dominance of (the observed performance
               of) one model over another, resulting in a directed acyclic graph (DAG).
            3. The DAG is represented by its adjacency matrix, which reads:
               adjacency[j0,j1] = p means that the j0-model dominates the j1-model in
               p datasets. Thus, adjacency[j0,j1] + adjacency[j1,j0] + ties = m*(n-1)
            """
            if(maximize_outcome):
              if(X[i,j0] > X[i,j1]):
                adjacency[j0,j1] += 1
              elif(X[i,j0] < X[i,j1]):
                adjacency[j1,j0] += 1
            else:
              if(X[i,j0] < X[i,j1]):
                adjacency[j0,j1] += 1
              elif(X[i,j0] > X[i,j1]):
                adjacency[j1,j0] += 1

      self.echelons = None

    return adjacency

  def compute_echelons(self, CD=1):

    (m,n) = (self.m, self.n)
    competitors   = self.treatment_names
    average_ranks = self.average_ranks

    """
    The script follows this analogy:
    Think of models as people. We want to staff our company.
    - The top ranking model is hired as the leader of the 1st echelon.
    - Once hired, the leader must fill the first echelon with competitive models.
    - A model M_{j1} is competitive with model M_{j0} if:
      abs(the number of times M_{j1} is dominated by model M_{j0} -
          the number of times M_{j0} is dominated by model M_{j1}) <= CD
    - Once the first echelon is completed, the next top ranking model (that has not yet
      been hired) is then hired as the leader of the 2nd echelon and the process restarts.
    - The process stops when all models have been hired.

    Thus,
    - Echelons are hierarchichally organised in levels.
    - Each model in the "1st echelon" dominates every model in the "2nd echelon",
      each model in the "2nd echelon" dominates every model in the "3rd echelon", etc.
    - An echelon is represented by a non-empty set (of model indices).
    """
    echelons = defaultdict(list)
    level = 0
    hired = set() # tracks the models that have been hired to some echelon

    # sorts/visits the models by average rank precedence
    L1 = sorted([(j, avrank) for (j, avrank) in enumerate(average_ranks)], key=lambda e:e[1])
    for (j0, avrank) in L1:
      if(j0 not in hired):
        hired.add(j0)
        echelons[level].append(j0)

        # seeks competitive models to hire
        for j1 in range(self.n):
          if(j1 not in hired):
            if(abs(self.adjacency[j0,j1] - self.adjacency[j1,j0]) <= CD):
              # the j0-th and j1-th models are competitive,
              # so the j1-th model is hired to the same echelon of j0
              hired.add(j1)
              echelons[level].append(j1)

        if(len(hired) == n):
          break
        else:
          level += 1

    return echelons

  def get_groups(self, alpha=.05, adjustment="holm", return_names=False, return_singletons=True):
    '''
    alpha and adjustment are not used in this implementation: they have been maintaind to
    ensure compatibility with Mirko's interface
    '''
    competitors =self.competitors
    echelons = self.echelons
    n_echelons = len(echelons)

    if(return_singletons):
      flat_echelons = [echelons[l] for l in range(n_echelons)]
    else:
      flat_echelons = [echelons[l] for l in range(n_echelons) if len(echelons[l]) > 1]

    if(return_names):
      L = []
      for echelon in flat_echelons:
        echelon_names = [competitors[j] for j in echelon]
        L.append(echelon_names)
      flat_echelons = L

    return tuple([tuple(flat_echelons[k]) for k in range(n_echelons)])

  def get_tikz_dag(self):

    n = self.n
    competitors = self.competitors
    short_names = self.short_names
    adjacency = self.adjacency

    template = r'''
      \documentclass{standalone}
      \usepackage{tikz}
      \usetikzlibrary{shapes.geometric, arrows.meta, matrix, fit}
      \begin{document}
          \begin{tikzpicture}[
              inner cells style/.style={fill=red!20},
              first row style/.style={fill=white!100, rotate=90, anchor=west, font=\tiny},
              first column style/.style={fill=white!100,anchor=east, font=\tiny},
              ]
            \matrix [
              nodes={inner cells style},
              row 1/.style={nodes={first row style}},
              column 1/.style={nodes={first column style}},
            ]
            {
            [CONTENT]
            };
          \end{tikzpicture}
      \end{document}
    '''

    content = []
    for j0 in range(n + 1):

      if(j0 == 0):
        # builds first row with model names
        L0 = [''] + [short_names[model_name] for model_name in competitors]
        L1 = [f'\\node{{{model_name}}};' for model_name in L0]

      else:
        # builds other rows (with model name in the first column,
        # and the entries of the adjacency matrix in the remaining columns
        L0 = [short_names[competitors[j0-1]]] + [adjacency[j0-1,j1] for j1 in range(n)]
        L1 = [f'\\node{{{value}}};' for value in L0]

      buffer = ' & '.join(L1) + '\\\\'
      content.append(buffer)

    wrapped_content = template.replace('[CONTENT]', '\n'.join(content))

    return wrapped_content

  def get_tikz_grp(self):

    n = self.n
    competitors = self.competitors
    short_names = self.short_names
    average_ranks = self.average_ranks
    echelons = self.echelons
    n_echelons = len(echelons)

    template = r'''
      \documentclass{standalone}
      \usepackage{tikz}
      \usetikzlibrary{shapes.geometric, arrows.meta, matrix, fit}
      \begin{document}
          \begin{tikzpicture}[node distance=0.7cm,
              block/.style={rectangle, rounded corners=3pt, minimum width=1.5cm, minimum height=.4cm,text centered, draw=none, fill=orange!30, font=\fontsize{8}{1}\selectfont},
              line/.style={draw, line width=1.0pt, rounded corners=3pt, -{Latex[length=0.8mm]}}
              ]
              [CONTENT]
          \draw [color=red] (-2.4, -0.01) -- (-2.4, 0.01);
          \draw [color=red] ( 2.4, -0.01) -- ( 2.4, 0.01);
          \end{tikzpicture}
      \end{document}
    '''

    nodes = []
    edges = []
    for level in range(n_echelons):
      L = []
      for model_idx in echelons[level]:
        model_name = short_names[competitors[model_idx]]
        model_rank = average_ranks[model_idx]
        L.append(f'{model_name} ({model_rank:4.2f})')

      if(level == 0):
        buffer = f"\\node (e{level:d}) [block]              {{{', '.join(L)}}};"
      else:
        buffer = f"\\node (e{level:d}) [block, below of=e{level-1:d}] {{{', '.join(L)}}};"
      nodes.append(buffer)

      if(level < n_echelons - 1):
        edges.append(f'\\path [line] (e{level:d}) -- (e{level+1:d});')

    content = nodes + edges
    wrapped_content = template.replace('[CONTENT]', '\n'.join(content))

    return wrapped_content

