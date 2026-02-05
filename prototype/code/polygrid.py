import os
import gettext
import secrets
import numpy as np

from copy                     import copy
from math                     import floor, ceil
from collections              import defaultdict, deque, namedtuple
from itertools                import permutations

from shapely.geometry         import Point, Polygon
from scipy.optimize           import minimize, differential_evolution
from scipy.linalg             import circulant
from sklearn.linear_model     import Ridge, Lasso, RidgeCV, LassoCV
from sklearn.tree             import DecisionTreeRegressor, export_text

from vorder                   import VerticesOrderer
from datasets                 import rank2presence, prune_rank
from metrics                  import f1_micro, f1_macro, f1_weigh, hammingl, mse
from layoutman                import LayoutManager

from matplotlib               import pyplot as plt, colors
from matplotlib               import ticker
from matplotlib.text          import Text
from matplotlib.patches       import Circle, Arc
from matplotlib.collections   import LineCollection, PolyCollection, PatchCollection
from matplotlib.backend_bases import MouseEvent

from customDefs               import ECO_SEED, ECO_DPI
from customDefs               import ECO_DEFICIT, ECO_CAPACITY
from customDefs               import ECO_CUTOFF_SINGLE, ECO_CUTOFF_MULTIPLE
from customDefs               import ECO_THRSHLDLVLS
from customDefs               import ECO_HIT, ECO_MISS
from datasets                 import ECO_DB_UNLABELLED, ECO_DB_MULTICLASS
from datasets                 import ECO_DB_MULTILABEL, ECO_DB_LABELRANK

# data type to store description of graphical elements in interactive diagrams
ListenerData = namedtuple('ListenerData', ['ec', 'fc', 'lw', 'alpha', 'zorder'])

# constants identifying polygon drawing patterns
ECO_PTRN_DEMAND = 0
ECO_PTRN_OFFER  = 1
ECO_PTRN_MATCH  = 2
ECO_PTRN_CROSS  = 3
ECO_PTRN_THIN1  = 5
ECO_PTRN_THIN2  = 6

# constants identifying diagrams subplot unit sizes
ECO_UNITSIZEW = 5.6
ECO_UNITSIZEH = 5.4

# constant used to control the cutoff ratio for small singular values in
# np.linalg.lstsq
ECO_RCOND = 1E-2

class CustomList:

  def __init__(self, L):
    self.L = L

  def index(self, j, default=None):
    try:
      res = self.L.index(j)
    except ValueError:
      res = default if default is not None else j
    return res

#-------------------------------------------------------------------------------------------------------------------------------------------
# Problem-specific definitions: Polygrid learning model
#-------------------------------------------------------------------------------------------------------------------------------------------
def validate_Polygrid_config(config):

  for param in config:

    if(param == 'nspd'):
      nspd = config[param]
      if(not(type(nspd) is int and nspd > 0)):
        raise ValueError(f'Invalid number of sectors per domain: {nspd}')

    if(param == 'na'):
      na = config[param]
      if(not(type(na) is int and na > 0)):
        raise ValueError(f'Invalid number of annuli: {na}')

    if(param == 'vorder'):
      vorder = config[param]
      if(vorder not in ['original', 'rho', 'rho-squared', 'averages', 'measures']):
        raise ValueError(f'Invalid vorder option: {vorder}')

    if(param == 'init'):
      init = config[param]
      if(init not in ['full', 'averages']):
        raise ValueError(f'Invalid initialisation option: {init}')

    if(param == 'norm'):
      norm = config[param]
      if(norm not in ['none', 'm-n', 'm-nas', 'n-nas', 'm', 'n', 'nas']):
        raise ValueError(f'Invalid normalisation option: {norm}')

    if(param == 'solver'):
      solver = config[param]
      if(solver not in ['lstsq', 'lstsqsym', 'lstsquni', 'ridge', 'lasso', 'identity', 'braids']):
        raise ValueError(f'Invalid solver option: {solver}')

    if(param == 'geoeng'):
      geoeng = config[param]
      if(geoeng not in ['shapely', ]):
        raise ValueError(f'Invalid computational geometry engine option: {geoeng}')

    if(param == 'annulus_type'):
      annulus_type = config[param]
      if(annulus_type not in ['r-invariant', 's-invariant', 'tree', 'Tree', 'random']):
        raise ValueError(f'Invalid annulus type option: {annulus_type}')

    if(param == 'sector_type'):
      sector_type = config[param]
      if(sector_type not in ['cover', 'miss', 'random']):
        raise ValueError(f'Invalid sector type option: {sector_type}')

    if(param == 'RAD'):
      RAD = config[param]
      if(RAD <= 0.):
        raise ValueError(f'Disc radius must be positive: {RAD}')

    if(param == 'ARES'):
      ARES = config[param]
      if(ARES < 1):
        raise ValueError(f'Annulus resolution must be larger than 1: {ARES}')

    if(param == 'SRES'):
      SRES = config[param]
      if(SRES < 1):
        raise ValueError(f'Sector resolution must be larger than 1: {SRES}')

    if(param == 'CORES'):
      CORES = config[param]
      if(CORES < 1):
        raise ValueError('Number of cores should be at least one')

  return True

def build_Polygrid_config(nspd, na, defaults=None):

  if(defaults is None):
    config = {'vorder':       'rho',
              'init':         'full',
              'norm':         'none',
              'solver':       'lstsq',
              'geoeng':       'shapely',
              'annulus_type': 's-invariant',
              'sector_type':  'miss',
              'RAD':          1.0,
              'ARES':         8,
              'SRES':         8,
               }
  else:
    config = copy(defaults)

  config['na'] = na
  config['nspd'] = nspd
  return config

class Polygrid:
  """
  This class implements the Polygrid learning model introduced in :

  [1] Andre Paulino de Lima, Brunela Orlandi, Suzana Andrade, Rosa Marcucci,
      Ruth Caldeira de Melo and Marcelo Garcia Manzato. 2025.
      "An Interpretable Recommendation Model that Leverages Psychometric Data in
      Gerontological Primary Care." ACM Trans. Recomm. Syst. Just Accepted (April 2025).
      https://doi.org/10.1145/_________

  Parameters:

  nspd (int) ..........: number of sectors per domain (used to segment the disc)
  na (int) ............: number of annuli (used to segment the disc)
  vorder (str) ........: the method used to determine how variables are chained around the disc
  initialisation (str) : the method used for weight initialisation
  normalisation (str) .: the method used for normalising the S tensor
  solver (str) ........: the method used to solve the system of equations
  geoeng (str) ........: the geometry engine to use (shapely or simple)
  annulus_type (str) ..: the type of annulus to use: same area ('s-invariant') or same with ('r-invariant')
  sector_type (str) ...: the type of sector to use: nearly miss domain axes ('miss') or cover ('cover')
  RAD (float) .........: radius of the disc
  ARES (int) ..........: resolution of an annulus, used in shapely.Point.buffer(..., resolution=_ARES)
  SRES (int) ..........: resolution of a sector (the number of vertices to approximate the sector)
  """

  def __init__(self, nspd   = 1,
                     na     = 1,
                     vorder = 'rho',
                     init   = 'full',
                     norm   = 'none',
                     solver = 'lstsq',
                     geoeng = 'shapely',
                     annulus_type = 's-invariant',
                     sector_type  = 'miss',
                     RAD    = 1.0,
                     ARES   = 8,  # approximates a disc by a regular n-gon with n=4*ARES
                     SRES   = 8,
                     CORES  = 4,
                     SEED   = ECO_SEED):

    # validates the parameter values
    self.validate(nspd, na, vorder, init, norm, solver, geoeng, annulus_type, sector_type,
                  RAD, ARES, SRES, CORES, SEED)

    # list of attributes with values set during instantiation
    # (size-determining or structure-defining model parameters)
    self.nspd   = nspd
    self.na     = na
    self.vorder = vorder
    self.init   = init
    self.norm   = norm
    self.solver = solver
    self.annulus_type = annulus_type
    self.sector_type  = sector_type

    self.corder = 'original'
    self.ang0   = 0.0
    self.RAD    = RAD
    self.ARES   = ARES
    self.SRES   = SRES
    self.SEED   = SEED

    # list of attributes determined during data fitting
    # (workarea)
    self.T   = None # (n,d)-array with offer weights (floats)
    self.P   = None # (n,d)-array with average/median levels of features (floats)
    self.M   = None # dict[i][j] with the match between each demand i and offer j (shapely Polygons)
    self.AS  = None # list of <nas> annular sectors into which the disc is partitioned (shapely polygons)
    self.S   = None # (m,n,nas)-array with the area of the intersection between each annular sector and a matching polygon in M
    self.W   = None # (n,nas)-array of learned weights that solve SW=Y (or SW=U in some cases)
    self.C   = None # n-array of learned intercepts (used when the solver is ridge/lasso)
    self.tw  = None # n-array of learned thresholds used as cutoff in the output scales
    self.svs = None # nas-array of singular values returned by np.linalg.lstsq in Polygrid.learning_weights
    self.rnk = None # integer; the rank of the matrix S transformed by the np.linalg.lstsq in Polygrid.learning_weights

    self.m   = None # (int)  number of instances
    self.n   = None # (int)  number of classes/labels
    self.d   = None # (int)  number of features
    self.ns  = None # (int)  number of sectors
    self.nas = None # (int)  number of annular sectors
    self.nw  = None # (int)  number of model weights/parameters
    self.vo  = None # (list) ordering of the vertices
    self.co  = None # (list) ordering of the learned offers

    self.radii   = None # list of annuli boundaries (0 and RAD included)
    self.angles  = None # list of sector boundaries
    self.annuli  = None # dictionary of polygonal shapes approximating annuli
    self.sectors = None # dictionary of polygonal shapes approximating sectors

    # list of attributes that are set with a default value, but are modifiable at runtime
    self.DPI        = ECO_DPI
    self.model_name = 'Polygrid'
    self.sym        = ECO_CAPACITY
    self.scenario   = ECO_DB_MULTICLASS
    self.cutoff     = ECO_CUTOFF_MULTIPLE
    self.metric     = f1_micro
    self.rcond      = ECO_RCOND
    self.avoid_FP   = False # if True, avoids FP during learning_threshold, and avoids
                            # FN otherwise. Can be used in situations where one type of
                            # error hurts trust more than the other, for example.


    # working area
    self.inspection = None

    # attributes that are used to handle diagrams and to control user interaction
    # (workarea)
    self.dataset_name    = None
    self.feature_names   = None
    self.target_names    = None
    self.sample_noun     = None
    self.class_noun      = None

    self.fig             = None # the figure on which events will be captured
    self.cbar            = None # the colour bar added to the Figure
    self.titleTips       = None # ties every pair of (offer/demand title, offer/demand tag)
    self.labelTips       = None # ties every dimension label with the tag of the first offer
    self.offerTips       = None # ties every pair of (offer segment, offer tag)
    self.demandTips      = None # ties every pair of (demand segment, demand tag)
    self.crossTips       = None # ties every pair of (cross segment, cross tag)
    self.cbar_points     = None # list of points that are currently plotted on cbar
    self.chart_marks     = None # list of marks that are currently being shown on charts
    self.last_k          = None # the last segment on which the user clicked
    self.bg_layer_offer  = None # data used to format the background shapes in offer charts
    self.bg_layer_demand = None # data used to format the background shapes in demand charts
    self.bg_layer_cross  = None # data used to format the background shapes in cross charts

  def validate(self, nspd, na, vorder, init, norm, solver, geoeng,
                     annulus_type, sector_type,
                     RAD, ARES, SRES, CORES, SEED):

    config = locals()
    validate_Polygrid_config(config)
    return None

  def reset_plot(self, fig, cbar):
    self.fig             = fig
    self.cbar            = cbar
    self.titleTips       = []
    self.labelTips       = []
    self.offerTips       = defaultdict(list)
    self.demandTips      = defaultdict(list)
    self.crossTips       = defaultdict(list)
    self.cbar_points     = []
    self.chart_marks     = []
    self.last_k          = 0
    self.bg_layer_offer  = None
    self.bg_layer_demand = None
    self.bg_layer_cross  = None
    return None

  def setup_vorder(self, X, maxiter=100):
    vorderer = VerticesOrderer(self.vorder)
    self.vo = vorderer.seq(X, maxiter)
    return None

  def get_reverse_vorder(self):
    seq = []
    for k in range(self.d):
      seq.append(self.vo.index(k))
    return seq

  def setup_corder(self, corder):
    self.corder = corder
    if(self.corder == 'original'):
      self.co = list(range(self.n))
    elif(self.corder == 'measures'):
      L0 = [(i, self.scores2poly(self.P[i]).area) for i in range(self.n)]
      L1 = sorted(L0, key=lambda e:e[1])
      #print(f'-- corder is {L1}')
      self.co = [i for (i,s) in L1]
    else:
      raise ValueError(f'Unknown class order type: {self.corder}')
    return None

  def init_weights(self, X, Y, U = None):
    (n, d) = (self.n, self.d)
    self.setup_vorder(X)

    # obtains the average levels of each feature for each class
    # (used in rendering offers, no purpose in learning, only visualisation)
    P = np.zeros((n, d))
    if(self.scenario in [ECO_DB_MULTICLASS, ECO_DB_MULTILABEL]):
      for j in range(n):
        idxs = np.where(Y[:, j] == 1)
        P[j] = np.mean(X[idxs][:, self.vo], axis=0)

    elif(self.scenario in [ECO_DB_LABELRANK]):
      # -- proposal #2 - each offer equals the average of demands assigned to it
      # -- strategy: uses membership data to deduce label presence
      # -- pros: produces visual content the user can rely on to simulate the task
      #          most similar with multilabel classification scenario
      # -- cons: the more complete the order, the more this scenario converges to proposal #1
      _Y = rank2presence(Y)
      for j in range(n):
        idxs = np.where(_Y[:, j] == 1)
        P[j] = np.mean(X[idxs][:, self.vo], axis=0)

    else:
      raise NotImplementedError(f'Scenario {self.scenario} not implemented')

    if(  self.init == 'full'):
      T = np.ones((n, d))
    elif(self.init == 'averages'):
      T = copy(P)

    return (P,T)

  def polar2coord(self, r, theta):
    x = r * np.cos(theta)  # the x-coord of the vertex that sits on the axis defined by theta
    y = r * np.sin(theta)  # the y-coord of the vertex that sits on the axis defined by theta
    return (x,y)

  def scores2coords(self, scores):
    d = self.d
    ra = 2 * np.pi / d  # angle between two axes, in radians
    thetas = [(k*ra) for k in range(d)] # angles at which domain axes are located
    L = [self.polar2coord(scores[k] * self.RAD, thetas[k]) for k in range(d)]
    return L

  def coords2poly(self, coords):
    return Polygon(coords)

  def poly2coords(self, polygon): # used in show_cross
    xs, ys = polygon.exterior.coords.xy
    L = [(xs[i], ys[i]) for i in range(len(xs) - 1)]
    return L

  def scores2poly(self, scores):
    return self.coords2poly(self.scores2coords(scores))

  def ones(self):
    return [1. for _ in range(self.d)]

  def segment_disc(self, X=None, Y=None, U=None):
    (na, ns, RAD, ARES, SRES) = (self.na, self.ns, self.RAD, self.ARES, self.SRES)

    # defines the annuli demarcating different domain levels
    if(  self.annulus_type == 'r-invariant'):
      dr = RAD / na     # annuli will be equally thick
      radii  = [i * dr for i in range(na + 1)]
    elif(self.annulus_type == 's-invariant'):
      ds = RAD**2 / na  # annuli will have the same area
      radii = [np.sqrt(i * ds) for i in range(na + 1)]
    elif(self.annulus_type == 'random'):
      if(na == 1):
        radii = [0., 1.]
      else:
        while(True):
          boundaries = np.random.uniform(size=na-1).tolist()
          radii = sorted([0., 1.] + boundaries)
          if(len(radii) == na + 1):
            break
    elif(self.annulus_type == 'tree'):
      # annuli will be placed in between thresholds found by a CART decision tree
      dt = DecisionTreeRegressor(max_depth=na,
                                 max_leaf_nodes=None,
                                 criterion='squared_error',
                                 splitter='best',
                                 random_state=ECO_SEED,
                                 )

      if(self.scenario in [ECO_DB_MULTICLASS, ECO_DB_MULTILABEL]):
        dt.fit(X, Y)
      elif(self.scenario in [ECO_DB_LABELRANK]):
        dt.fit(X, U)
      else:
        raise NotImplementedError(f'segment_disc cannot handle {self.scenario} scenario')

      thresholds = [(round(threshold,2)) for threshold in dt.tree_.threshold if threshold > 0]
      radii = [0., RAD]
      while(len(thresholds) > 0 and len(radii) < self.na + 1):
        candidate = thresholds.pop(0)
        nearest = min([abs(candidate - radius) for radius in radii])
        if(nearest > 0.05):
          radii.append(candidate)
      radii = sorted(radii)

      if(len(radii) < self.na + 1):
        # handles when candidates are in shortage
        missing = (self.na + 1) - len(radii)
        dr = (radii[-1] - radii[-2])/(missing + 1)
        filler = [radii[-2] + (i+1) * dr for i in range(missing)]
        radii = radii[0:-1] + filler + [RAD]

    elif(self.annulus_type == 'Tree'):
      # annuli will be placed in between thresholds found by a CART decision tree
      # --- experimental
      dt = DecisionTreeRegressor(max_depth=na,
                                 max_leaf_nodes=None,
                                 criterion='squared_error',
                                 splitter='best',
                                 random_state=ECO_SEED,
                                 )

      if(self.scenario in [ECO_DB_MULTICLASS, ECO_DB_MULTILABEL]):
        dt.fit(X, Y)
      elif(self.scenario in [ECO_DB_LABELRANK]):
        dt.fit(X, U)
      else:
        raise NotImplementedError(f'segment_disc cannot handle {self.scenario} scenario')

      decimals = 2
      mindif = 2 * 10 ** (-decimals)
      thresholds = [(round(threshold, decimals)) for threshold in dt.tree_.threshold if threshold > 0]
      #thresholds = [threshold for threshold in dt.tree_.threshold if threshold > 0]
      radii = [0., RAD]
      while(len(thresholds) > 0 and len(radii) < self.na + 1):
        candidate = thresholds.pop(0)
        nearest = min([abs(candidate - radius) for radius in radii])
        if(nearest >= mindif):
          radii.append(candidate)
      radii = sorted(radii)

      if(len(radii) < self.na + 1):
        # handles when candidates are in shortage
        missing = (self.na + 1) - len(radii)
        print(f'-- completing {missing} annuli')
        dr = (radii[-1] - radii[-2])/(missing + 1)
        filler = [radii[-2] + (i+1) * dr for i in range(missing)]
        radii = radii[0:-1] + filler + [RAD]

    annuli = {i: Point(0,0).buffer(radii[i+1], resolution=ARES).difference(
                 Point(0,0).buffer(radii[i],   resolution=ARES)) for i in range(na)}

    # defines the sectors demarcating different domain neighbourhoods
    da = 2 * np.pi / ns
    if(  self.sector_type == 'miss'):
      angles = [self.ang0 + (i * da) for i in range(ns+1)]
    elif(self.sector_type == 'cover'):
      angles = [self.ang0 + (i * da - da/2) for i in range(ns+1)]
    elif(self.sector_type == 'random'):
      if(ns == self.d):
        angles = [self.ang0 + (i * da) for i in range(ns+1)]
      else:
        da = 2 * np.pi / self.d
        while(True):
          boundaries = (2 * np.pi * np.random.uniform(size=(ns-self.d))).tolist()
          angles = [self.ang0 + (i * da) for i in range(self.d+1)]
          angles = sorted(angles + boundaries)
          if(len(angles) == ns+1):
            break

    sectors = {}
    for j in range(len(angles)-1):
      da = angles[j+1] - angles[j]
      das = da / SRES # delta angular step
      subangles = np.arange(angles[j], angles[j+1] + das, das)
      coords = [(0.0, 0.0)] + [self.polar2coord(RAD, subangles[k]) for k in range(self.SRES+1)]
      sectors[j] = Polygon(coords)

    # defines the annular sectors in which the disc is segmented
    AS  = [annuli[i].intersection(sectors[j]) for i in annuli for j in sectors]
    nas = len(AS) # number of annular sectors

    # ensures the annular sectors are are polygons
    for (k, geom) in enumerate(AS):
      if(type(geom) is not Polygon):
        for e in geom.geoms:
          if(type(e) is Polygon):
            AS[k] = e
            break

    self.radii   = radii
    self.angles  = angles
    self.annuli  = annuli
    self.sectors = sectors

    return (AS, nas)

  def crossdo(self, X):
    (m, n) = (X.shape[0], self.n)

    M = defaultdict(dict)
    if(self.init == 'full'):
      for i in range(m):
        demand = self.coords2poly(self.scores2coords(X[i,self.vo]))
        for j in range(n):
          M[i][j] = demand
    else:
      """
      This applies some form of clipping to the match polygon.
      Although an increase in performance was observed during the development tests,
      as this feature hurts diagram interpretability (match polygons with #vertices > d),
      we did not explore this feature in our thesis.
      (this is a remnant of the first model we explored during the research)
      """
      cutters = {j: self.coords2poly(self.scores2coords(self.T[j])) for j in range(n)}
      for i in range(m):
        demand = self.coords2poly(self.scores2coords(X[i,self.vo]))
        for j in range(n):
          M[i][j] = cutters[j].intersection(demand) # stores the resulting (single) shapely Polygon
    return M

  def decompose(self, M):
    (m, n, nas) = (len(M.keys()), self.n, self.nas)

    # creates the tensor S, which can be framed as an (m,n)-matrix
    # in which each element is a nas-vector
    S = np.zeros((m, n, nas))
    if(self.init == 'full'):
      for i in range(m):
        j = 0
        for k in range(nas):
          S[i,:,k] = M[i][j].intersection(self.AS[k]).area
    else:
      for i in range(m):
        for j in range(n):
          for k in range(nas):
            S[i,j,k] = M[i][j].intersection(self.AS[k]).area

    return S

  def normalise(self, S):
    """
    This applies some linear form of normalisation to the S tensor.
    Although an increase in performance was observed during the development tests,
    as this feature hurts diagram interpretability (proportions stop making sense),
    we did not include this feature in our thesis.
    """
    (m, n, nas) = (self.m, self.n, self.nas)

    if(S is None):
      S = self.S
    norm = self.norm

    if(norm in ['m-n']):
      # each mode-3 fiber (tube S[i,j,:]) is normalised to unit, meaning that:
      # - each of the (m x n) nas-sized vectors are normalised to unit
      # - each demand*offer (DO) is given equal measure
      #   (at the cost of inflating/deflating the measure of the DO in relation to that of other DOs)
      # - each lateral slice (S[:,j,:]) is made up of stochastic row vectors
      for i in range(m):
        for j in range(n):
          s_ = S[i,j,:].sum()      # surely scalar s_ > 0 because:
          S[i,j,:] = S[i,j,:]/s_   # (a) zeroes in X are replaced by a small value, AND
                                   # (b) zeroes do not occur in T (full/averages init'd)

    elif(norm in ['m-nas']):
      # each mode-2 fiber (tube S[i,:,k]) is normalised to unit, meaning that:
      # - each of the (m x nas) n-sized vectors are normalised to unit
      # - each demand*segment (DS) is given equal measure
      #   (at the cost of inflating/deflating the measure of the DS in relation to that of other DSs)
      for i in range(m):
        for k in range(nas):
          s_ = S[i,:,k].sum()      # scalar s_ >= 0; scaling is unsafe
          if(s_ > 0):
            S[i,:,k] = S[i,:,k]/s_
          else:
            S[i,:,k] = 1/n         # when demand i fails to cover region k for every offer, we stick
                                   # with the idea that tubes are normalised to unit
            #S[i,:,k] = 0.         # (replacing with zero does not seem modify model performance)

    elif(norm in ['n-nas']):
      # each mode-1 fiber (tube S[:,j,k]) is normalised to unit, meaning that:
      # - each of the (n x nas) m-sized vectors are normalised to unit
      # - each offer*segment (OS) is given equal measure
      #   (at the cost of inflating/deflating the measure of the OS in relation to that of other OSs)
      # - each lateral slice (S[:,j,:]) is made up of stochastic column vectors
      for j in range(n):
        for k in range(nas):
          s_ = S[:,j,k].sum()      # scalar s_ >= 0; scaling is unsafe
          if(s_ > 0):
            S[:,j,k] = S[:,j,k]/s_
          else:
            S[:,j,k] = 1/m         # (when offer j fails to cover region k for every demand)
            #S[:,j,k] = 0.         # xxx why not just zero?

    elif(norm == 'm'):
      # each horizontal slice (S[i,:,:]) is normalised to unit, meaning that:
      # - each of the m (n x nas) matrices are normalised to unit
      # - each demand is given equal measure
      #   (at the cost of inflating/deflating the measure of the demand in relation to that of others)
      s_ = S.sum(axis=(1,2))         # s_ ~ <m>
      for i in range(m):
        S[i,:,:] = S[i,:,:]/s_[i]    # scalar s_[i] > 0 because:
                                     # (a) zeroes in P are replaced by small value, AND
                                     # (b) zeroes do not occur in T (full/averages init'd)

    elif(norm == 'n'):
      # each lateral slice (S[:,j,:]) is normalised to unit, meaning that:
      # - each of the n (m x nas) matrices are normalised to unit
      # - each offer is given equal measure
      #   (at the cost of inflating/deflating the measure of the offer in relation to that of others)
      s_ = S.sum(axis=(0,2))         # s_ ~ <n>
      for j in range(n):
        S[:,j,:] = S[:,j,:]/s_[j]    # scalar s_[j] > 0 (same reason above)

    elif(norm == 'nas'):
      # each frontal slice (S[:,:,k]) is normalised to unit, meaning that:
      # - each of the nas (m x n) matrices are normalised to unit
      # - each segment is given equal measure
      #   (at the cost of inflating/deflating the measure of the offer in relation to that of others)
      s_ = S.sum(axis=(0,1))         # s_ ~ <nas>
      for k in range(nas):
        if(s_[k] > 0):
          S[:,:,k] = S[:,:,k]/s_[k]
        else:
          S[:,:,k] = 1/(m*n)         # (when demand i fails to cover region k for every offer)

    return S

  def learn_weights(self, S, Y, U=None):
    (n, nas) = (self.n, self.nas)
    (m, _)   = Y.shape

    W = np.zeros((n, nas))
    C = np.zeros(n)
    if(self.solver == 'lstsq'):

      if(self.scenario in [ECO_DB_MULTICLASS, ECO_DB_MULTILABEL]):
        """
        Obtains a least squares solution to each [SW=Y]_{j \\in 0..n-1} subsystem
        In these scenarios, Y encodes label presence/absence
        According to the v-policy we adopted,
          We have   Y == U in multiclass, so W_j=S^{-1}Y_j == S^{-1}U_j
          Possibly, Y != U in multilabel, so W_j=S^{-1}Y_j != S^{-1}U_j
        Both Y == U and Y != U in a multilabel scenario.
        Remember that multiclass is a specific case of multilabel with cardinality = 1,
        so Y == U in that case. When cardinality > 1, the policy we adopted will produce
        Y != U for any assignment track, namely multilabel -> (assign o) -> multilabel
        or label ranking -> (assign f .) -> multilabel)
        """
        for j in range(n):
          A = S[:,j]
          b = Y[:,j]
          x = np.linalg.lstsq(A, b, rcond=self.rcond)
          W[j] = x[0]

      elif(self.scenario in [ECO_DB_LABELRANK]):
        """
        Obtains a least squares solution to each [SW=U]_{j \\in 0..n-1} subsystem
        In this scenario, Y encodes label rankings
        In our experience, solving for SW=Y may leads to poor results
        """
        for j in range(n):
          A = S[:,j]
          b = U[:,j]
          x = np.linalg.lstsq(A, b, rcond=self.rcond)
          W[j] = x[0]

      else:
        raise NotImplementedError(f'Solver lstsq does not handle the {self.scenario} scenario')

    elif(self.solver == 'lstsqsym'):
      """
      This solver is identical to 'lstsq', except that the orignal encoding for
      Y (presence=1, absence=0) is replaced by a symmetric encoding before
      solving the [SW=Y]_j equations (presence=1, absence=-1).
      """
      if(self.scenario in [ECO_DB_MULTICLASS, ECO_DB_MULTILABEL]):

        for j in range(n):
          A = S[:,j]
          b = 2*Y[:,j]-np.ones(m)
          x = np.linalg.lstsq(A, b, rcond=self.rcond)
          W[j] = x[0]

      elif(self.scenario in [ECO_DB_LABELRANK]):
        """
        for label ranking, does exactly what the lstsq solver does
        """
        for j in range(n):
          A = S[:,j]
          c = U[:,j]
          x = np.linalg.lstsq(A, c, rcond=self.rcond)
          W[j] = x[0]

      else:
        raise NotImplementedError(f'Solver lstsqsym does not handle the {self.scenario} scenario')

    elif(self.solver == 'lstsquni'):
      """
      This solver applies a uniform treatment for all scenarios: solves SW=U subsystems
      """
      if(self.scenario in [ECO_DB_MULTICLASS, ECO_DB_MULTILABEL, ECO_DB_LABELRANK]):
        for j in range(n):
          A = S[:,j]
          c = U[:,j]
          x = np.linalg.lstsq(A, c, rcond=self.rcond)
          W[j] = x[0]
      else:
        raise NotImplementedError(f'Solver lstsquni does not handle the {self.scenario} scenario')

    elif(self.solver == 'ridge'):
      # potential benefits a ridge solver would bring to our application:
      # https://stats.stackexchange.com/questions/258808/why-we-use-ridge-regression-instead-of-least-squares-in-multicollinearity
      if(self.scenario in [ECO_DB_MULTICLASS, ECO_DB_MULTILABEL]):
        # obtains a ridge solution to each [SW+C=Y]_{j \in 0..n-1} system
        for j in range(n):
          A = S[:,j]
          b = Y[:,j]
          #clf = Ridge(solver='cholesky').fit(A, b)
          #--------------------------------------------------------------
          # xxx see https://www.youtube.com/watch?v=YGdXHo3M2gE&t=554s
          # focus on the use of Cholesky decomposition to enforce sparsity
          #--------------------------------------------------------------
          clf = RidgeCV(fit_intercept=True, scoring='r2').fit(A, b)
          W[j] = clf.coef_
          C[j] = clf.intercept_

      elif(self.scenario in [ECO_DB_LABELRANK]):
        # obtains a ridge solution to each [SW+C=U]_{j \in 0..n-1} system
        for j in range(n):
          A = S[:,j]
          c = U[:,j]
          #clf = Ridge(solver='cholesky').fit(A, c)
          clf = RidgeCV(fit_intercept=True, scoring='r2').fit(A, c)
          W[j] = clf.coef_
          C[j] = clf.intercept_

      else:
        raise NotImplementedError(f'Solver ridge does not handle the {self.scenario} scenario')

    elif(self.solver == 'lasso'):

      if(self.scenario in [ECO_DB_MULTICLASS, ECO_DB_MULTILABEL]):
        # obtains a lasso solution to each [SW+C=Y]_{j \in 0..n-1} system
        for j in range(n):
          A = S[:,j]
          b = Y[:,j]
          #clf = LassoCV(alpha=1., solver='cholesky', fit_intercept=True).fit(A, b)
          clf = LassoCV(fit_intercept=True, positive=False, max_iter=100000).fit(A, b)
          W[j] = clf.coef_
          C[j] = clf.intercept_

      elif(self.scenario in [ECO_DB_LABELRANK]):
        # obtains a lasso solution to each [SW+C=U]_{j \in 0..n-1} system
        for j in range(n):
          A = S[:,j]
          c = U[:,j]
          #clf = LassoCV(alpha=1., solver='cholesky', fit_intercept=True).fit(A, c)
          clf = LassoCV(fit_intercept=True, positive=False, max_iter=100000).fit(A, c)
          W[j] = clf.coef_
          C[j] = clf.intercept_

      else:
        raise NotImplementedError(f'Solver lasso does not handle the {self.scenario} scenario')

    elif(self.solver == 'identity'):

      if(self.d != self.n):
        raise NotImplementedError(f"Solver 'identity' requires d == n, but {self.d} != {self.n}")

      if(self.na > 1):
        raise NotImplementedError(f"Solver 'identity' requires na == 1, but na = {self.na}")

      W = np.eye(self.d)

    elif(self.solver == 'braids'):

      if(self.scenario in [ECO_DB_MULTICLASS, ECO_DB_MULTILABEL]):
        raise ValueError('For multiclass and multilabel datasets, use solver=lstsq instead.')

      elif(self.scenario in [ECO_DB_LABELRANK]):

        labelsets = []
        braid_level = defaultdict(dict)
        for i in range(m):
          labelset = tuple(Y[i])
          all_labels = [j for j in range(n)]
          if(labelset not in labelsets):
            for pos in range(n):
              label = labelset[pos]
              if(label >= 0):
                # the j-th position of the labelset represents a real label
                pair = (labelset, label)
                braid_level[pair] = n - pos - 1
                all_labels.remove(label)

            for missing_label in all_labels:
              # in label ranking with incomplete ranking, the labels that are
              # missing in a labelset must be assigned to the lowest segment of the
              # scale/strand/tope
              pair = (labelset, missing_label)
              braid_level[pair] = 0

            labelsets.append(labelset)

        for j in range(n):
          A = S[:,j]
          b = np.array([braid_level[(tuple(Y[i]),j)] for i in range(m)])
          x = np.linalg.lstsq(A, b, rcond=self.rcond)
          W[j] = x[0]

      else:
        raise NotImplementedError(f'Solver lstsq does not handle the {self.scenario} scenario')

    else:
      raise NotImplementedError(f'Solver {self.solver} not implemented in Polygrid')

    if(self.rcond is None):
      pass

    if(self.solver in ['lstsq', 'lstsqsym', 'lstsquni', 'braids']):
      (rank, svs) = (x[2], x[3])
      self.rnk = rank
      self.svs = svs
      if(self.rcond != ECO_RCOND):
        (r,c) = self.S[:,0].shape
        rank_chk = sum(svs >= max(svs) * self.rcond)
        print(f'   The feature matrix S is a ({r},{c})-matrix with rank {rank:3d}.')
        print(f'   There are {rank_chk:3d} singular values larger than rcond={self.rcond} x dominant sv = {self.rcond * max(svs)}.')
        #print(f'   {svs}')
    else:
      self.rnk = None
      self.svs = None

    return (W, C)

  def _collapse(self, S, W, C, m, n):
    YorU_hat = np.zeros((m,n))
    for j in range(n):
      YorU_hat[:,j] = np.dot(S[:,j,:], W[j]) + C[j]
    return YorU_hat

  def learn_thresholds(self, S, W, C, Y):

    metric = self.metric

    if(self.cutoff == ECO_CUTOFF_SINGLE):

      if(self.scenario == ECO_DB_MULTICLASS):
        tw = None

      elif(self.scenario in [ECO_DB_MULTILABEL, ECO_DB_LABELRANK]):
        (m,n) = Y.shape
        YorU_hat = self._collapse(S, W, C, m, n)

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
        tw = best * np.ones(self.n)

      else:
        raise ValueError(f'Threshold learning for {self.scenario} tasks is not implemented.')

    else: # in multiple thresholds mode

      if(self.scenario == ECO_DB_MULTICLASS):
        tw = None

      elif(self.scenario in [ECO_DB_MULTILABEL, ECO_DB_LABELRANK]):
        (m,n) = Y.shape
        YorU_hat = self._collapse(S, W, C, m, n)

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

          if(Y_real[:,j].size == (Y_real[:,j]==1).astype(int).sum()):
            # in the case in which there are only true positives
            # we do not want to define a threshold
            # -- this is equivalent to a ranking problem with complete rankings
            best = watermark
          else:
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
        tw = np.array(cutoffs)

      else:
        raise ValueError(f'Not implemented threshold learning for {self.scenario}')

    return tw

  def fit(self, X, Y, U=None):

    (m,d) = X.shape # recovers the number of instances and the number of features
    (_,n) = Y.shape # recovers the number of classes
    (self.m, self.d, self.n, self.ns) = (m, d, n, d * self.nspd)

    # initialises the prototypes and weights of each offer
    (self.P, self.T) = self.init_weights(X, Y, U)

    # partitions the closed unit disc into annular sectors
    # self.AS is a list of polygonal shapes approximating the annuli in the partitioning
    (self.AS, self.nas) = self.segment_disc(X, Y, U)

    # obtains the match polygons from each pair of demand and offer
    self.M = self.crossdo(X)

    # decomposes the match polygons into annular sectors
    self.S = self.decompose(self.M)
    if(self.norm != 'none'):
      self.S = self.normalise(self.S)

    # learns weights for each cell of the partitioning of the unit disc
    # as well as the scale thresholds
    (self.W, self.C) = self.learn_weights(self.S, Y, U)
    self.tw = self.learn_thresholds(self.S, self.W, self.C, Y)

    # updates the number of weights of the model
    self.nw = self.get_size(consider_thresholds=False)

    """
    The intermediate results derived directly from the training data could be discarded,
    as they are not model parameters. We decided to keep them to reduce the computational
    cost of the show_* methods, as well as simplifying theis signatures
    """
    #self.M = None
    #self.S = None

    return self

  def predict(self, X, raw=False, return_scores=False):

    (m, _) = X.shape # recovers the number of instances in the sample
    n = self.n

    safe_norms = ['none', 'm-n', 'm-nas', 'm']
    if(self.norm not in safe_norms or (self.norm == 'm-nas' and self.init=='full')):
      print('*****************************************************************************')
      print('* THERE ARE KNOWN ISSUES REGARDING THE USE OF PREDICT WITH NORMALISATION     *')
      print('* CONSIDER REDOING THIS WITHOUT NORMALISATION AND CHECK HOW RESULTS COMPARE *')
      print('*****************************************************************************')

    M = self.crossdo(X)
    S = self.decompose(M)
    YorU_hat = self._collapse(S, self.W, self.C, m, n)

    if(raw):
      Y_pred = YorU_hat
    else:
      if(self.scenario == ECO_DB_MULTICLASS):
        Y_pred = (YorU_hat == YorU_hat.max(axis=1)[:,None]).astype(int)

      elif(self.scenario == ECO_DB_MULTILABEL):
        Y_pred = np.zeros((m,n)).astype(int)
        for j in range(n):
          Y_pred[:,j] = (YorU_hat[:,j] >= self.tw[j]).astype(int)

      elif(self.scenario == ECO_DB_LABELRANK):
        Y_pred = (-YorU_hat).argsort()
        Y_pred = prune_rank(Y_pred, YorU_hat, self.tw).astype(int)

    if(return_scores):
      res = (Y_pred, YorU_hat)
    else:
      res = Y_pred

    return res

  def get_size(self, consider_thresholds=False):
    # counts the number of weights taken up by this Polygrid instance
    # -- the size of the actual parameter/weight matrices are computed
    # -- by default, threshold weights are NOT considered
    try:
      size = self.W.size
      # accounts for model intercepts
      if(self.solver in ['ridge', 'lasso']):
        size += self.C.size
      # accounts for model cutoffs
      if(consider_thresholds):
        if(self.scenario in [ECO_DB_MULTILABEL, ECO_DB_LABELRANK]):
          if(self.cutoff == ECO_CUTOFF_SINGLE):
            size += 1      # in single threshold mode
          else:
            size += self.n # in multiple thresholds mode
    except AttributeError:
      size = None
    return size

  def get_sizing_data(self):
    """
    PolygridCLI uses get_size() to create evaluation reports for the user, and uses
    get_sizing_data() to inform BaseCompetitor on the ideal average size limit of the
    competing model. Note that, differently from get_size(), get_sizing_data() does
    not account for cutoffs because all models take up the same number of cutoffs in a
    given learning/evaluation scenario.
    """
    try:
      size = self.get_size(consider_thresholds=False)
      res = (size, self.n, self.d)
    except AttributeError:
      res = None
    return res

  def calc_size_core(d, n, nspd, na, solver):
    # estimates the number of weights a Polygrid(nspd, na, solver) instance
    # would take up when fit to a (d, n) dataset
    # -- the size of the parameter/weight matrices are estimated
    # -- the returned size DOES NOT ACCOUNT FOR threshold weights
    ns = nspd * d
    size_w = (ns * na) * n
    if(solver in ['ridge', 'lasso']):
      size_c = n # adds an extra weight (intercept) for each offer
    else:
      size_c = 0
    return size_w + size_c

  def calc_size_full(scenario, d, n, nspd, na, solver, cutoff):
    # estimates the number of weights a Polygrid(nspd, na, solver, cutoff) instance
    # would take up when fit to a (d, n) dataset
    # -- the size of the parameter/weight matrices are estimated
    # -- the returned size ACCOUNTS FOR threshold weights
    size = self.calc_size_core(d, n, nspd, na, solver)
    if(scenario in [ECO_DB_MULTILABEL, ECO_DB_LABELRANK]):
      if(cutoff == ECO_CUTOFF_SINGLE):
        size += 1 # adds an extra weight to implement the shared cutoff
      elif(cutoff == ECO_CUTOFF_MULTIPLE):
        size += n # adds an extra weight (cutoff) for each scale
    return size

  def get_config_summary(self, include_model_name=False, details=[]):
    summary = []
    summary.append(f'dataset: {self.dataset_name}')
    summary.append(f'task: {self.scenario}')
    if(include_model_name):
      summary.append(f'model: {self.model_name}, with {self.get_size()} weights')
    else:
      summary.append(f'model size: {self.get_size()} weights')
    summary.append(f'vorder: {self.vorder}')
    summary.append(f'annuli: {self.na}, {self.annulus_type}')
    summary.append(f'sectors: {self.ns}, {self.sector_type}')
    summary.append(f'solver: {self.solver}')
    if(self.scenario in [ECO_DB_MULTILABEL, ECO_DB_LABELRANK]):
      summary.append(f'cutoff: {self.cutoff}')
    return(summary)

  def get_corder(self, offset=False):
    if(self.co is None):
      self.setup_corder()

    if(offset):
      co = [0] + [i+1 for i in self.co]
    else:
      co = self.co

    return CustomList(co)

  def set_symbol(self, value):
    self.sym = value

  def set_scenario(self, value):
    if(value in [ECO_DB_MULTICLASS, ECO_DB_MULTILABEL, ECO_DB_LABELRANK]):
      self.scenario = value
    else:
      raise NotImplementedError(f'Evaluation scenario {value} not implemented')

  def set_cutoff(self, value):
    if(value in [ECO_CUTOFF_SINGLE, ECO_CUTOFF_MULTIPLE]):
      self.cutoff = value
    else:
      raise NotImplementedError(f'Threshold mode {value} not implemented')

  def set_rcond(self, value):
    if(value is None):
      self.rcond = value
    else:
      value = float(value)
      if(0. <= value < 1. or value == -1.):
        self.rcond = value
      else:
        raise NotImplementedError(f'Cutoff for singular values not recognised: {value}')

  def set_metric(self, fn):
    self.metric = fn

  def set_avoid_FP(self, value):
    self.avoid_FP = value

  def set_dataset_name(self, value):
    self.dataset_name = value

  def set_feature_names(self, value):
    self.feature_names = value

  def set_target_names(self, value):
    self.target_names = value

  def set_nouns(self, sample_noun, class_noun):
    self.sample_noun = sample_noun
    self.class_noun  = class_noun

  def get_prototype(self, j):
    return self.P[j,:]

  def get_rcond_by_nsvs(self, nsvs):
    # returns the value for rcond so that as many as nsvs singular values
    # are used by np.linalg.lstsq
    if(self.svs is None):
      raise ValueError('Can only compute rcond if solver=lstsq, braids')
    if(not isinstance(nsvs, int) or nsvs < 1 or nsvs > self.svs.size):
      raise ValueError('Number of singular vectors must be between 1 and {self.nas - 1}, got {nsvs}')

    if(nsvs == self.svs.size):
      res = -1.
    else:
      res = (self.svs[nsvs-1] + self.svs[nsvs])/(2 * self.svs[0])

    return res

  #--------------------------------------------------------------------
  # methods that render diagrams
  #--------------------------------------------------------------------

  def draw_guides(self, axescolor='gainsboro', axeslw=1):

    # creates the guiding elements of the diagram - disc's outer boundary
    outerBoundary = Circle((0,0), self.RAD,
                    color='black',
                    fill=False,
                    linewidth = 3,
                    zorder=5)

    # creates the guiding elements of the diagram - domain axes
    lines  = []
    origin = (0.0, 0.0)
    vertices = self.scores2coords([self.RAD for _ in range(self.d)])
    lines  = [(origin, vertex) for vertex in vertices]
    domainAxes = LineCollection(lines,
                                colors = [axescolor for _ in lines],
                                linewidth = axeslw,
                                zorder=5)

    return outerBoundary, domainAxes

  def get_drawing_pattern(self, pattern):
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    lss = '-'
    if(self.sym == ECO_CAPACITY):

      if(pattern   == ECO_PTRN_DEMAND):
        (ec, fc, alpha, lws, layer) = ('#FF6600',    '#FF6600',      0.95,  3, 10)
      elif(pattern == ECO_PTRN_OFFER):
        (ec, fc, alpha, lws, layer) = ('#FF6600',    'none',         1.00,  4, 10)
      elif(pattern == ECO_PTRN_CROSS):
        (ec, fc, alpha, lws, layer) = ('#FF6600',    'none',         1.00,  4, 10)
      elif(pattern == ECO_PTRN_THIN1):
        (ec, fc, alpha, lws, layer) = ('#FF6600',    'none',         1.00, .5, 20)
      elif(pattern == ECO_PTRN_THIN2):
        (ec, fc, alpha, lws, layer) = ('black',      'none',         0.70,  1, 30)
        lss = '--'
      elif(pattern == ECO_PTRN_MATCH):
        (ec, fc, alpha, lws, layer) = ('none',       'lemonchiffon', 0.30,  1, 20)

    else:

      if(pattern   == ECO_PTRN_DEMAND):
        (ec, fc, alpha, lws, layer) = ('#FF6347',    '#FF6347',      0.95,  3, 10)
      elif(pattern == ECO_PTRN_OFFER):
        (ec, fc, alpha, lws, layer) = ('#FF6347',    'none',         1.00,  4, 10)
      elif(pattern == ECO_PTRN_CROSS):
        (ec, fc, alpha, lws, layer) = ('#FF6347',    'none',         1.00,  4, 10)
      elif(pattern == ECO_PTRN_THIN1):
        (ec, fc, alpha, lws, layer) = ('mediumblue', 'none',         1.00, .5, 20)
      elif(pattern == ECO_PTRN_THIN2):
        (ec, fc, alpha, lws, layer) = ('black',      'none',         0.70,  1, 30)
        lss = '--'
      elif(pattern == ECO_PTRN_MATCH):
        (ec, fc, alpha, lws, layer) = ('none',       'orchid',       0.20,  1, 20)

    return (ec, fc, alpha, lws, lss, layer)

  def draw_polygon(self, vertices, pattern):

    (ec, fc, alpha, lws, lss, layer) = self.get_drawing_pattern(pattern)
    # creates the n-gon specified by vertices, according to specified visual pattern
    pc = PolyCollection([vertices],
                        edgecolors = [ec for _ in vertices],
                        facecolors = [fc for _ in vertices],
                        alpha      = alpha,
                        linewidths = lws,
                        linestyles = lss,
                        zorder     = layer)

    return pc

  def add_domain_labels(self, feature_names, column2label, label_props):

    # unpacks layout and format configs
    font_dict = label_props['fontdict']
    offset = label_props['offset']
    alignment = label_props['alignment']

    # creates domain labels and tooltips
    labels   = []
    tooltips = []
    vertices = self.scores2coords([offset * self.RAD for _ in range(self.d)])
    for k in range(self.d):
      longLbl  = feature_names[k]
      shortLbl = column2label[longLbl]
      (x, y) = vertices[k]

      if(alignment == '-'):
        anchor = Text(x, y, shortLbl,
                      **font_dict)

      elif(alignment == '/'):
        angle  = k*2*np.pi/self.d - (np.pi if x < 0 else 0)
        anchor = Text(x, y, shortLbl,
                      rotation=np.rad2deg(angle),
                      **font_dict)

      elif(alignment == '|'):
        angle  = k*2*np.pi/self.d - np.pi/2 + (np.pi if y < 0 else 0)
        anchor = Text(x, y, shortLbl,
                      rotation=np.rad2deg(angle),
                      **font_dict)

      else:
        raise ValueError('Unexpected alignment mode for add_domain_labels.')

      labels.append(anchor)
      tooltips.append(longLbl)

    return (labels, tooltips)

  def get_colour_scheme(self, fig, layout, weights=None, show_weights=True, use_qualitative_scale=False):

    if(weights is None):
      weights = self.W

    # sets the appropriate colour map
    if(layout('cbar.cmap') == 'diverging'):
      cmap = plt.cm.coolwarm
    elif(layout('cbar.cmap') == 'uniform'):
      cmap = plt.cm.binary
    else:
      raise ValueError('Unexpected colour map in get_colour_scheme')

    # sets the appropriate colour norm to be used in the colour bar
    if(layout('cbar.norm') == 'TwoSlopeNorm'):
      wmin = weights.min()
      wmax = weights.max()
      if(wmin == wmax):
        wmin = wmin - 0.1
        wmax = wmax + 0.1
      wref = 0. if wmin < 0. < wmax else weights.mean()
      norm = colors.TwoSlopeNorm(vmin=wmin,
                                 vcenter=wref,
                                 vmax=wmax)

    elif(layout('cbar.norm') == 'SymLogNorm'):
      lnrwidth = 0.5
      gain = abs(weights.min() / weights.max())
      if(gain < 1):
        gain = 1/gain
      norm=colors.SymLogNorm(linthresh=lnrwidth, #xxxa can we improve here?
                             linscale=1,
                             vmin=-gain,
                             vmax=gain,
                             base=10)
    else:
      raise ValueError('Unexpected norm in get_colour_scheme')

    # creates a colour bar, as specified in layout
    oom = floor(np.log10(max(abs(weights.min()), abs(weights.max()))))
    #print()
    #print('Range of areas: ',   (self.S.min(),  self.S.max()))
    #print('Range of weights: ', (weights.min(), weights.max()))
    #print('Order of magnitude: ', oom)

    cax = plt.axes( [0.95, 0.05, 0.01, 0.80])
    sm  = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    if(use_qualitative_scale):

      cbar = fig.colorbar(sm, cax=cax,
                          ticks=[wmin, wmax],
                          alpha=layout('offer.region')['alpha'])
      cbar.ax.set_yticklabels(['Low', 'High'])
      cbar.ax.tick_params(**layout('cbar.ticks'))

    else:

      def fmt(x, pos):
          a, b = '{0:1.0e}'.format(x).split('e')
          a = int(a)
          b = int(b)
          return '{0: d}'.format(a) # DO NOT remove extra space after 0:

      ool = cax.text(0.75, 1.03, r'$10^{{{0}}}$'.format(oom),
                     transform=cax.transAxes,
                     **layout('cbar.factor'))
      cbar = fig.colorbar(sm, cax=cax,
                          format=ticker.FuncFormatter(fmt),
                          alpha=layout('offer.region')['alpha'])
      cbar.ax.tick_params(**layout('cbar.ticks'))
      tick_locator = ticker.MultipleLocator(base=10**(oom))
      cbar.locator = tick_locator
      cbar.update_ticks()

    # adds points to the colour bar to show where weights are inciding
    if(show_weights):
      x_max=cbar.ax.get_xlim()[1]
      ys = weights.flatten()
      xs = [x_max for _ in ys]
      cbar.ax.scatter(x=xs, y=ys, c='grey', s=50, marker='D', alpha=0.7)

    return (cmap, norm, cbar)

  def onclick_cross(self, event):
    # enables the tooltips during visualisation

    selection_ec = 'darkolivegreen'
    selection_lw = 2
    coords = (event.xdata, event.ydata)

    def update_with_titles():
      for (tipType, listener, tag, content) in self.titleTips:
        tag.set_text('{0}'.format(content))
      return None

    def update_with_weights(k, ground):

      # updates the offer charts
      for j in self.offerTips:
        (tipType, listener, tag, content) = self.offerTips[j][k]
        if(ground == 'foreground'):
          tag.set_text('{0}'.format(content))
          listener.set_edgecolor(selection_ec)
          listener.set_facecolor('none')
          listener.set_linewidth(selection_lw)
          listener.set_alpha(1.0)
          layer = listener.get_zorder()
          listener.set_zorder(layer+2)
          val = float(content)
          point = self.cbar.ax.scatter(x=0.0, y=val, c='k', s=30, marker='>')
          self.cbar_points.append(point)
          mark = listener.axes.scatter(x=coords[0], y=coords[1], c='k', s=35, marker='o', zorder=layer+2)
          self.chart_marks.append(mark)
        else:
          listener.set_edgecolor(self.bg_layer_offer.ec)
          listener.set_facecolor(self.bg_layer_offer.fc)
          listener.set_linewidth(self.bg_layer_offer.lw)
          listener.set_alpha(self.bg_layer_offer.alpha)
          listener.set_zorder(self.bg_layer_offer.zorder)
          for point in list(self.cbar_points): # iterates over a copy of points
            self.cbar_points.remove(point)
            point.remove()
          for mark in list(self.chart_marks):
            self.chart_marks.remove(mark)
            mark.remove()

      # updates the demand charts
      for i in self.demandTips:
        (tipType, listener, tag, content) = self.demandTips[i][k]
        if(ground == 'foreground'):
          tag.set_text('{0}'.format(content))
          listener.set_edgecolor(selection_ec)
          listener.set_facecolor('none')
          listener.set_linewidth(selection_lw)
          listener.set_alpha(1.0)
          layer = listener.get_zorder()
          listener.set_zorder(layer+2)
          mark = listener.axes.scatter(x=coords[0], y=coords[1], c='k', s=35, marker='o', zorder=layer+2)
          self.chart_marks.append(mark)
        else:
          listener.set_edgecolor(self.bg_layer_demand.ec)
          listener.set_facecolor(self.bg_layer_demand.fc)
          listener.set_linewidth(self.bg_layer_demand.lw)
          listener.set_alpha(self.bg_layer_demand.alpha)
          listener.set_zorder(self.bg_layer_demand.zorder)
          for mark in list(self.chart_marks):
            self.chart_marks.remove(mark)
            mark.remove()

      # updates the matching charts
      for entry in self.crossTips:
        (tipType, listener, tag, content) = self.crossTips[entry][k]
        if(ground == 'foreground'):
          tag.set_text('{0}'.format(content))
          listener.set_edgecolor(selection_ec)
          listener.set_facecolor('none')
          listener.set_linewidth(selection_lw)
          listener.set_alpha(1.0)
          layer = listener.get_zorder()
          listener.set_zorder(layer + 2)
          mark = listener.axes.scatter(x=coords[0], y=coords[1], c='k', s=35, marker='o', zorder=layer+2)
          self.chart_marks.append(mark)
        else:
          listener.set_edgecolor(self.bg_layer_cross.ec)
          listener.set_facecolor(self.bg_layer_cross.fc)
          listener.set_linewidth(self.bg_layer_cross.lw)
          listener.set_alpha(self.bg_layer_cross.alpha)
          listener.set_zorder(self.bg_layer_cross.zorder)
          for mark in list(self.chart_marks):
            self.chart_marks.remove(mark)
            mark.remove()

      return None

    # process event 'user clicked on offer/demand title'
    for (tipType, listener, tag, content) in self.titleTips:
      (contains, ind) = listener.contains(event)
      if(contains):
        update_with_titles()
        update_with_weights(self.last_k, 'background')
        self.fig.canvas.draw_idle()
        break

    # process event 'user clicked on dimension label'
    if(not contains):
      for (tipType, listener, tag, content) in self.labelTips:
        (contains, ind) = listener.contains(event)
        if(contains):
          tag.set_text("{0}".format(content))
          update_with_weights(self.last_k, 'background')
          self.fig.canvas.draw_idle()
          break

    # process event 'user clicked on an offer partition'
    if(not contains):
      for j in self.offerTips:
        if(contains):
          break
        for (k, (tipType, listener, tag, content)) in enumerate(self.offerTips[j]):
          (contains, ind) = listener.contains(event)
          if(contains):
            update_with_weights(self.last_k, 'background')
            update_with_weights(k,           'foreground')
            self.last_k = k
            self.fig.canvas.draw_idle()
            break

    # process event 'user clicked on a demand segment'
    if(not contains):
      for i in self.demandTips:
        if(contains):
          break
        for (k, (tipType, listener, tag, content)) in enumerate(self.demandTips[i]):
          (contains, ind) = listener.contains(event)
          if(contains):
            update_with_weights(self.last_k, 'background')
            update_with_weights(k,           'foreground')
            self.last_k = k
            self.fig.canvas.draw_idle()
            break

    # process event 'user clicked on a cross partition'
    if(not contains):
      for entry in self.crossTips:
        if(contains):
          break
        for(k, (tipType, listener, tag, content)) in enumerate(self.crossTips[entry]):
          (contains, ind) = listener.contains(event)
          if(contains):
            update_with_weights(self.last_k, 'background')
            update_with_weights(k,           'foreground')
            self.last_k = k
            self.fig.canvas.draw_idle()
            break

  def show_demands(self, X, idxs, perf_params, show_params, show_idxs=None):

    # unpacks the parameters
    (gridCols, transpose, column2label, layout_data, hide_tags, filename) = show_params
    feature_names = [self.feature_names[vo] for vo in self.vo]
    if(show_idxs is None):
      show_idxs = idxs
    instance_ids = [f'{self.sample_noun.title()} {i}' for i in show_idxs]
    (m, d) = X.shape
    (nrows, ncols) = (ceil(m/gridCols), min(m, gridCols))

    # recovers data to organise the layout of the diagram
    layout = LayoutManager(layout_data)
    (sizew, sizeh, adjusts) = layout('demand.figure_sizes', nrows=nrows, ncols=ncols)

    # initialises the plot grid
    fig, axes = plt.subplots(nrows, ncols, figsize=(sizew, sizeh))

    plt.subplots_adjust(**adjusts)
    # NOTE: this diagram does not contain a colour bar

    # initialises the variables that are used to handle interaction with diagram
    pass

    # renders the required diagram on each subplot
    for i in range(nrows*ncols):

      plt.subplot(nrows, ncols, i + 1)

      ax = plt.gca()
      ax.axis('off') # DO NOT INVERT: first axis off, then autoscale
      ax.autoscale()
      ax.set_box_aspect(1)

      if(i+1 > m):
        # shows an empty subplot in case m is not multiple of gridCols)
        pass

      else:
        # plots a demand chart in the current subplot

        # 1. draws the demand tag (an annotate on top/right of subplot)
        pass

        # 2. draws the title (top/center of subplot)
        font_properties = layout('demand.title_1st')
        anchor = plt.title(f'{instance_ids[i]}', **font_properties)

        # 3. draws the annular sectors (plotted first so as they appear as background)
        pass

        # 4. draws the guiding components (domain axes, outter boundary) and
        #    the demand polygon
        (outerBoundary, domainAxes) = self.draw_guides()
        dpc = self.draw_polygon(self.scores2coords(X[i,self.vo]), ECO_PTRN_DEMAND)

        ax.add_patch(outerBoundary)
        ax.add_collection(domainAxes)
        ax.add_collection(dpc)

        if(i == 0):
          label_props = layout('dimension.label')
          (domain_lbls, lbl_tips) = self.add_domain_labels(feature_names, column2label, label_props)
          for (l, anchor) in enumerate(domain_lbls):
            anchor.transform=ax.transData
            ax._add_text(anchor)

    # enables the tooltips during visualisation
    pass

    ## annotations for the last figure in Appendix B of thesis
    #origin = (0.0, 0.0)
    #coords = self.scores2coords(X[0,self.vo])
    #lmbda  = X[0,self.vo] / max(X[0,self.vo])
    #gamma  = 2*np.pi/self.d
    #beta0  = np.arcsin( (lmbda[1] * np.sin(gamma)) /
    #                    np.sqrt(lmbda[0]**2 + lmbda[1]**2 - 2 * lmbda[0] * lmbda[1] * np.cos(gamma))
    #                  )
    #beta1  = np.arcsin( (lmbda[2] * np.sin(gamma)) /
    #                    np.sqrt(lmbda[1]**2 + lmbda[2]**2 - 2 * lmbda[1] * lmbda[2] * np.cos(gamma))
    #                  )
    #alpha0 = np.pi - gamma - beta0
    #
    #x_0    = coords[0]
    #x_1    = coords[1]
    #x_2    = coords[2]
    #
    ## -- plus 5 lines
    #lines  = [(origin, x_0), (x_0, x_1), (x_1, origin),
    #          (origin, x_2), (x_2, x_1), ]
    #ovl_lines = LineCollection(lines, colors='k', ls='--', lw=1.0, zorder=10)
    #ax.add_collection(ovl_lines)
    #
    ## plus 4 arcs
    #arc_args = dict(height=0.2, width=0.2, edgecolor='red', linewidth=2, zorder=10)
    #
    #gamma_ = gamma*(180/np.pi)
    #gamma_arc = Arc(xy=origin, angle=0., theta1=0., theta2=gamma_, **arc_args)
    #ax.add_patch(gamma_arc)
    #
    #beta0_ = beta0*(180/np.pi)
    #beta0_arc = Arc(xy=x_0, angle=180-beta0_, theta1=0., theta2=beta0_, **arc_args)
    #ax.add_patch(beta0_arc)
    #
    #beta1_ = beta1*(180/np.pi)
    #beta1_arc = Arc(xy=x_1, angle=(180-beta1_)+gamma_, theta1=0., theta2=beta1_, **arc_args)
    #ax.add_patch(beta1_arc)
    #
    #alpha0_ = alpha0*(180/np.pi)
    #alpha0_arc = Arc(xy=x_1, angle=180+gamma_, theta1=0., theta2=alpha0_, **arc_args)
    #ax.add_patch(alpha0_arc)
    #
    ## -- plus 6 text labels
    #font_dict = {'family': 'arial',
    #             'color':  'black',
    #             'weight': 'normal',
    #             'size':   12,
    #             'zorder': 10,
    #           }
    #
    #plt.scatter(x_0[0], x_0[1], s=30, c='black', marker='o', zorder=10)
    #anchor = Text(x_0[0]+0.04, x_0[1]-0.08, r'$x_{i0}$', **font_dict)
    #anchor.transform=ax.transData
    #ax._add_text(anchor)
    #
    #plt.scatter(x_1[0], x_1[1], s=30, c='black', marker='o', zorder=10)
    #anchor = Text(x_1[0]+0.04, x_1[1]-0.035, r'$x_{i1}$', **font_dict)
    #anchor.transform=ax.transData
    #ax._add_text(anchor)
    #
    #anchor = Text(origin[0]+0.15, origin[1]+0.03, r'$\gamma$', **font_dict)
    #anchor.transform=ax.transData
    #ax._add_text(anchor)
    #
    #anchor = Text(x_0[0]-0.20, x_0[1]+0.08, r'$\beta_0$', **font_dict)
    #anchor.transform=ax.transData
    #ax._add_text(anchor)
    #
    #anchor = Text(x_1[0]-0.14, x_1[1]-0.19, r'$\alpha_0$', **font_dict)
    #anchor.transform=ax.transData
    #ax._add_text(anchor)
    #
    #anchor = Text(x_1[0]-0.26, x_1[1]-0.028, r'$\beta_1$', **font_dict)
    #anchor.transform=ax.transData
    #ax._add_text(anchor)
    #
    #plt.tight_layout()

    # shows or saves the figure
    if(filename is None):
      fig.tight_layout()
      plt.show()
    else:
      plt.savefig(filename, dpi=ECO_DPI)
    (fw, fh) = (fig.get_figwidth(), fig.get_figheight())

    plt.close(fig)
    print(f'-- gridspec has {nrows} rows {ncols} colums')
    print(f'-- figure width is {fw} and height is {fh}')

    return None

  def show_clusters(self, X, Y, perf_params, show_params):

    # unpacks the parameters
    (gridCols, transpose, #feature_names_, target_names,
     column2label, layout_data, hide_tags, filename) = show_params
    feature_names=[self.feature_names[vo] for vo in self.vo]
    (m, d) = X.shape
    (_, n) = Y.shape #self.n
    (nrows, ncols) = (ceil(n/gridCols), min(n, gridCols))

    if(perf_params is None):
      (label_counts, tr_idxs, te_idxs) = ({0: m}, range(m), [])
    else:
      (label_counts, tr_idxs, te_idxs) = perf_params

    # recovers data to organise the layout of the diagram
    layout = LayoutManager(layout_data)
    (sizew, sizeh, adjusts) = layout('offer.figure_sizes', nrows=nrows, ncols=ncols)

    # initialises the plot grid
    fig, axes = plt.subplots(nrows, ncols, figsize=(sizew, sizeh))
    plt.subplots_adjust(**adjusts)
    (cmap, norm, cbar) = self.get_colour_scheme(fig, layout)
    cbar.remove()

    # initialises the variables that are used to handle interaction with diagram
    pass

    # renders the required diagram on each subplot
    co = self.get_corder()
    for j in range(nrows*ncols):

      plt.subplot(nrows, ncols, co.index(j) + 1)
      ax = plt.gca()
      ax.axis('off') # DO NOT INVERT: first axis off, then autoscale
      ax.autoscale()
      ax.set_box_aspect(1)

      if(j+1 > n):
        # ensures an empty subplot is shown in case n is not a multiple of gridCols
        pass

      else:

        # plots a clusters diagram in the current subplot

        # 1. draws the cluster tag (an annotate on top/right of subplot)
        annot = None
        if(filename is None and label_counts is not None):
          annot = ax.annotate('{0}'.format(label_counts[j]),
                              #zorder=0,
                              **layout('offer.tag'))

        # 2. draws the title (top/center of this cluster subplot)
        font_properties = layout('offer.title_1st') if co.index(j) == 0 else layout('offer.title')
        anchor = plt.title(f'{column2label[self.target_names[j]]}', **font_properties)

        # 3. draws the annular sectors (plotted first so as they appear as background)
        weights = self.W
        for (k, geom) in enumerate(self.AS):
          (xs, ys) = geom.exterior.xy
          #anchor = ax.fill(xs, ys, ec='lightgrey', fc='none', lw=1)
          anchor = ax.fill(xs, ys, fc=cmap(norm(weights[j,k])), **layout('offer.region'))

        # 4. draws the guiding components (domain axes, outter boundary) and
        #    the demand polygon
        (outerBoundary, domainAxes) = self.draw_guides()
        ax.add_patch(outerBoundary)
        ax.add_collection(domainAxes)

        if(co.index(j) == 0):
          label_props = layout('dimension.label')
          (domain_lbls, lbl_tips) = self.add_domain_labels(feature_names, column2label, label_props)
          for (l, anchor) in enumerate(domain_lbls):
            anchor.transform=ax.transData
            ax._add_text(anchor)

        # first draws the polygons referring to training instances ...
        for i in range(m):
          if(Y[i, j] == 1):
            if(i in tr_idxs):
              dpc = self.draw_polygon(self.scores2coords(X[i,self.vo]), ECO_PTRN_THIN1)
              ax.add_collection(dpc)

        # ... then the test instances (so the latter appear more clearly)
        for i in range(m):
          if(Y[i, j] == 1):
            if(i in te_idxs):
              dpc = self.draw_polygon(self.scores2coords(X[i,self.vo]), ECO_PTRN_THIN2)
              ax.add_collection(dpc)

    # enables the tooltips during visualisation
    pass

    # shows or saves the figure
    if(filename is None):
      fig.tight_layout()
      plt.show()
    else:
      plt.savefig(filename, dpi=ECO_DPI)
    (fw, fh) = (fig.get_figwidth(), fig.get_figheight())

    plt.close(fig)
    print(f'-- gridspec has {nrows} rows {ncols} colums')
    print(f'-- figure width is {fw} and height is {fh}')

    return None

  def show_offers(self, perf_params, show_params, weights=None, show_protos=True):

    # unpacks the parameters
    (gridCols, transpose, column2label, layout_data, hide_tags, filename) = show_params
    feature_names=[self.feature_names[vo] for vo in self.vo]
    interactive = (filename is None)
    if(weights is None):
      weights = self.W
    (n, d) = (self.n, self.d)
    (nrows, ncols) = (ceil(n/gridCols), min(n, gridCols))

    # recovers data to organise the layout of the diagram
    layout = LayoutManager(layout_data)
    (sizew, sizeh, adjusts) = layout('offer.figure_sizes', nrows=nrows, ncols=ncols)

    # initialises the plot grid and sets its colour bar
    fig, axes = plt.subplots(nrows, ncols, figsize=(sizew, sizeh))
    plt.subplots_adjust(**adjusts)
    (cmap, norm, cbar) = self.get_colour_scheme(fig, layout, weights)

    # initialises the variables that are used to handle interaction with diagram
    self.reset_plot(fig, cbar)

    # renders the required diagram on each subplot
    co = self.get_corder()
    for j in range(nrows*ncols):

      plt.subplot(nrows, ncols, co.index(j) + 1)
      ax = plt.gca()
      ax.axis('off') # DO NOT INVERT: first axis off, then autoscale
      ax.autoscale()
      ax.set_box_aspect(1)

      if(j+1 > n):
        # ensures an empty subplot is shown in case n is not a multiple of gridCols
        pass

      else:
        # plots an offer chart in the current subplot

        # 1. draws the annotation (top/right of this offer subplot)
        #annot = ax.annotate(label_counts[j], **layout('offer.tag'))
        show_dummy_content = (hide_tags or self.scenario == ECO_DB_MULTICLASS)
        content = '-' if show_dummy_content else '{0:5.3f}'.format(self.tw[j])
        tag = ax.annotate(content, **layout('offer.tag')) if interactive else None
        listener = tag
        self.titleTips.append(('title', listener, tag, content))

        # 2. draws the title (top/center of this offer subplot)
        font_properties = layout('offer.title_1st') if co.index(j) == 0 else layout('offer.title')
        listener = plt.title(f'{column2label[self.target_names[j]]}', **font_properties)
        self.titleTips.append(('title', listener, tag, content))

        # 3. draws the annular sectors (plotted first so as they appear as background)
        (ec, fc, alpha, lws, lss, layer) = self.get_drawing_pattern(ECO_PTRN_OFFER)
        bg_layer = ListenerData(ec='none', fc='none', lw=0.0, alpha=0.0, zorder=layer-1)
        self.bg_layer_offer = bg_layer
        for (k, geom) in enumerate(self.AS):

          # draws the annular sector using a transparent pattern, and stores
          # the weight of each annular sector
          # (this creates the listeners, for later use in interaction)
          (xs, ys) = geom.exterior.xy
          weight = weights[j,k] #self.W[j,k]
          listener = ax.fill(xs, ys, ec     = bg_layer.ec,
                                     fc     = bg_layer.fc,
                                     alpha  = bg_layer.alpha,
                                     lw     = bg_layer.lw,
                                     zorder = bg_layer.zorder)
          content = '{0:8.5f}'.format(weight)
          self.offerTips[j].append(('segment', listener[0], tag, content))

          # draws the annular sectors with the weight mapped to color scale
          #dummy = ax.fill(xs, ys, fc=cmap(norm(self.W[j,k])), **layout('offer.region'))
          dummy = ax.fill(xs, ys, fc=cmap(norm(weights[j,k])), **layout('offer.region'))

          # plots the centroid of the annular sector
          # (used when exporting data to MATLAB Nevanlinna-Pick interpolation)
          #(xc, yc) = geom.centroid.xy
          #ax.plot(xc[0], yc[0], marker = 'o', markersize=10, color='k')

        # 4. draws the guiding components (domain axes, outter boundary) and
        #    the average levels of features for this class
        (outerBoundary, domainAxes) = self.draw_guides()
        opc = self.draw_polygon(self.scores2coords(self.P[j]), ECO_PTRN_OFFER)
        ax.add_patch(outerBoundary)
        ax.add_collection(domainAxes)
        if(show_protos):
          ax.add_collection(opc)
        if(co.index(j) == 0):
          label_props = layout('dimension.label')
          (domain_lbls, lbl_tips) = self.add_domain_labels(feature_names, column2label, label_props)
          for (l, listener) in enumerate(domain_lbls):
            listener.transform=ax.transData
            ax._add_text(listener)
            self.labelTips.append(('domain labels', listener, tag, lbl_tips[l]))

    # shows or saves the figure
    if(interactive):
      cid = fig.canvas.mpl_connect("button_press_event", self.onclick_cross)
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

  def show_mass(self, Y, idxs, perf_params, show_params):
    (label_counts, subcmd) = perf_params
    weights = np.zeros((self.n, self.nas))
    if(subcmd == 'mass'):
      for j in range(self.n):
        aux_idxs = np.where(Y[idxs, j] == 1)[0]
        weights[j] = self.S[aux_idxs, j].sum(axis=0)
    elif(subcmd == 'prob'):
      for j in range(self.n):
        aux_idxs = np.where(Y[idxs, j] == 1)[0]
        normaliser = self.S[aux_idxs, j].sum()
        weights[j] = self.S[aux_idxs, j].sum(axis=0)/normaliser # each row sums to 1.
    self.show_offers(label_counts, show_params, weights=weights)
    return None

  def show_cross(self, X, idxs, perf_params, show_params, show_idxs=None, show_protos=True):

    # unpacks the parameters
    (gridCols, transpose, column2label, layout_data, hide_tags, filename) = show_params
    feature_names = [self.feature_names[vo] for vo in self.vo]
    if(show_idxs is None):
      show_idxs = idxs
    instance_ids = [f'{self.sample_noun.title()} {i}' for i in show_idxs]
    interactive = (filename is None)
    (m, d) = X.shape
    n = self.n
    (nrows, ncols) = (m+1, n+1)

    # the onclick event handler needs to know how many demands are being displayed
    m_old  = self.m
    self.m = m

    # recovers data to organise the layout of the diagram
    layout = LayoutManager(layout_data)
    (sizew, sizeh, adjusts) = layout('cross.figure_sizes', nrows=nrows, ncols=ncols)

    # initialises the plot grid and sets its colour scheme
    fig, axes = plt.subplots(nrows, ncols, figsize=(sizew, sizeh))
    plt.subplots_adjust(**adjusts)
    (cmap, norm, cbar) = self.get_colour_scheme(fig, layout)

    # initialises the variables that are used to handle interaction with diagram
    self.reset_plot(fig, cbar)

    # renders the required diagram on each subplot
    co = self.get_corder(offset=True)
    for ii in range(m+1):
      for jj in range(n+1):
        if(transpose):
          plt.subplot(ncols, nrows, co.index(jj) * nrows + ii + 1)
        else:
          plt.subplot(nrows, ncols, ii * ncols + co.index(jj) + 1)
        ax = plt.gca()
        ax.axis('off') # DO NOT INVERT: first axis off, then autoscale
        ax.autoscale()
        ax.set_box_aspect(1)

        #-----------------------------------------------------------------------------
        # summary "post it" (top/left annotation)
        #-----------------------------------------------------------------------------
        if(ii == 0 and jj == 0):
          # shows an empty subplot in the leftmost/top position of the grid
          ax.set_xlim(0.0, 1.0)
          ax.set_ylim(0.0, 1.0)
          annot = ax.annotate('\n'.join(self.get_config_summary()), multialignment='left', **layout('summary.tag'))

        #-----------------------------------------------------------------------------
        # offer charts
        #-----------------------------------------------------------------------------
        elif(ii == 0 and jj > 0): # offer chart
          # plots an offer chart in the current subplot
          j = jj - 1

          # 1. draws the offer tag (with the cutoff value)
          hide_offer_tag = (hide_tags and filename is not None)
          show_dummy_content = (hide_tags or self.scenario == ECO_DB_MULTICLASS)
          content = '-' if show_dummy_content else '{0:5.3f}'.format(self.tw[j])
          tag = None if hide_offer_tag else ax.annotate(content, **layout('offer.tag'))
          listener = tag
          self.titleTips.append(('title', listener, tag, content))

          # 2. draws the title (top/center of this offer subplot)
          if(transpose):
            font_properties = layout('demand.title_1st' if j == 0 else 'demand.title')
          else:
            font_properties = layout('offer.title_1st')
          listener = plt.title(f'{column2label[self.target_names[j]]}', **font_properties)
          self.titleTips.append(('title', listener, tag, content))

          # 3. draws the annular sectors
          (ec, fc, alpha, lws, lss, layer) = self.get_drawing_pattern(ECO_PTRN_OFFER)
          bg_layer = ListenerData(ec='none', fc='none', lw=0.0, alpha=0.0, zorder=layer-1)
          self.bg_layer_offer = bg_layer
          for (k, geom) in enumerate(self.AS):

            # draws the annular sector using a transparent pattern, and stores
            # the weight of each annular sector
            # (this creates the listeners, for later use in interaction)
            (xs, ys) = geom.exterior.xy
            weight = self.W[j,k]
            listener = ax.fill(xs, ys, ec     = bg_layer.ec,
                                       fc     = bg_layer.fc,
                                       alpha  = bg_layer.alpha,
                                       lw     = bg_layer.lw,
                                       zorder = bg_layer.zorder)
            content = '{0:8.5f}'.format(weight)
            self.offerTips[j].append(('segment', listener[0], tag, content))

            # draws the annular sectors with the weight mapped to color scale
            dummy = ax.fill(xs, ys, fc=cmap(norm(self.W[j,k])), **layout('offer.region'))

          # 4. draws the guiding components (domain axes, outter boundary) and
          #    the average feature levels of this offer
          (outerBoundary, domainAxes) = self.draw_guides()
          opc = self.draw_polygon(self.scores2coords(self.P[j]), ECO_PTRN_OFFER)
          ax.add_patch(outerBoundary)
          ax.add_collection(domainAxes)
          if(show_protos):
            ax.add_collection(opc)
          if(co.index(jj) == 1):
            label_props = layout('dimension.label')
            (domain_lbls, lbl_tips) = self.add_domain_labels(feature_names, column2label, label_props)
            for (l, listener) in enumerate(domain_lbls):
              listener.transform=ax.transData
              ax._add_text(listener)
              self.labelTips.append(('domain labels', listener, tag, lbl_tips[l]))

        #-----------------------------------------------------------------------------
        # demand charts
        #-----------------------------------------------------------------------------
        elif(ii > 0 and jj == 0 and self.init == 'full'):
          # plots a demand chart in the current subplot (when not using cutters)
          i = ii - 1

          # 0. prepares supporting data
          demand = self.coords2poly(self.scores2coords(X[i,self.vo]))
          M = self.crossdo(X)
          S = self.decompose(M)[:,0,:] #when using cutters, this would not work
          area_score = S[i].sum()

          # 1. draws the demand tag (with the area score)
          hide_demand_tag = (hide_tags and filename is not None)
          content = '-' if hide_tags else '{0:5.3f}'.format(area_score)
          tag = None if hide_demand_tag else ax.annotate(content, **layout('demand.tag'))
          listener = tag
          self.titleTips.append(('title', listener, tag, content))

          # 2. draws the title (top/center of subplot)
          if(transpose):
            font_properties = layout('offer.title_1st')
          else:
            font_properties = layout('demand.title_1st' if i == 0 else 'demand.title')
          listener = plt.title(f'{instance_ids[i]}', **font_properties)
          self.titleTips.append(('title', listener, tag, content))

          # 3. draws the annular sectors and the segments of the demand polygon
          (ec, fc, alpha, lws, lss, layer) = self.get_drawing_pattern(ECO_PTRN_DEMAND)
          bg_layer = ListenerData(ec='none', fc='none', lw=0.0, alpha=0.0, zorder=layer-1)
          self.bg_layer_demand = bg_layer
          for (k, ansec) in enumerate(self.AS):

            # draws the annular sector using a transparent pattern, and stores
            # the area of the demand polygon that it intersects
            # (this creates the listeners, for later use in interaction)
            (xs, ys) = ansec.exterior.xy
            measure = ansec.intersection(demand).area
            listener = ax.fill(xs, ys, ec     = bg_layer.ec,
                                       fc     = bg_layer.fc,
                                       alpha  = bg_layer.alpha,
                                       lw     = bg_layer.lw,
                                       zorder = bg_layer.zorder)
            content = '{0:8.5f}'.format(measure)
            self.demandTips[i].append(('segment', listener[0], tag, content))

            # draws the segments of the demand polygon
            # NOTE: the intersection of an annular sector with a demand polygon
            #       may result in a MultiPolygon or Point object.
            #       In the first case, each component Polygon must be drawn individually
            #       In the latter case, the geometry is ignored
            segments = ansec.intersection(demand)
            if(type(segments) is Polygon or type(segments) is Point):
              list_of_polygons = [segments]
            else:
              list_of_polygons = segments.geoms
            for geom in list_of_polygons:
              if(type(geom) is not Point):
                (xs, ys) = geom.exterior.xy
                dummy = ax.fill(xs, ys, ec=ec, fc=fc, alpha=alpha, lw=0.0, zorder=layer)
              else:
                pass

          # 4. draws the guiding components (domain axes, outter boundary)
          (outerBoundary, domainAxes) = self.draw_guides()
          ax.add_patch(outerBoundary)
          ax.add_collection(domainAxes)
          if(i == 0):
            label_props = layout('dimension.label')
            (domain_lbls, lbl_tips) = self.add_domain_labels(feature_names, column2label, label_props)
            for (l, listener) in enumerate(domain_lbls):
              listener.transform=ax.transData
              ax._add_text(listener)
              self.labelTips.append(('domain labels', listener, tag, lbl_tips[l]))

        elif(ii > 0 and jj == 0 and self.init != 'full'):
          # plots a demand chart in the current subplot
          i = ii - 1

          # 1. draws the annotation (top/right of subplot)
          pass

          # 2. draws the title (top/center of subplot)
          font_properties = layout('demand.title_1st' if i == 0 else 'demand.title')
          plt.title(f'{instance_ids[i]}', **font_properties)

          # 3. draws the annular sectors (plotted first so as they appear as background)
          pass

          # 4. draws the guiding components (domain axes, outter boundary) and
          #    the demand polygon
          (outerBoundary, domainAxes) = self.draw_guides()
          dpc = self.draw_polygon(self.scores2coords(X[i,self.vo]), ECO_PTRN_DEMAND)
          ax.add_patch(outerBoundary)
          ax.add_collection(domainAxes)
          ax.add_collection(dpc)
          if(i == 0):
            label_props = layout('dimension.label')
            (domain_lbls, lbl_tips) = self.add_domain_labels(feature_names, column2label, label_props)
            for (l, anchor) in enumerate(domain_lbls):
              anchor.transform=ax.transData
              ax._add_text(anchor)

        #-----------------------------------------------------------------------------
        # matching charts
        #-----------------------------------------------------------------------------
        else:
          # plots a cross diagram in the current subplot
          i = ii - 1
          j = jj - 1

          # 0. prepares supporting data
          demand = self.coords2poly(self.scores2coords(X[i,self.vo]))
          offer  = self.coords2poly(self.scores2coords(self.T[j]))
          match  = offer.intersection(demand)
          Y_pred = self.predict(X, raw=True)

          # 1. draws the matching tag (with the inner product)
          hide_cross_tag = (hide_tags and filename is not None)
          content = '-' if hide_tags else '{0:5.3f}'.format(Y_pred[i,j])
          tag = None if hide_cross_tag else ax.annotate(content, **layout('cross.tag'))
          listener = tag
          self.titleTips.append(('title', listener, tag, content))

          if(not hide_tags and tag is not None):
            if(self.scenario == ECO_DB_MULTICLASS):
              if(Y_pred[i,j] == max(Y_pred[i,:])):
                tag.set_backgroundcolor('lightgreen')
            elif(self.scenario in [ECO_DB_MULTILABEL, ECO_DB_LABELRANK]):
              # In MLC tasks, raw scores in Y_pred are \hat{Y}
              # -- the case is positive for C_j if \hat{Y}_{ij} > cutoff_j
              # In LR  tasks, raw scores in Y_pred are \hat{U}
              # -- the case is positive for C_j if \hat{U}_{ij} > cutoff_j
              # -- also note that presence/absence do not encode order
              # -- for this, the user must consult at the score
              if(Y_pred[i,j] >= self.tw[j]):
                tag.set_backgroundcolor('lightgreen')
            else:
              pass

          # 2. draws the title (in this case, the tag plays the role of the title)
          pass

          # 3. draws the annular sectors and the segments of the cross polygon
          (ec, fc, alpha, lws, lss, layer) = self.get_drawing_pattern(ECO_PTRN_CROSS)
          bg_layer = ListenerData(ec='none', fc=cmap(norm(0.0)), lw=0.0, alpha=0.25, zorder=layer-2)
          self.bg_layer_cross = bg_layer
          for (k, ansec) in enumerate(self.AS):

            # draws the annular sector using a transparent pattern, and stores
            # the weighted area of the demand polygon that it intersects
            # (this creates the listeners, for later use in interaction)
            (xs, ys) = ansec.exterior.xy
            weighted_area = ansec.intersection(demand).area * self.W[j,k]
            listener = ax.fill(xs, ys, ec     = bg_layer.ec,
                                       fc     = bg_layer.fc,
                                       alpha  = bg_layer.alpha,
                                       lw     = bg_layer.lw,
                                       zorder = bg_layer.zorder)
            content = '{0:8.5f}'.format(weighted_area)
            self.crossTips[(i,j)].append(('segment', listener[0], tag, content))

            # draws the segments of the cross polygon
            # NOTE: the intersection of an annular sector with a demand polygon
            #       may result in a MultiPolygon or Point object.
            #       In the first case, each component Polygon must be drawn individually
            #       In the latter case, the geometry is ignored
            segments = ansec.intersection(match)
            if(type(segments) is Polygon or type(segments) is Point):
              list_of_polygons = [segments]
            else:
              list_of_polygons = segments.geoms
            for geom in list_of_polygons:
              if(type(geom) is not Point):
                (xs, ys) = geom.exterior.xy
                _fc = cmap(norm(self.W[j,k]))
                dummy = ax.fill(xs, ys, ec=ec, fc=_fc, alpha=alpha, lw=0.0, zorder=layer-1)
              else:
                pass

          # 4a. draws the guiding components (domain axes, outter boundary)
          (outerBoundary, domainAxes) = self.draw_guides()
          ax.add_patch(outerBoundary)
          ax.add_collection(domainAxes)

          # 4b. draws the match between demand and offer
          #vertices = self.poly2coords(match)
          vertices = self.poly2coords(demand)
          cpc = self.draw_polygon(vertices, ECO_PTRN_CROSS)
          ax.add_collection(cpc)

          # 5. adds tags with intercept data
          if(ii == m):
            if(self.solver in ['ridge', 'lasso']):
              annot = ax.annotate(f'{self.C[j]:5.3f}', **layout('intercept.tag'))

    # shows or saves the figure
    if(interactive):
      cid = fig.canvas.mpl_connect("button_press_event", self.onclick_cross)
      plt.show()
      fig.canvas.mpl_disconnect(cid)
    else:
      plt.savefig(filename, dpi=self.DPI)
      print(f'-- sent to file {filename}')

    (fw, fh) = (fig.get_figwidth(), fig.get_figheight())
    print(f'-- gridspec has {nrows} rows {ncols} colums')
    print(f'-- figure width is {fw} and height is {fh}')
    plt.close(fig)

    self.m = m_old

    return None

  def show_bar_cross(self, X, idxs, perf_params, show_params, show_idxs=None, scaling=None):

    # unpacks the parameters
    (gridCols, transpose, column2label, layout_data, hide_tags, filename) = show_params
    feature_names=[self.feature_names[vo] for vo in self.vo]
    if(show_idxs is None):
      show_idxs = idxs
    instance_ids = [f'{self.sample_noun.title()} {i}' for i in show_idxs]
    (m, d) = X.shape
    n = self.n
    (nrows, ncols) = (m+1, n+1)
    if(scaling is None):
      scaling=np.array([1.0 for _ in range(self.d)])

    # recovers data to organise the layout of the diagram
    layout = LayoutManager(layout_data)
    (sizew, sizeh, adjusts) = layout('cross.figure_sizes', nrows=nrows, ncols=ncols)

    # initialises the plot grid and sets its colour scheme
    fig, axes = plt.subplots(nrows, ncols, figsize=(sizew, sizeh))
    plt.subplots_adjust(**adjusts)
    (cmap, norm, cbar) = self.get_colour_scheme(fig, layout)

    # initialises the variables that are used to handle interaction with diagram
    self.reset_plot(fig, cbar)

    # renders the required diagram on each subplot
    co = self.get_corder(offset=True)
    for ii in range(m+1):
      for jj in range(n+1):
        if(transpose):
          plt.subplot(ncols, nrows, co.index(jj) * nrows + ii + 1)
        else:
          plt.subplot(nrows, ncols, ii * ncols + co.index(jj) + 1)
        ax = plt.gca()
        ax.axis('off') # DO NOT INVERT: first axis off, then autoscale
        ax.autoscale()
        ax.set_box_aspect(1)

        if(ii == 0 and jj == 0):
          # shows an empty subplot in the leftmost/top position of the grid
          pass

        elif(ii == 0 and jj > 0):

          #--------------------------------------------------------------------------
          # plots an offer chart in the current subplot
          #--------------------------------------------------------------------------
          j = jj - 1

          # 1. draws the annotation (top/right of this offer subplot)
          annot = None

          # 2. draws the title (top/center of this offer subplot)
          font_properties = layout('offer.title_1st') # if j == 0 else 'offer.title')
          anchor = plt.title(f'{column2label[self.target_names[j]]}', **font_properties)

          # 3. draws the annular sectors (plotted first so as they appear as background)
          pass

          # 4. draws the guiding components (domain axes, outter boundary) and
          label_props = layout('dimension.label')
          (domain_lbls, lbl_tips) = self.add_domain_labels(feature_names, column2label, label_props)
          domain_lbls = [e.get_text() for e in domain_lbls]
          lenghts = self.P[j] * scaling
          bar_colors = [cmap(norm(self.W[j,k])) for k in range(self.nas)]
          #bar_colors = ['sandybrown'] * self.d
          #bar_colors = ['crimson'] * self.d

          ax.axis(True)
          ax.set_xlim(0.0, ceil(scaling.max()))
          ax.set_facecolor('whitesmoke')
          ax.grid(alpha=0.3, color='silver', lw=1, ls='-')

          y_pos = np.arange(self.d)
          # https://matplotlib.org/3.3.4/api/_as_gen/matplotlib.axes.Axes.barh.html?highlight=barh#matplotlib.axes.Axes.barh
          # also https://public.tableau.com/app/profile/nicole.mark/vizzes
          ax.barh(y=y_pos,  width=lenghts, align='center', color=bar_colors)
          ax.set_yticks(y_pos)
          ax.invert_yaxis()  # labels read top-to-bottom

          if(j == 0):
            ax.set_yticklabels(domain_lbls, fontsize=13)
          else:
            ax.set_yticklabels([])


        elif(ii > 0 and jj == 0):

          #--------------------------------------------------------------------------
          # plots a demand chart in the current subplot
          #--------------------------------------------------------------------------
          i = ii - 1

          # 1. draws the annotation (top/right of subplot)
          annot = None

          # 2. draws the title (top/center of subplot)
          font_properties = layout('demand.title_1st' if i == 0 else 'demand.title')
          anchor = plt.title('{0}'.format(instance_ids[i]), **font_properties)

          # 3. draws the annular sectors (plotted first so as they appear as background)
          pass

          # 4. draws the guiding components (domain axes, outter boundary) and
          #    the demand polygon
          label_props = layout('dimension.label')
          (domain_lbls, lbl_tips) = self.add_domain_labels(feature_names, column2label, label_props)
          domain_lbls = [e.get_text() for e in domain_lbls]
          lenghts = X[i,self.vo] * scaling[self.vo]
          bar_colors = ['darkorange'] * self.d

          ax.axis(True)
          ax.set_xlim(0.0, ceil(scaling.max()))
          #ax.set_facecolor('whitesmoke')
          ax.grid(alpha=0.3, color='silver', lw=1, ls='-')
          ax.set_xlabel('(cm)')

          y_pos = np.arange(self.d)
          ax.barh(y=y_pos, width=lenghts, align='center', color=bar_colors)
          ax.set_yticks(y_pos)

          ax.invert_yaxis()  # labels read top-to-bottom
          if(i == 0):
            #ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_yticklabels(domain_lbls, fontsize=13) #label_props['fontdict']['size'])
          else:
            ax.get_yaxis().set_visible(False)

          # enables the tooltips during visualisation
          pass

        else:

          #--------------------------------------------------------------------------
          # plots a cross diagram in the current subplot
          #--------------------------------------------------------------------------
          i = ii - 1
          j = jj - 1

          demand = self.coords2poly(self.scores2coords(X[i,self.vo]))
          offer  = self.coords2poly(self.scores2coords(self.T[j]))
          match  = offer.intersection(demand)

          # 1. draws the annotation (top/right of subplot)
          Y_pred = self.predict(X, raw=True)
          if(filename is None):
            if(hide_tags):
              annot = ax.annotate('-', **layout('bars.tag'))
            else:
              annot = ax.annotate('{0:5.3f}'.format(Y_pred[i,j]),
                                  **layout('bars.tag'))
          else:
            if(hide_tags):
              annot = None
            else:
              annot = ax.annotate('{0:5.3f}'.format(Y_pred[i,j]),
                                  **layout('bars.tag'))

          if(not hide_tags and annot is not None):
            if(self.scenario == ECO_DB_MULTICLASS):
              if(Y_pred[i,j] == max(Y_pred[i,:])):
                annot.set_backgroundcolor('lightgreen')
            elif(self.scenario in [ECO_DB_MULTILABEL, ECO_DB_LABELRANK]):
              # In MLC tasks, raw scores in Y_pred are \hat{Y}
              # -- the case is positive for C_j if \hat{Y}_{ij} > cutoff_j
              # In LR  tasks, raw scores in Y_pred are \hat{U}
              # -- the case is positive for C_j if \hat{U}_{ij} > cutoff_j
              # -- also note that presence/absence do not encode order
              # -- for this, the user must consult at the score
              if(Y_pred[i,j] >= self.tw[j]):
                annot.set_backgroundcolor('lightgreen')
            else:
              pass

          # 2. draws the title (top/center of subplot)
          pass

          # 3. draws the annular sectors (plotted first so as they appear as background)
          pass

          # 4a. draws the guiding components (domain axes, outter boundary)
          label_props = layout('dimension.label')
          (domain_lbls, lbl_tips) = self.add_domain_labels(feature_names, column2label, label_props)
          domain_lbls = [e.get_text() for e in domain_lbls]
          lenghts = X[i,self.vo] * scaling[self.vo]
          bar_colors = [cmap(norm(self.W[j,k])) for k in range(self.nas)]
          #bar_colors = ['crimson' if lenghts[i] > 0 else 'cornflowerblue' for i in range(self.d)]
          #bar_colors = ['darkorange' if lenghts[i] > 0 else 'crimson' for i in range(self.d)]

          ax.axis(True)
          ax.set_xlim(0, ceil(scaling.max()))
          ax.set_facecolor('whitesmoke')
          ax.grid(alpha=0.3, color='silver', lw=1, ls='-')
          ax.set_xlabel('(cm)')

          y_pos = np.arange(self.d)
          ax.barh(y=y_pos, width=lenghts, align='center', color=bar_colors, alpha=1.0)
          ax.set_yticks(y_pos)
          ax.invert_yaxis()  # labels read top-to-bottom
          ax.set_yticklabels([])

    # shows or saves the figure
    plt.show() if filename is None else plt.savefig(filename, dpi=self.DPI)
    (fw, fh) = (fig.get_figwidth(), fig.get_figheight())

    plt.close(fig)
    print(f'-- gridspec has {nrows} rows {ncols} colums')
    print(f'-- figure width is {fw} and height is {fh}')

    return None

  def show_disc(self, show_params, domain_labels=True, cell_labels=True):

    # unpacks the parameters
    (gridCols, transpose, column2label, layout_data, hide_tags, filename) = show_params
    feature_names=[self.feature_names[vo] for vo in self.vo]
    d = self.d
    (nrows, ncols) = (1, 1)

    # recovers data to organise the layout of the diagram
    layout = LayoutManager(layout_data)
    (sizew, sizeh, adjusts) = layout('disc.figure_sizes',
                                      nrows=nrows,
                                      ncols=ncols)

    # initialises the plot grid
    fig, axes = plt.subplots(nrows, ncols, figsize=(sizew, sizeh))
    plt.subplots_adjust(**adjusts)
    (cmap, norm, cbar) = self.get_colour_scheme(fig, layout)
    cbar.remove()

    for j in range(nrows*ncols):

      plt.subplot(nrows, ncols, j + 1)
      ax = plt.gca()
      ax.axis('off') # do not invert: first axis off, then autoscale
      ax.autoscale()
      ax.set_box_aspect(1)

      # plots an offer chart in the current subplot

      # 1. draws the annotation (top/right of subplot)
      pass

      # 2. draws the title (top/center of subplot)
      pass

      # 3. draws the annular sectors (plotted first so as they appear as background)
      na = self.na
      ns = self.ns
      AS = self.AS
      w = np.array([10000 * (-1)**si * (-1)**ai for ai in range(na) for si in range(ns)])

      for (k, geom) in enumerate(AS):
        (xs, ys) = geom.exterior.xy
        anchor = ax.fill(xs, ys, fc=cmap(norm(w[k])), **layout('disc.region'))

        if(cell_labels):

          fontdict = layout('disc.label')['fontdict']
          fontdict['color'] = cmap(norm(w[k+1])) if (k+1)%self.ns > 0 else cmap(norm(w[k-1]))
          fontdict['weight'] = 'bold'

          (p,q) = divmod(k, self.ns)
          _r = self.radii[p] + (self.radii[p+1] - self.radii[p])/2
          _den = (1 if self.sector_type == 'cover' else 2)
          _theta = self.angles[q] + self.angles[1]/_den

          (xx, yy) = self.polar2coord(_r, _theta)
          #ax.scatter(xx, yy, c='red', marker='+', s=100, zorder=100)
          annot = ax.annotate(r'$\omega_{{{0}}}$'.format(k),
                              xy = (xx, yy),
                              annotation_clip = False,
                              zorder = 20,
                              **fontdict)
          #annot = None

      # 4. draws the guiding components (domain axes, outter boundary) and
      #    the offer polygon
      (outerBoundary, domainAxes) = self.draw_guides()
      opc = self.draw_polygon(self.scores2coords(self.T[j]), ECO_PTRN_OFFER)
      ax.add_patch(outerBoundary)
      #ax.add_collection(domainAxes)
      #ax.add_collection(opc)
      if(j == 0 and domain_labels):
        label_props = layout('dimension.label')
        #label_props['fontdict']['size']=28 #20
        (domain_lbls, lbl_tips) = self.add_domain_labels(feature_names, column2label, label_props)
        for (l, anchor) in enumerate(domain_lbls):
          anchor.transform=ax.transData
          ax._add_text(anchor)

    # presents or saves the figure
    plt.show() if filename is None else plt.savefig(filename, dpi=ECO_DPI)
    (fw, fh) = (fig.get_figwidth(), fig.get_figheight())
    plt.close(fig)
    print(f'-- gridspec has {nrows} rows {ncols} colums')
    print(f'-- figure width is {fw} and height is {fh}')

    return None

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
    #if(self.inspection is None):
    #  self.inspection = self.inspect(X, Y, idxs, tr_idxs)
    #(Y_pred, YorU_hat, names, coords, cutoffs, yticklabels) = self.inspection
    (Y_pred, YorU_hat, names, coords, cutoffs, yticklabels) = self.inspect(X, Y, idxs, tr_idxs)

    (xmin, xmax) = (YorU_hat.min(), YorU_hat.max())
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
