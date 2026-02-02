import os
import re
import gettext
import numpy as np

from copy             import copy
from collections      import defaultdict, Counter
from numpy.random     import shuffle

from sklearn          import datasets as uci_datasets
from palmerpenguins   import load_penguins as _load_penguins
from sklearn.utils    import Bunch
from sklearn.cluster  import AgglomerativeClustering
from sklearn.metrics  import precision_score, recall_score, confusion_matrix
from skfuzzy.cluster  import cmeans

from skmultilearn.model_selection import IterativeStratification

from customDefs       import getMountedOn, loadAsText, deserialise
from customDefs       import ECO_SEED
from customDefs       import ECO_DEFICIT, ECO_CAPACITY

ECO_DB_UNLABELLED = 'unlabelled'
ECO_DB_MULTICLASS = 'multiclass'
ECO_DB_MULTILABEL = 'multilabel'
ECO_DB_LABELRANK  = 'label ranking'

ECO_ASSIGNMENT_O  = 'original'
ECO_ASSIGNMENT_H  = 'hierarchical'
ECO_ASSIGNMENT_F  = 'fuzzy'
ECO_ASSIGNMENT_R  = 'ranking'

ECO_SPLIT_MVS = 'mvs' # mean vector splitting strategy
ECO_SPLIT_STV = 'stv' # Sechidis, Tsoumakas, and Vlahavas (2011), Szymanski & Kajdanowicz (2017)

ECO_PRESENCE  = 'presence-encoded targets'
ECO_RANKING   = 'ranking-encoded targets'

# sets up the user message translation services
# (to use it inside a function, redefine _ = local_gettext)
lang = None
try:
  lang = os.environ['ECO_LANGUAGE']
  translator_domain = os.path.basename(__file__).split('.')[0]
  translator = gettext.translation(translator_domain, localedir='locale', languages=[lang])
  translator.install()
  local_gettext = _
except (KeyError, FileNotFoundError):
  local_gettext = lambda e: e

#-------------------------------------------------------------------------------------------------------------------------------------------
# Problem-specific definitions - interface to handle research datasets
# NOTE: the order of the columns is preserved (Polygrid may impose a different order)
#-------------------------------------------------------------------------------------------------------------------------------------------
def load_dataset(sourcepath, datasetName):

  if(datasetName == 'iris'):
    # loads the UCI Iris dataset
    dataset = uci_datasets.load_iris() # returns a Bunch object
    dataset.name  = datasetName
    dataset.atype = ECO_DB_MULTICLASS  # indicates this is a multiclass dataset
    dataset.sample_noun = 'specimen'   # indicates the noun that describes a sample
    dataset.class_noun  = 'species'    # indicates the noun that describes a class

  elif(datasetName == 'iris@pb'):
    # loads the Iris dataset downloaded from the Paderborn repository
    dataset = paderborn_load(sourcepath, datasetName)
    dataset.name  = datasetName
    dataset.atype = ECO_DB_LABELRANK   # indicates this is a label ranking dataset
    dataset.sample_noun = 'specimen'
    dataset.class_noun  = 'species'

  elif(datasetName == 'wine'):
    # loads the UCI Wine dataset
    dataset = uci_datasets.load_wine()
    dataset.name  = datasetName
    dataset.atype = ECO_DB_MULTICLASS
    dataset.sample_noun = 'sample'
    dataset.class_noun  = 'cultivar'

  elif(datasetName == 'wine@pb'):
    # loads the Paderborn Wine dataset
    dataset = paderborn_load(sourcepath, datasetName)
    dataset.name  = datasetName
    dataset.atype = ECO_DB_LABELRANK
    dataset.sample_noun = 'sample'
    dataset.class_noun  = 'cultivar'

  elif(datasetName == 'vowel@pb'):
    # loads the Paderborn Wine dataset
    dataset = paderborn_load(sourcepath, datasetName)
    dataset.name  = datasetName
    dataset.atype = ECO_DB_LABELRANK
    dataset.sample_noun = 'sample'
    dataset.class_noun  = 'vowel'

  elif(datasetName == 'cancer'):
    # loads the UCI Cancer dataset
    dataset = uci_datasets.load_breast_cancer()
    dataset.name  = datasetName
    dataset.atype = ECO_DB_MULTICLASS
    dataset.sample_noun = 'subject'
    dataset.class_noun  = 'diagnosis'

    # keeps only the features that are related to mean measurements
    # and discards error and extremes
    (m,n) = dataset.data.shape
    cols_filter  = [j for j in range(n) if dataset.feature_names[j].startswith('mean')]
    dataset.data = dataset.data[:,cols_filter]
    dataset.feature_names = [dataset.feature_names[j] for j in cols_filter]

  elif(datasetName == 'digits'):
    # loads (a small stratified sample of) the UCI Digits dataset
    dataset = uci_datasets.load_digits()
    dataset.name  = datasetName
    dataset.atype = ECO_DB_MULTICLASS
    dataset.sample_noun = 'sample'
    dataset.class_noun  = 'numeral'

    # selects just a small sample of the original dataset
    L = defaultdict(list)
    for (i, j) in enumerate(dataset.target):
      L[j].append(i)

    n = len(dataset.target_names)
    spc  = 25 # desired number of samples per class
    idxs = []
    for j in range(n):
      idxs += L[j][0:spc]

    dataset.data   = dataset.data[idxs,:]
    dataset.target = dataset.target[idxs]

    # changes the types of the target names to string
    dataset.target_names = [str(e) for e in dataset.target_names]

  elif(datasetName == 'penguins'):
    # loads the Palmer penguins dataset
    dataset = palmer_load_penguins()
    dataset.name  = datasetName
    dataset.atype = ECO_DB_MULTICLASS
    dataset.sample_noun = 'specimen'
    dataset.class_noun  = 'species'

  elif(datasetName in ['whoqol', 'whoqol-ml-11', 'whoqol-ml-22',
                                 'whoqol-lr-11', 'whoqol-lr-22', ]):

    atype = {'whoqol':       ECO_DB_MULTICLASS,
             'whoqol-ml-11': ECO_DB_MULTILABEL,
             'whoqol-ml-22': ECO_DB_MULTILABEL,
             'whoqol-lr-11': ECO_DB_LABELRANK,
             'whoqol-lr-22': ECO_DB_LABELRANK,
              }

    # loads any of the UFSCar WHOQOL datasets
    dataset = load_whoqol(sourcepath, datasetName)
    dataset.name  = datasetName
    dataset.atype = atype[datasetName]
    dataset.sample_noun = 'participant'
    dataset.class_noun  = 'class'

  elif(datasetName in ['ampiab', 'ampiab-ml-11', 'ampiab-ml-22',
                                 'ampiab-lr-11', 'ampiab-lr-22', ]):

    atype = {'ampiab':       ECO_DB_MULTICLASS,
             'ampiab-ml-11': ECO_DB_MULTILABEL,
             'ampiab-ml-22': ECO_DB_MULTILABEL,
             'ampiab-lr-11': ECO_DB_LABELRANK,
             'ampiab-lr-22': ECO_DB_LABELRANK,
              }

    # loads any of the EACH/PMSP AMPIAB datasets
    dataset = load_ampiab(sourcepath, datasetName)
    dataset.name  = datasetName
    dataset.atype = atype[datasetName]
    dataset.sample_noun = 'patient'
    dataset.class_noun  = 'referrals' if datasetName == 'ampiab-ml-11' else 'class'

  elif(datasetName in ['elsio1', 'elsio1-ml-11', 'elsio1-ml-22',
                                 'elsio1-lr-11', 'elsio1-lr-22', ]):

    atype = {'elsio1':       ECO_DB_MULTICLASS,
             'elsio1-ml-11': ECO_DB_MULTILABEL,
             'elsio1-ml-22': ECO_DB_MULTILABEL,
             'elsio1-lr-11': ECO_DB_LABELRANK,
             'elsio1-lr-22': ECO_DB_LABELRANK,
              }

    # loads the EACH/PMSP ELSIO1 dataset
    dataset = load_elsio1(sourcepath, datasetName)
    dataset.name  = datasetName
    dataset.atype = atype[datasetName]
    dataset.sample_noun = 'participant'
    dataset.class_noun  = 'class'

  elif(datasetName == 'foodtruck'):
    # loads the foodtruck dataset
    # (downloaded from the Cometa repository, preprocessed with readfoodtruck.py script)
    dataset = load_foodtruck(sourcepath, datasetName)
    dataset.name  = datasetName
    dataset.atype = ECO_DB_MULTILABEL
    dataset.sample_noun = 'participant'
    dataset.class_noun  = 'cuisine'

  elif(datasetName == 'water'):
    # loads the water quality dataset
    # (downloaded from the KDIS repository, preprocessed with readwater.py script)
    dataset = load_water(sourcepath, datasetName)
    dataset.name  = datasetName
    dataset.atype = ECO_DB_MULTILABEL
    dataset.sample_noun = 'sample'
    dataset.class_noun  = 'taxa'

  else:
    raise ValueError('Invalid dataset name')

  #---------------------------------------------------------------------------------
  # ensures the recovered data satisfy Polygrid requirements
  #---------------------------------------------------------------------------------

  # 1. identifies features that take on negative values
  # -- standard procedure: ignore such features
  # -- need to incorporate a negative-valued feature?
  #    -- investigate its measurement process:
  #       -- is it obtained by counting of evidence for presence?
  #          e.g., how many eggs are there in that egg box?
  #          e.g., how many eggs are there in that nest?
  #       -- is it obtained by counting of evidence for absence?
  #          e.g., how many eggs are missing in that egg box?
  #          e.g., how many eggs are missing in that nest?
  #    -- what about the remaining features? they encode presence or absence?
  #    -- recode the feature so it matches the way the remaining features are encoded

  (m,d) = dataset.data.shape
  ignoreCols = []                   # list of columns that will be ignored
  cn = dataset.data.min(axis=0) < 0 # identify columns with negative values
  if(cn.sum() > 0):
    aux = np.where(cn == True)[0].tolist()
    ignoreCols += aux
    ignoreFeatures = [dataset.feature_names[i] for i in aux]
    print('-- dataset contains negative-valued feature(s), which will be ignored: {0}'.format(ignoreFeatures))

  # 2. identifies non-discriminative features (constant column)
  cc = np.all(dataset.data == dataset.data[0,:], axis=0) # constant columns
  if(cc.sum() > 0):
    aux = np.where(cc == True)[0].tolist()
    ignoreCols += aux
    ignoreFeatures = [dataset.feature_names[i] for i in aux]
    print('-- dataset contains constant feature(s), which will be ignored: {0}'.format(ignoreFeatures))

  # 3. removes features that failed tests 1 or 2
  P = copy(dataset.data)
  keepCols = [i for i in range(d) if i not in ignoreCols]
  P = P[:, keepCols]
  feature_names = [dataset.feature_names[i] for i in range(d) if i not in ignoreCols]
  P_bs = copy(P) # makes a copy of P before changing any remaining values

  # 4. replaces zero-measurements with small positive constant
  #    -- zero values are replaced by a fraction of the smallest value larger than zero
  #       major implication: match(p,t) > 0 for all p, t
  #       assuming t obtained by full or averages initialisation
  zero_replacement = [e for e in sorted(set(P.flatten())) if e > 0][0]/5
  P[np.where(P == 0)] = zero_replacement

  # rescales features to the unit interval (projection to the unit positive hypercube)
  scaling = P.max(axis=0)
  P = P / scaling

  # sets the attributes of the dataset (a Bunch object)
  dataset.name  = datasetName
  dataset.odata = P_bs # not the original data:
                       # some features may have been excluded or modified
  dataset.data  = P
  dataset.scaling = scaling
  dataset.onclasses = len(dataset.target_names) # original number of classes
  dataset.feature_names = feature_names
  dataset.assignments = {}   #  synthetic assignements will be stored here ...
  dataset.memberships = {}   # and levels of membership to classes in here

  dataset.xdata    = None # used to store data during filtering by factors
  dataset.xodata   = None # used to store data during filtering by factors
  dataset.xtarget  = None # used to store data during filtering by factors
  dataset.xcaseIDs = None # used to store data during filtering by factors

  if(not hasattr(dataset, 'symbol')):
    # indicates that the underlying construct measures a capacity
    # (instead of deficit)
    dataset.symbol = ECO_CAPACITY

  if(not hasattr(dataset, 'caseIDs')):
    (m,d) = dataset.data.shape
    dataset.caseIDs = ['P{0}'.format(i) for i in range(m)]

  return dataset

def get_labels(dataset):

  # selects the module-specific translation services
  _ = local_gettext

  datasetName = dataset.name

  if(  datasetName in ['iris', 'iris@pb']):
    column2label = {'sepal length (cm)': _('sepal-l'),
                    'sepal width (cm)':  _('sepal-w'),
                    'petal length (cm)': _('petal-l'),
                    'petal width (cm)':  _('petal-w'),

                    'sepallength':       _('sepal-l'),
                    'sepalwidth':        _('sepal-w'),
                    'petallength':       _('petal-l'),
                    'petalwidth':        _('petal-w'),

                    'Iris-setosa':       'Setosa',
                    'Iris-versicolor':   'Versicolor',
                    'Iris-virginica':    'Virginica',
                   }

    offset = 1.15
    alignment = '|'

  elif(  datasetName in ['penguins']):
    column2label = {'bill_length_mm':    _('bill-L'),
                    'bill_depth_mm':     _('bill-D'),
                    'flipper_length_mm': _('flipper-L'),
                    'body_mass_g':       _('weight'),
                   }

    offset = 1.15
    alignment = '|'

  elif(datasetName in ['wine', 'wine@pb']): #xxx missing translations
    column2label = {'alcohol':                        'alc',
                    'malic_acid':                     'ma',
                    'ash':                            'ash',
                    'alcalinity_of_ash':              'aash',
                    'magnesium':                      'Mg',
                    'total_phenols':                  'TPC',
                    'flavanoids':                     'TFV',
                    'nonflavanoid_phenols':           'NFP',
                    'proanthocyanins':                'PAC',
                    'color_intensity':                'color',
                    'hue':                            'hue',
                    'od280/od315_of_diluted_wines':   'OD',
                    'proline':                        'Pro',

                    'Alcohol':                        'alc',
                    'Malic_acid':                     'ma',
                    'Ash':                            'ash',
                    'Alcalinity_of_ash':              'aash',
                    'Magnesium':                      'Mg',
                    'Total_phenols':                  'TPC',
                    'Flavanoids':                     'TFV',
                    'Nonflavanoid_phenols':           'NFP',
                    'Proanthocyanins':                'PAC',
                    'Color_intensity':                'color',
                    'Hue':                            'hue',
                    'OD280%2FOD315_of_diluted_wines': 'OD',
                    'Proline':                        'Pro',

                    'class_0':                        'Wine 0',
                    'class_1':                        'Wine 1',
                    'class_2':                        'Wine 2',

                    '1':                              'Wine 0',
                    '2':                              'Wine 1',
                    '3':                              'Wine 2',

                    }

    offset = 1.1
    alignment = '|'

  elif(datasetName == 'cancer'): #xxx missing translations
    column2label = {'mean radius':            'rad',
                    'mean texture':           'tex',
                    'mean perimeter':         'per',
                    'mean area':              'area',
                    'mean smoothness':        'smo',
                    'mean compactness':       'comp',
                    'mean concavity':         'conc',
                    'mean concave points':    'cp',
                    'mean symmetry':          'sym',
                    'mean fractal dimension': 'frac',
                   }

    offset = 1.1
    alignment = '|'

  elif(datasetName == 'digits'):
    column2label = {}
    for name in dataset.feature_names:
      column2label[name] = name
    #for name in dataset.target_names:
    #  column2label[name] = name

    offset = 1.1
    alignment = '|'

  elif(datasetName == 'vowel@pb'):
    column2label = {}
    for name in dataset.feature_names:
      column2label[name] = name
    #for name in dataset.target_names:
    #  column2label[name] = name

    offset = 1.1
    alignment = '|'

  elif(datasetName in ['whoqol', 'whoqol-ml-11', 'whoqol-ml-22',
                                 'whoqol-lr-11', 'whoqol-lr-22',]):
    #xxx missing translations for WHOQOL datasets
    column2label = {'physical health':      'physical',
                    'psychological':        'psychol.',
                    'social relationships': 'social',
                    'environment':          'environ.',
                   }

    offset = 1.1
    alignment = '|'

  elif(datasetName in ['ampiab', 'ampiab-ml-11', 'ampiab-ml-22',
                                 'ampiab-lr-11', 'ampiab-lr-22',]):
    column2label = {
                    'déficit em saúde bucal':               'oral health',
                    'déficit em ABVDs':                     'ADL',
                    'déficit cognitivo':                    'cognitive',
                    'déficit em AIVDs':                     'IADL',
                    'morbidades':                           'morbidity',

                    'Frágil: maior ou igual a 11 pontos':   'Frail',
                    'Pré-frágil: 6 a 10 pontos':            'Pre-Frail',
                    'Saudável: 0 a 5 pontos':               'Healthy',

                    'Equipe Saúde Bucal':                   'Oral Health',
                    'NASF':                                 'NASF',
                    # NASF  Núcleo de Apoio à Saúde da Família
                    'Outros':                               'Other(I)',
                    'AVALIAÇÃO MÉDICA ESPECIALIZADA':       'Specialty',
                    'EXAMES COMPLEMENTARES':                'Exams+',
                    'Centro Especializado de reabilitação': 'CER',
                    # -- https://www.gov.br/saude/pt-br/assuntos/novo-pac-saude/centros-especializados-em-reabilitacao
                    # O CER ... pode ser organizado conforme o número de modalidades de reabilitação (auditiva, física, intelectual e visual).
                    'URSI':                                 'URSI',
                    'AME Idoso':                            'AME',
                    'CAPS/CRAS/CREAS/SASF':                 'Social',
                    # -- from https://mds.gov.br/webarquivos/publicacao/brasil_sem_miseria/livro_o_brasil_sem_miseria/siglas.pdf
                    # CAPS  Centros de Atenção Psicossocial
                    # CRAS  Centro de Referência de Assistência Social
                    # CREAS Centro de Referência Especializado de Assistência Social
                    # --
                    # SASF  Serviço de Assistência Social à Família
                    'OUTROS':                               'Other(E)',
                    'NCI':                                  'NCI',
                   }

    offset = 1.1
    alignment = '|'

  elif(datasetName in ['elsio1', 'elsio1-ml-11', 'elsio1-ml-22',
                                 'elsio1-lr-11', 'elsio1-lr-22',]): #xxx missing translations
    column2label = {'cognitive':     'cognitive',
                    'locomotor':     'locomotor',
                    'psychological': 'psych',
                    'sensory':       'sensory',
                    'vitality':      'vitality',
                   }

    offset = 1.1
    alignment = '|'

  elif(datasetName == 'foodtruck'): #missing translations
    column2label = {
                    # features
                    'taste':           'taste',
                    'hygiene':         'hyg',
                    'menu':            'menu',
                    'presentation':    'pres',
                    'attendance':      'att',
                    'ingredients':     'ingr',
                    'place.to.sit':    'sits',
                    'takeaway':        'take',
                    'variation':       'var',
                    'stop.strucks':    'stop',
                    'schedule':        'schd',

                    # factors
                    'frequency':       'freq',
                    'time':            'time',
                    'expenses':        'exps',
                    'motivation':      'motiv',
                    'gender':          'gender',
                    'marital.status':  'marital',
                    'age.group':       'age-gr',
                    'scholarity':      'educ',
                    'average.income':  'inc-gr',
                    'has.work':        'empl/t',

                    # outcomes
                    'street_food':     'street',
                    'gourmet':         'gourmet',
                    'italian_food':    'italian',
                    'brazilian_food':  'brazilian',
                    'mexican_food':    'mexican',
                    'chinese_food':    'chinese',
                    'japanese_food':   'japanese',
                    'arabic_food':     'arabic',
                    'snacks':          'snacks'  ,
                    'healthy_food':    'healthy',
                    'fitness_food':    'fitness',
                    'sweets_desserts': 'desserts'

                   }

    offset = 1.1
    alignment = '|'

  elif(datasetName == 'water'): #xxx missing translations
    column2label = {
                    # features
                    'std_temp':  'T',
                    'std_pH':    'pH',
                    'conduct':   'cond',
                    'o2':       r'$O_2$',
                    'o2sat':    r'$O_2*$',
                    'co2':      r'$CO_2$',
                    'hardness':  'hard',
                    'no2':      r'$NO_2$',
                    'no3':      r'$NO_3$',
                    'nh4':      r'$NH_4$',
                    'po4':      r'$PO_4$',
                    'cl':        'Cl',
                    'sio2':     r'$SiO_2$',
                    'kmno4':    r'$KMnO_4$',
                    'k2cr2o7':  r'$K_2Cr_2O_7$',
                    'bod':       'BOD',

                    # factors

                    # outcomes

                   }

    offset = 1.15
    alignment = '/'

  else:
    raise ValueError('Invalid dataset name')

  # must keep this loop even when you add the target names explicitly
  # because the original labels may be discarded after reassignment
  # (e.g., assign h 3), and this block will ensure the new labels are
  # properly treated
  for target_name in dataset.target_names:
    if(target_name not in column2label):
      column2label[target_name] = target_name

  return (column2label, offset, alignment)

def assign_groups(dataset, ngroups, params = None, fuzzfactor = 1.0):

  # unpacks parameters and recovers cutoff levels
  (procedure, savePlot, targetpath, max_labels, seed) = params
  (m, d)  = dataset.data.shape

  needs_post_processing = True
  if(procedure[0] == ECO_ASSIGNMENT_O):

    if(dataset.atype == ECO_DB_MULTICLASS):
      # uses the original multiclass targets as assignments
      assignments = [ [label] for label in dataset.target]
      memberships = [ [1.0]   for label in dataset.target]
      ngroups = len(dataset.target_names)

    elif(dataset.atype == ECO_DB_MULTILABEL):
      # uses the original multilabel targets as assignments
      Y = dataset.target
      U = dataset.target / dataset.target.sum(axis=1)[:, None]

      needs_post_processing = False

    elif(dataset.atype == ECO_DB_LABELRANK):
      # uses the original label rankings as assignments
      # in this case, the membership matrix U is built according to this policy:
      # -- given two consecutive labels in a ranking, \ell_i \succ \ell_{i+1},
      #    the degree of membership u(\ell_i) = 2 u(\ell_{i+1}).
      # this is referred to as @(v-policy) in the remaining of this file.
      # the relationship is enforced by the constant v0 below, which ensures
      # that \sum_i u(\ell_i) = 1 after n-1 divisions by 2.
      Y = dataset.target
      (m, n) = Y.shape
      U = np.zeros((m,n))
      v0 = 2**(n-1)/(2**n - 1)
      for i in range(m):
        v = v0
        for r in range(n):
          j = Y[i,r]   # j identifies the label in rank r for the i-th individual
          if(j >= 0):  # (j == -1) is used to represent incomplete rankings
            U[i,j] = v # v is the degree of membership of the i-th individual to
                       # the label j
            v = v/2    # the next label in the ranking is assumed to have half of
                       # the degree membership of the current label

      # renormalises U in case of incomplete rankings
      U = U / U.sum(axis=1)[:, None]

      needs_post_processing = False

    else:
      raise ValueError(f'Unexpected dataset type {dataset.atype}')

  elif(procedure[0] == ECO_ASSIGNMENT_H):

    # applies hierarchical clustering
    linkage = procedure[1]
    model = AgglomerativeClustering(n_clusters = ngroups,
                                    linkage    = linkage).fit(dataset.data)

    assignments = [ [label] for label in model.labels_]
    memberships = [ [1.0]   for label in model.labels_]

  elif(procedure[0] in [ECO_ASSIGNMENT_F, ECO_ASSIGNMENT_R]):

    """
    Obtains an estimate for the fuzzifier parameter using calibration rules from:
    Yu, Jian, Qiansheng Cheng, and Houkuan Huang.
    "Analysis of the weighting exponent in the FCM."
    IEEE Transactions on Systems, Man, and Cybernetics (2004).
    DOI: 10.1109/TSMCB.2003.810951
    """
    X = copy(dataset.data)
    X = X - X.mean(axis=0)
    X = X / np.linalg.norm(X, axis=1)[:,None]
    X_tilde = (1/m) * X.dot(X.T)
    _lambda_max = np.linalg.eigh(X_tilde)[0].max()

    if(min(m-1, d) >= 3):
      fuzzifier = min(d, m-1)/(min(d, m-1) - 2)
    elif(_lambda_max < 0.5):
      fuzzifier = 1/(1 - 2 * _lambda_max)
    else:
      fuzzifier = 2.0

    # submits the data to fuzzy clustering
    cntr, u, u0, d, jm, p, fpc = cmeans(data    = dataset.data.T, # data to be clustered
                                        c       = ngroups,        # desired number of clusters
                                        m       = fuzzifier,
                                        error   = 1E-6,
                                        maxiter = 1000,
                                        init    = None,
                                        seed    = ECO_SEED)

    # Let m be the number of samples
    #     d be the number of features
    #     n the number of clusters
    # Then:
    # V.T  ~ (d x m)-matrix: the training data
    # cntr ~ (n x d)-matrix: cluster centers
    # u    ~ (n x m)-matrix: final   fuzzy c-partitioned matrix
    # u0   ~ (n x m)-matrix: initial fuzzy c-partitioned matrix
    # d    ~ (n x m)-matrix: final Euclidean distance matrix (point, cluster center)
    # jm   ~ (p)-array:      objective function history
    # p    ~ scalar:         number of iterations run
    # fpc  ~ scalar:         final fuzzy partition coefficient

    assignments = []
    memberships = []
    cutoff = u.T.mean(axis=0) * fuzzfactor
    for i in range(m):

      # recovers the degrees of membership of the i-th instance to all clusters,
      # enumerate them and have them sorted in descending order of membership
      # e.g., L0 = [(id of cluster with highest membership, membership degree),
      #             (id of cluster with second highest membership, degree), ...
      #             (id of cluster with lowest membership, degree)]
      L0 = sorted([(j, mu_ij) for (j, mu_ij) in enumerate(u.T[i])], key=lambda e: -e[1])

      # removes from L0 any clusters with membership lower than the cutoff, but
      # ensures at least the cluster with highest membership degree is kept.
      # @(cut-off-filter)
      L1 = [L0[0]] + [(j, mu_ij) for (j, mu_ij) in L0[1:] if mu_ij >= cutoff[j]]

      # standardises the membership degree to comply with the policy described
      # earlier as the @(v-policy):
      # -- given two consecutive labels in a ranking, \ell_i \succ \ell_{i+1},
      #    the degree of membership u(\ell_i) = 2 u(\ell_{i+1}).
      n  = ngroups
      u1 = lambda l: 2**(n-1)/((2**n - 1)*(2**l))
      L2 = [(p, u1(l)) for (l,(p,_)) in enumerate(L1)]
      #@@@figure10
      #    enable this to reproduce the figure at the end of section 4.3 of the thesis
      #    (images/proposal/figure_10.pdf)
      #L2=L1

      # ensures the final relationship (for multilabel and label ranking) has at
      # most max_labels elements.
      # NOTE that L1[0:max_labels] == L1 if max_labels is None
      # @(min-max-labels-filter)
      assignments.append([j     for (j, mu_ij) in L2[0:max_labels]])
      memberships.append([mu_ij for (j, mu_ij) in L2[0:max_labels]])

    if(procedure[0] == ECO_ASSIGNMENT_R):

      # builds the membership matrix, which is equal to u.T, except for the
      # elements that were removed by the @(cut-off-filter) and
      # @(min-max-labels-filter) above
      U = np.zeros((m,ngroups), dtype=np.float64)
      for i in range(m):
        for (j, mu_ij) in zip(assignments[i], memberships[i]):
          U[i,j] = mu_ij
      U = U / U.sum(axis=1)[:, None]

      # creates a ranking-encoded Y matrix from the membership matrix U,
      # NOTE: encodes zero membership degree as -1 (in case of a partial order)
      Y = (-U).argsort()
      Y = prune_rank(Y, U, 0., dominance=False)

      needs_post_processing = False

  else:
    raise ValueError('Assignment procedure not implemented: {0}'.format(procedure[0]))

  # converts the assignments/memberships from 'list of lists' to an (m x n) array
  if(needs_post_processing):
    Y = np.zeros((m,ngroups), dtype=int)
    U = np.zeros((m,ngroups), dtype=np.float64)
    for i in range(m):
      for (j, mu_ij) in zip(assignments[i], memberships[i]):
        Y[i,j] = 1
        U[i,j] = mu_ij

    # ensures the total membership of each sample is 1
    U = U / U.sum(axis=1)[:, None]

  return (Y, U)

def assignments2text(dataset, procedure):
  columns = (dataset.feature_names +
             ['A.G{0}'.format(ngroups) for ngroups in dataset.assignments] +
             ['L.G{0}'.format(ngroups) for ngroups in dataset.memberships]
            )
  mask = '\t'.join(['{' + '{0}'.format(pos) + '}' for pos in range(len(columns))])
  header = mask.format(*columns)
  content = [header]
  (m,d) = dataset.data.shape
  for i in range(m):
    temp = ( dataset.odata[i].tolist() +
            [dataset.assignments[ngroups][i] for ngroups in dataset.assignments] +
            [dataset.memberships[ngroups][i] for ngroups in dataset.memberships]
           )
    buffer = mask.format(*temp)
    content.append(buffer)

  return '\n'.join(content)

def split_dataset(X, Y_, scenario=ECO_DB_MULTICLASS, test_fraction=0.33, strategy=ECO_SPLIT_MVS, force_balance=False, threshold=0.05, max_tries=np.inf, random_seed=ECO_SEED):

  np.random.seed(random_seed)

  if(strategy == ECO_SPLIT_MVS):

    if(scenario == ECO_DB_MULTICLASS):
      # each label in Y_ identities a labelset
      # thus, we can obtain near-congruent train/test partitions directly from Y_
      # (here, two partitions are near-congruent if the proportion of instances with
      # labelset {L_i} is approximately the same in both partitions for all labels)
      Y = Y_
    elif(scenario in [ECO_DB_MULTILABEL, ECO_DB_LABELRANK]):
      # in this case, a label does not identify a labelset
      # to obtain near-congruent partitions, we need to account for labelsets instead of
      # labels. Thus, we create Y from Y_ by transforming each unique labelset into a
      # label and ensure partitions are congruent according to Y
      (m,n) = Y_.shape
      unique_label_sets = sorted(set([tuple(e) for e in Y_.tolist()]))
      N = len(unique_label_sets)
      Y = np.zeros((m, N), dtype=int)
      for i in range(m):
        j = unique_label_sets.index(tuple(Y_[i]))
        Y[i,j] = 1

    (m,n) = Y.shape
    if(force_balance):
      # ensures each labelset is equally represented in the resulting partitions
      # -- this approach is only valid for multiclass assignments
      if(scenario not in [ECO_DB_MULTICLASS]):
        raise ValueError('Force balance only works on multiclass datasets')
      label_instances = defaultdict(list)
      for (i,j) in enumerate(Y.argmax(axis=1)):
        label_instances[j].append(i)
      class_sizes = {j:len(label_instances[j]) for j in label_instances}
      max_ninstances = min(list(class_sizes.values()))
      if(max_ninstances == 0):
        raise ValueError(f'Force-balance strategy found empty class: {class_sizes}')

      sao = [] # sao ~ sample access order
      te_border = int(test_fraction * max_ninstances)
      if(te_border == 0):
        raise ValueError(f'Force-balance strategy could not split partition: consider increasing the fraction of samples allocated to test')

      # fills in the test partition (i.e., sao[0:border]) preserving label balance
      for j in label_instances:
        shuffle(label_instances[j])
        sao += label_instances[j][0:te_border]
      # fills in the training partition preserving label balance
      for j in label_instances:
        sao += label_instances[j][te_border:max_ninstances]
      v = Y[sao,:].mean(axis = 0)
      border = te_border * n

    else:
      # uses the whole dataset
      sao = list(range(m)) # sao ~ sample access order
      v = Y.mean(axis = 0)
      border = int(test_fraction * m)

    (tries, last, best) = (0, np.inf, copy(sao))
    while(True):
      (te_idxs, tr_idxs) = (sao[0:border], sao[border:])
      #w = Y[te_idxs].mean(axis = 0)
      w = Y[tr_idxs].mean(axis = 0)
      watermark = np.linalg.norm(v - w)
      #watermark = np.arccos(v.dot(w)/np.sqrt(v.dot(v) * w.dot(w)))
      if(watermark < last):
        last = watermark
        best = copy(sao)
      success = watermark < threshold
      tries += 1
      if(success or tries > max(m, max_tries)):
        break
      shuffle(sao)
    te_idxs, tr_idxs = best[0:border], best[border:]

  elif(strategy == ECO_SPLIT_STV):
    # this strategy is based on:
    # Sechidis, Tsoumakas, and Vlahavas (2011), Szymanski & Kajdanowicz (2017)
    if(scenario in [ECO_DB_MULTICLASS]):
      raise ValueError('STV strategy is not recommended for multiclass datasets')
    test_size = test_fraction
    stratifier = IterativeStratification(
        n_splits=2,
        order=1,
        sample_distribution_per_fold=[test_size, 1.0 - test_size],
        shuffle=False,
        random_state=random_seed
    )

    Y = Y_
    tr_idxs, te_idxs = next(stratifier.split(X, Y))
    v = Y.mean(axis = 0)
    w = Y[te_idxs].mean(axis = 0)
    last = np.linalg.norm(v - w)
    success = last < threshold
    tries = 1

  else:
    raise NotImplementedError(f'Splitting strategy {strategy} not implemented')

  return(np.array(te_idxs), np.array(tr_idxs), success, last, tries)

def rank2presence(Y_):
  Y = np.zeros(Y_.shape, dtype=int)
  (m,n) = Y.shape
  for i in range(m):
    for j in range(n):
      p = Y_[i,j]
      if(p >= 0):
        Y[i,p] = 1
  return Y

def presence2rank(Y_):
  Y = -1 * np.ones(Y_.shape, dtype=int)
  l = -1
  j = -1
  for (i, p) in zip(*np.where(Y_ == 1)):
    if(i > l):
      j = 0
      l = i
    else:
      j += 1
    Y[i,j] = p
  return Y


def membership2presence(U):
  Y = np.argsort(-U)
  Y = prune_rank(Y, U, 0., dominance=False)
  return Y

def compute_split_stats(Y, scenario):

  # counts the number of occurrences of each label
  # in multiclass/multilabel scenarios, Y is presence encoded,
  # while in   label ranking scenarios, Y is ranking encoded
  # (not worry, count_labels handles all scenarios)
  g = count_labels(Y, scenario)
  label_counts = np.array([g[label] for label in sorted(g)])
  imbalance = 1.0 - label_counts.min()/label_counts.max()

  # counts the number of occurrences of each labelset
  # in multiclass/multilabel scenarios, Y is presence encoded,
  # -- thus each labelset is a binary string
  # while in   label ranking scenarios, Y is ranking encoded,
  # -- thus each labelset is a (possibly partial) permutation of labels
  labelsets = [tuple(e) for e in Y.tolist()]
  single_labelsets = []
  summary = Counter(labelsets)
  for labelset in summary:
    if(summary[labelset] == 1):
      single_labelsets.append(labelset)

  # next, cardinality and density are computed, and they expect Y to be presence-encoded
  if(scenario in [ECO_DB_MULTICLASS, ECO_DB_MULTILABEL]):
    Y_p = Y #
  elif(scenario in [ECO_DB_LABELRANK]):
    Y_p = rank2presence(Y)
  else:
    raise NotImplementedError('Evaluation scenario not implemented')
  cardinality  = Y_p.sum(axis=1).mean()
  density = Y_p.mean()

  return (cardinality, label_counts, imbalance, labelsets, single_labelsets, density)

def report_split_stats(Y, scenario):

  (m,n) = Y.shape
  (cardinality, label_counts, imbalance,
   labelsets, single_labelsets, density) = compute_split_stats(Y, scenario)

  content = []
  buffer = f'\n   partition has {m} rows and {n} columns'
  content.append(buffer)

  buffer = '\n   cardinality .....: {0:2.2f}'.format(cardinality)
  content.append(buffer)

  buffer = '\n   class imbalance .: {0:2.2f}'.format(imbalance)
  content.append(buffer)

  buffer = '\n   #labelsets ......: {0:4d}'.format(len(set(labelsets)))
  content.append(buffer)

  buffer = '\n   #single labelsets: {0:4d}'.format(len(single_labelsets))
  content.append(buffer)

  buffer = '\n   density .........: {0:2.2f}'.format(density)
  content.append(buffer)

  return content

def count_labels(Y, scenario):
  (m,n) = Y.shape

  if(scenario in [ECO_DB_MULTICLASS, ECO_DB_MULTILABEL]):
    ysum = Y.sum(axis=0)
    counter = {j: ysum[j] for j in range(n)}
  elif(scenario in [ECO_DB_LABELRANK]):
    counter = {j: 0 for j in range(n)}
    for i in range(m):
      for r in range(n):
        j = Y[i,r]
        if(j >= 0): #-1 encodes empty in partial orders
          counter[j] += 1
  return counter

def prune_rank(Y, U, thresholds, dominance=True):
  # assumes Y is encoded as label ranking, and
  #         U is a matrix with membership degrees
  (m,n) = Y.shape
  _Y = copy(Y)
  if(isinstance(thresholds, float)):
    tw = thresholds * np.ones(n)
  elif(isinstance(thresholds[0], float)):
    tw = thresholds
  else:
    raise ValueError('Parameter thresholds should be float or array of floats')

  if(dominance):
    # in creating the ranking matrix Y, this block includes items whose degree
    # of membership U[i,j] is the minimal observed value. This is used by
    # Polygrid.learn_thresholds and .predict, as well as
    # BaseCompetitor.learn_thresholds, .post_predict to search for an optimal
    # cutoff point for all scales (or each scale individually).
    for i in range(m):
      for r in range(n):
        j = Y[i,r]           # j is the index of the label of rank r for instance i
        if(U[i,j] < tw[j]):  # the degree of membership of instance i to label j is
          _Y[i,r] = -1       # dominated by the cut off, then j is pruned
  else:
    # in creating the ranking matrix Y, this block excludes items whose degree
    # of membership U[i,j] is zero (which happens to be the minimal possible
    # degree of membership). This is used by assign_groups to encode partial
    # orders in scenarios where the original assignments are replaced to
    # represent a different task (multilabel classification, label ranking)
    for i in range(m):
      for r in range(n):
        j = Y[i,r]           # j is the index of the label of rank r for instance i
        if(U[i,j] <= tw[j]): # the degree of membership of instance i to label j is
          _Y[i,r] = -1       # semi-dominated by the cut off, then j is pruned

  return _Y

#-------------------------------------------------------------------------------------------------------------------------------------------
# Problem-specific definitions - interface to handle research datasets
#-------------------------------------------------------------------------------------------------------------------------------------------
def list_conflicts(X, feature_names):

  # obtains the correlation matrix of X
  C = np.corrcoef(X, rowvar=False)
  (d,_) = C.shape

  # finds the feature that most negatively correlates with the others ...
  counter = np.zeros(d, dtype=int)
  for j1 in range(d-1):
    for j2 in range(j1+1, d):
      if(C[j1,j2] < 0.):
        counter[j1] += 1
        counter[j2] += 1

  header  = ('feature', '#conflicts')
  content = [header]
  if(counter.sum() == 0):
    content.append(('none', '-'))
  else:
    for j in (-counter).argsort():
      if(counter[j] > 0):
        content.append((feature_names[j], counter[j]))

  return (content)

def remove_conflicts(X, feature_names, maxiter=10):
  it = 0
  while(it < maxiter):

    # obtains the correlation matrix of X
    C = np.corrcoef(X, rowvar=False)
    (d,_) = C.shape

    # finds the feature that most negatively correlates with the others ...
    counter = np.zeros(d, dtype=int)
    for j1 in range(d-1):
      for j2 in range(j1+1, d):
        if(C[j1,j2] < 0.):
          counter[j1] += 1
          counter[j2] += 1
    conflict = counter.argmax()
    if(counter.sum() == 0):
      break

    # ... and removes it from the matrix of features X
    cols_filter = [j for j in range(d) if j != conflict]
    X = X[:,cols_filter]
    feature_names = [feature_names[j] for j in cols_filter]

    it += 1

  return (X, feature_names)

def reverse_conflicts(X, feature_names, maxiter=10):
  it = 0
  while(it < maxiter):

    # obtains the correlation matrix of X
    C = np.corrcoef(X, rowvar=False)
    (d,_) = C.shape

    # finds the feature that most negatively correlates with the others ...
    counter = np.zeros(d, dtype=int)
    for j1 in range(d-1):
      for j2 in range(j1+1, d):
        if(C[j1,j2] < 0.):
          counter[j1] += 1
          counter[j2] += 1
    conflict = counter.argmax()
    if(counter.sum() == 0):
      break

    # ... and reverses its encoding in the matrix of features X
    X[:,conflict] = 1 - X[:,conflict]

    it += 1

  return (X, feature_names)

def palmer_load_penguins():
  # from https://github.com/mcnakhaee/palmerpenguins
  # python -m pip install palmerpenguins

  # loads the Palmer Penguins dataset and removes rows with missing data
  df = _load_penguins()
  df.insert(0, 'id', df.index) # so that we can trace to original subject ids
  df = df.dropna()

  # gets all features to fit the same order of magnitude
  df['flipper_length_mm'] = df['flipper_length_mm'] / 10
  df['body_mass_g'] = df['body_mass_g'] / 100

  # groups columns that will become similar resources in the bunch
  (_id, _target) = (0, 1)
  var_data    = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
  var_factors = ['island', 'sex', 'year']

  # iterates over the dataset rows and transforms the data, as required
  caseIDs = []
  data    = []
  factors = []
  target  = []
  for (index, row) in df.iterrows():
    caseIDs.append(row.iloc[_id])
    data.append(row[var_data].values)
    factors.append(''.join([f'/{str(val).lower()}' for val in row[var_factors].values]) + '/')
    target.append(row.iloc[_target])

  fdescr = 'Palmer penguins'
  feature_names = var_data
  target_names  = sorted(set(target))
  factor_names  = var_factors
  data    = np.array(data, dtype=np.float64)
  factors = np.array(factors)
  target  = np.array([target_names.index(outcome) for outcome in target])

  dataset = Bunch(
      DESCR=fdescr,
      feature_names=feature_names,
      target_names=target_names,
      factor_names=factor_names,
      data=data,
      factors=factors,
      target=target,
      caseIDs=caseIDs,
      symbol=ECO_CAPACITY,
  )

  return dataset

def load_whoqol(basepath, basename):
  sourcepath = basepath + ['ufscar', 'preprocessed',]
  dataset    = deserialise(os.path.join(*sourcepath, f'{basename}-data'))
  return dataset

def load_ampiab(basepath, basename):
  sourcepath = basepath + ['each', 'preprocessed']
  dataset    = deserialise(os.path.join(*sourcepath, f'{basename}-data'))
  return dataset

def load_elsio1(basepath, basename):
  sourcepath = basepath + ['fiocruz', 'preprocessed']
  dataset    = deserialise(os.path.join(*sourcepath, f'{basename}-data'))
  return dataset

def paderborn_load(basepath, basename):
  sourcepath = basepath + ['paderborn', 'preprocessed']
  dataset    = deserialise(os.path.join(*sourcepath, basename))
  return dataset

def load_foodtruck(basepath, basename):
  sourcepath = basepath + ['cometa', 'preprocessed']
  dataset    = deserialise(os.path.join(*sourcepath, f'{basename}-data'))
  return dataset

def load_water(basepath, basename):
  sourcepath = basepath + ['kdis', 'preprocessed']
  dataset    = deserialise(os.path.join(*sourcepath, f'{basename}-data'))
  return dataset

