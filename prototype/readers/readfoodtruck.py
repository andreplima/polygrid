import sys
import numpy as np

from os             import listdir, makedirs, remove
from os.path        import join, exists
from collections    import Counter
from scipy.io       import arff
from customDefs     import getMountedOn, setupEssayConfig, getEssayParameter
from customDefs     import tsprint, saveLog, serialise
from customDefs     import ECO_CAPACITY
from sklearn.utils  import Bunch

# functions used to encode raw data to numeric format for select attributes
# -- used only when PARAM_FOODTRUCK_NO_FACTORS=True
def encode_time(vals):
  val2code = {
    b'lunch':      0,
    b'afternoon':  1,
    b'dawn':       2,
    b'happy_hour': 3,
    b'dinner':     4,
  }
  return np.array([val2code[val] for val in vals])

def encode_motivation(vals):
  val2code = {
    b'ads':            0,
    b'by_chance':      1,
    b'friend':         2,
    b'social_network': 3,
    b'web':            4,
  }
  return np.array([val2code[val] for val in vals])

def encode_marital_status(vals):
  val2code = {
    b'single':   0,
    b'married':  1,
    b'divorced': 2,
  }
  return np.array([val2code[val] for val in vals])

def encode_gender(vals):
  val2code = {
    b'M': 0,
    b'F': 1,
  }
  return np.array([val2code[val] for val in vals])

# functions used to recode raw data to specific datatypes
def recode_as_int(vals):
  return np.array([int(val) for val in vals])

def recode_as_str(vals):
  return np.array([val.decode('utf-8') for val in vals])

# functions used to recode raw data to string format for attributes cast as factors
# -- used only when PARAM_FOODTRUCK_NO_FACTORS=False
def recode_frequency(vals):
  val2code = {
    0: 'rarely',
    1: 'monthly',
    2: 'weekly',
    3: 'twice_a_week',
    4: 'daily',
  }
  return np.array([val2code[int(val)] for val in vals])

def recode_expenses(vals):
  val2code = {
    15: '$up_to_15',
    20: '$15_to_20',
    30: '$20_to_30',
    40: '$30_to_40',
    50: '$40_or_above',
  }
  return np.array([val2code[int(val)] for val in vals])

def recode_gender(vals):
  val2code = {
    'F': 'female',
    'M': 'male',
  }
  return np.array([val2code[val.decode('utf-8')] for val in vals])

def recode_age_group(vals):
  val2code = {
    1: 'up_to_19',
    2: '20_to_25',
    3: '26_to_30',
    4: '31_to_35',
    5: '36_to_40',
    6: '41_to_45',
    7: '46_to_50',
    8: '51_or_above',
  }
  return np.array([val2code[int(val)] for val in vals])

def recode_scholarity(vals):
  val2code = {
     0: 'no_diploma',
    10: 'high_school',
    15: 'in_graduation',
    20: 'graduation',
    30: 'specialization',
    40: 'masters',
    50: 'phd',
  }
  return np.array([val2code[int(val * 10)] for val in vals])

def recode_average_income(vals):
  val2code = {
    1: '$around_1k',
    2: '$around_3k',
    3: '$around_5k',
    4: '$around_10k',
    5: '$around_20k',
    6: '$above_20k',
  }
  return np.array([val2code[int(val)] for val in vals])

def recode_has_work(vals):
  val2code = {
    0: 'unemployed',
    1: 'employed',
  }
  return np.array([val2code[int(val)] for val in vals])

# converts transformed data to comply with scikit-learn datasets API
def convert2Bunch(meta, rawdata, variables):

  (features, factors, targets) = variables
  (m,n) = rawdata.shape

  fdescr  = 'Dataset Foodtruck (Rivolli, Parker, and Carvalho, 2017)'
  caseIDs = list(range(m))

  col_features  = [meta.names().index(feature) for feature in features]
  feature_data  = rawdata[:, col_features].astype(float)
  feature_names = features

  col_factors   = [meta.names().index(factor) for factor in factors]
  factor_data   = np.array([''.join([f'/{val}' for val in row]) + '/' for row in rawdata[:, col_factors]])
  factor_names  = factors

  col_targets   = [meta.names().index(target) for target in targets]
  target_data   = rawdata[:, col_targets].astype(int)
  target_names  = targets

  newBunch = Bunch(
      DESCR=fdescr,
      feature_names = feature_names,
      target_names  = target_names,
      factor_names  = factor_names,
      data    = feature_data,
      factors = factor_data,
      target  = target_data,
      caseIDs = caseIDs,
      symbol  = ECO_CAPACITY,
      meta    = meta,
      rawdata = rawdata,
      col_factors = col_factors,
  )

  return newBunch

# saves data into a csv file
def saveView(meta, data, filename):

  def type2fmt(e):
    if(type(e) is int):
      fmt = '%1d'
    elif(type(e) is np.float64):
      fmt = '%.3f'
    elif(type(e) is float):
      fmt = '%.3f'
    elif(type(e) is str):
      fmt = '%s'
    elif(type(e) is bytes):
      fmt = '%s'
    else:
      raise ValueError(f'Unexpected type for {e} of type {type(e)}')
    return fmt

  header = '\t'.join(meta.names())
  fmt = [type2fmt(val) for val in data[0,:]]
  np.savetxt(filename,
             data,
             fmt=fmt,
             delimiter='\t',
             newline='\n',
             header=header,
             encoding='utf-8',
            )
  return None

# computes relevant statistics over the dataset
def show_statistics(meta, data, variables):

  (features, factors, targets) = variables

  tsprint('Reproducing statistics shown in the original publication (see Table 2)')
  tsprint('----------------------------------------------------------------------')
  tsprint('Characteristic\t\tValue')
  tsprint('----------------------------------------------------------------------')

  attributes = features + factors
  tsprint(f'#Attributes\t\t{len(attributes)}')

  (m, _) = data.shape
  tsprint(f'#Instances\t\t{m}')

  labels = targets
  tsprint(f'#Labels\t\t{len(labels)}')

  cols_filter = [meta.names().index(target) for target in targets]
  cardinality = data[:,cols_filter].sum(axis=1).mean()
  tsprint(f'Cardinality\t\t{cardinality}')

  labelsets = [tuple(e) for e in data[:,cols_filter].tolist()]
  tsprint(f'#Labelsets\t\t{len(set(labelsets))}')

  single_labelsets = []
  hist = Counter(labelsets)
  for labelset in hist:
    if(hist[labelset] == 1):
      single_labelsets.append(labelset)
  tsprint(f'#Single labelsets\t{len(single_labelsets)}')

  c = np.corrcoef(data[:,cols_filter].astype(float), rowvar=False)
  m = len(cols_filter)
  vals = []
  for i in range(m):
    for j in range(i+1, m):
      vals.append(c[i,j])
  label_dependency = np.array(vals).mean()
  tsprint(f'Label dependency\t{label_dependency}')

  density = data[:,cols_filter].mean()
  tsprint(f'Density\t\t{density}')
  tsprint('----------------------------------------------------------------------')

  return None

def main(configfile):

  tsprint(f'Loading parameters from {configfile}')

  setupEssayConfig(configfile)
  sourcepath  = getEssayParameter('PARAM_FOODTRUCK_SOURCEPATH')
  targetpath  = getEssayParameter('PARAM_FOODTRUCK_TARGETPATH')
  no_factors  = getEssayParameter('PARAM_FOODTRUCK_NO_FACTORS')
  a4id        = getEssayParameter('PARAM_FOODTRUCK_A4ID')

  # ensures the target folder is available and empty
  if(exists(join(*targetpath))):
    for f in listdir(join(*targetpath)):
      remove(join(*targetpath, f))
  else:
    makedirs(join(*targetpath))

  # reads the dataset file
  filename = getEssayParameter('PARAM_FOODTRUCK_FILENAME')
  tsprint(f'Loading and parsing file {join(*sourcepath, filename)}')
  (data, meta) = arff.loadarff(join(*sourcepath, filename)) # data is 1d-array of tuples
  data = np.array(data.tolist(), dtype=object)              # data is now a 2d-array

  # saves a view of the dataset (before conversions and transformations)
  filename = 'foodtruck-data-before.csv'
  saveView(meta, data, join(*targetpath, filename))

  # maps each field to its corresponding function in a PolygridCLI environment
  tsprint(f'Converting data to types required by Polygrid CLI environment')
  features = ['taste', 'hygiene', 'menu', 'presentation', 'attendance', 'ingredients', 'place.to.sit', 'takeaway', 'variation', 'stop.strucks', 'schedule', ]
  factors  = ['frequency', 'time', 'expenses', 'motivation', 'gender', 'marital.status', 'age.group', 'scholarity', 'average.income', 'has.work', ]
  targets  = ['street_food', 'gourmet', 'italian_food', 'brazilian_food', 'mexican_food', 'chinese_food', 'japanese_food', 'arabic_food', 'snacks', 'healthy_food', 'fitness_food', 'sweets_desserts', ]

  # converts data types of factor/target fields to comply with Polygrid requisites
  conversions = {

    # factors
    'time': recode_as_str,
    'motivation': recode_as_str,
    'marital.status': recode_as_str,
    'frequency': recode_frequency,
    'expenses': recode_expenses,
    'gender': recode_gender,
    'age.group': recode_age_group,
    'scholarity': recode_scholarity,
    'average.income': recode_average_income,
    'has.work': recode_has_work,

    # targets
    'street_food': recode_as_int,
    'gourmet': recode_as_int,
    'italian_food': recode_as_int,
    'brazilian_food': recode_as_int,
    'mexican_food': recode_as_int,
    'chinese_food': recode_as_int,
    'japanese_food': recode_as_int,
    'arabic_food': recode_as_int,
    'snacks': recode_as_int,
    'healthy_food': recode_as_int,
    'fitness_food': recode_as_int,
    'sweets_desserts': recode_as_int,
  }

  # modifies the above definitions to consider all factors as features
  # -- this is used to create a version of the dataset needed by mlc_reproduce.py,
  #    which reproduces some of the relevant statistics published in the original article
  if(no_factors):

    # all factors become features
    features = ['taste', 'hygiene', 'menu', 'presentation', 'attendance', 'ingredients', 'place.to.sit', 'takeaway', 'variation', 'stop.strucks', 'schedule', 'frequency', 'expenses', 'age.group', 'scholarity', 'average.income', 'has.work', 'time', 'motivation', 'gender', 'marital.status',]
    factors  = []

    # converts nominal factors into numeric features
    # -- WARNING converting nominal to interval/ratio is problematic, in these cases
    conversions = {

      # factors turned features
      'time': encode_time,
      'motivation': encode_motivation,
      'marital.status': encode_marital_status,
      'gender': encode_gender,

      # targets
      'street_food': recode_as_int,
      'gourmet': recode_as_int,
      'italian_food': recode_as_int,
      'brazilian_food': recode_as_int,
      'mexican_food': recode_as_int,
      'chinese_food': recode_as_int,
      'japanese_food': recode_as_int,
      'arabic_food': recode_as_int,
      'snacks': recode_as_int,
      'healthy_food': recode_as_int,
      'fitness_food': recode_as_int,
      'sweets_desserts': recode_as_int,
    }

  # applies the conversions defined above
  for field_name in conversions:
    j = meta.names().index(field_name)
    vals = data[:,j].tolist()
    data[:,j] = conversions[field_name](vals)

  # transforms features to account for individual differences when answering
  # Likert-based questionnaires
  if(a4id):
    tsprint('Accounting for individual differences when answering Likert-base questionnaires')
    cols_filter = [meta.names().index(feature) for feature in features]
    mu = data[:,cols_filter].mean()
    bi = data[:,cols_filter].mean(axis=0) - mu
    bu = data[:,cols_filter].mean(axis=1) - mu
    (m,n) = data[:,cols_filter].shape
    for u in range(m):     # participant #u (user)
      for i in range(n):   # questionnaire item #i (using a 5-point Likert scale)
        ii = cols_filter[i]
        data[u,ii] = mu + bu[u] + bi[i]

  tsprint('----------------------------------------------------------------------')
  tsprint('Group\tField\tField Type')
  for (label, fieldset) in [('feature', features), ('target', targets), ('factor', factors)]:
    tsprint('----------------------------------------------------------------------')
    for field in fieldset:
      j = meta.names().index(field)
      tsprint(f'{label}\t{field}\t{type(data[0,j])}')
  tsprint('----------------------------------------------------------------------')

  # serialises the converted dataset as a Bunch object
  tsprint('Saving the converted data')
  filename = 'foodtruck-data'
  newBunch = convert2Bunch(meta, data, (features, factors, targets))
  serialise(newBunch, join(*targetpath, filename))

  # creates a view of the dataset
  filename = 'foodtruck-data-after.csv'
  saveView(meta, data, join(*targetpath, filename))

  # reproduces statistics shown in the original publication (see Table 2 of:)
  #   Rivolli, Adriano, Larissa C. Parker, and Andre CPLF de Carvalho.
  #   "Food truck recommendation using multi-label classification."
  #   Progress in Artificial Intelligence: 18th EPIA Conference on Artificial Intelligence,
  #   EPIA 2017, Porto, Portugal, September 5-8, 2017, Proceedings 18.
  #   Springer International Publishing, 2017.
  show_statistics(meta, data, (features, factors, targets))

  # saves the execution log
  logfile = 'readfoodtruck.log'
  tsprint('This report was saved to {0}'.format(join(*targetpath, logfile)))
  tsprint('Done.')
  saveLog(join(*targetpath, logfile))


if(__name__ == '__main__'):

  main(sys.argv[1])
