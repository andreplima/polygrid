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

# converts transformed data to comply with scikit-learn datasets API
def convert2Bunch(meta, rawdata, variables):

  (features, factors, targets) = variables
  (m,n) = rawdata.shape

  fdescr  = 'Dataset Water Quality (Blockeel, Dzeroski, and Grbovic, 1999)'
  caseIDs = list(range(m))

  # this dataset has samples that are not assigned to any labels
  # these are removed
  col_targets   = [meta.names().index(target) for target in targets]
  rows_to_keep  = np.where(rawdata[:, col_targets].sum(axis=1) > 0)[0]
  caseIDs       = [caseID for caseID in caseIDs if caseID in rows_to_keep]
  rawdata       = rawdata[rows_to_keep,:]
  print( '-- WARNING: samples that have not been assigned to any label are being removed:')

  col_features  = [meta.names().index(feature) for feature in features]
  feature_data  = rawdata[:, col_features].astype(float)
  feature_names = features

  col_factors   = [meta.names().index(factor) for factor in factors]
  factor_data   = np.array([''.join([f'/{val}' for val in row]) + '/' for row in rawdata[:, col_factors]])
  factor_names  = factors

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
  )

  return (newBunch, rawdata)

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

  tsprint('Reproducing statistics shown in the original publication (see page 34)')
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
  tsprint(f'Cardinality\t\t{cardinality:5.2f}')

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
  tsprint(f'Label dependency\t{label_dependency:5.3f}')

  density = data[:,cols_filter].mean()
  tsprint(f'Density\t\t{density:5.3f}')
  tsprint('----------------------------------------------------------------------')

  return None

def main(configfile):

  tsprint(f'Loading parameters from {configfile}')

  setupEssayConfig(configfile)
  sourcepath  = getEssayParameter('PARAM_WATER_SOURCEPATH')
  targetpath  = getEssayParameter('PARAM_WATER_TARGETPATH')

  # ensures the target folder is available and empty
  if(exists(join(*targetpath))):
    for f in listdir(join(*targetpath)):
      remove(join(*targetpath, f))
  else:
    makedirs(join(*targetpath))

  # reads the dataset file
  filename = getEssayParameter('PARAM_WATER_FILENAME')
  tsprint(f'Loading and parsing file {join(*sourcepath, filename)}')
  (data, meta) = arff.loadarff(join(*sourcepath, filename)) # data is 1d-array of tuples
  data = np.array(data.tolist(), dtype=object)              # data is now a 2d-array

  # maps each field to its corresponding function in a PolygridCLI environment
  tsprint(f'Converting data to types required by Polygrid CLI environment')
  features = ['std_temp', 'std_pH', 'conduct', 'o2', 'o2sat', 'co2', 'hardness', 'no2', 'no3', 'nh4', 'po4', 'cl', 'sio2', 'kmno4', 'k2cr2o7', 'bod', ]
  factors  = []
  targets  = ['25400', '29600', '30400', '33400', '17300', '19400', '34500', '38100', '49700', '50390', '55800', '57500', '59300', '37880',  ]

  # in the arff file, the targets are defined as length-1 binary strings (bits)
  # we need them to be integers, so we force this casting
  for j in [meta.names().index(target) for target in targets]:
    data[:,j] = data[:,j].astype(int)

  # saves a view of the dataset (before conversions and transformations)
  filename = 'water-data-before.csv'
  saveView(meta, data, join(*targetpath, filename))

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
  filename = 'water-data'
  (newBunch, data) = convert2Bunch(meta, data, (features, factors, targets))
  serialise(newBunch, join(*targetpath, filename))

  # creates a view of the dataset
  filename = 'water-data-after.csv'
  saveView(meta, data, join(*targetpath, filename))

  """
  This reproduces statistics shown in the original publication (see page 34 of)
    Blockeel, Hendrik, Sašo Džeroski, and Jasna Grbović.
    Simultaneous prediction of multiple chemical parameters of river water quality
    with TILDE." In European Conference on Principles of Data Mining and
    Knowledge Discovery, pp. 32-40. Berlin, Heidelberg: Springer, 1999.
  """
  show_statistics(meta, data, (features, factors, targets))

  # saves the execution log
  logfile = 'readwater.log'
  tsprint('This report was saved to {0}'.format(join(*targetpath, logfile)))
  tsprint('Done.')
  saveLog(join(*targetpath, logfile))


if(__name__ == '__main__'):

  main(sys.argv[1])
