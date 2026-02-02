import re
import os
import codecs
import pickle
import numpy as np

from datetime     import datetime, timedelta
from collections  import OrderedDict
from configparser import RawConfigParser

ECO_SEED = 23
ECO_DATETIME_FMT = '%Y%m%d%H%M%S'
ECO_DATETIME_OUT = '%Y-%m-%d %H:%M'
ECO_DPI = 350

# constants identifying the nature of the content of the latent variable
# (used to determine color encoding of the assessment polygons in the Polygrid class)
ECO_DEFICIT  = -1
ECO_CAPACITY =  1

# number of cutoffs to be used in Model.learn_thresholds(...)
ECO_CUTOFF_SINGLE   = 'single'    # a single threshold for all interventions
ECO_CUTOFF_MULTIPLE = 'multiple'  # one threshold for each intervention

# number of threshold levels to be used in Model.learn_thresholds(...)
ECO_THRSHLDLVLS = 100

# constants identifying the types of model outcome events (used in show_scales)
ECO_HIT  = 'hit(s)  ' #'correctly classified/ranked'
ECO_MISS = 'miss(es)' #'wrongly classified/ranked'

#--------------------------------------------------------------------------------------------------
# General purpose definitions - I/O interfaces used in logging and serialisation
#--------------------------------------------------------------------------------------------------

# buffer where all tsprint messages are stored
LogBuffer = []

def stimestamp():
  return(datetime.now().strftime(ECO_DATETIME_FMT))

def stimediff(finishTs, startTs):
  return str(datetime.strptime(finishTs, ECO_DATETIME_FMT) - datetime.strptime(startTs, ECO_DATETIME_FMT))

def stimefuture(ahead):
  return (datetime.now() + timedelta(minutes=ahead)).strftime(ECO_DATETIME_OUT)

def tsprint(msg, verbose=True, stamp=True):
  if(stamp):
    buffer = '[{0}] {1}'.format(stimestamp(), msg)
  else:
    buffer = '{0}'.format(msg)
  if(verbose):
    print(buffer)
  LogBuffer.append(buffer)

class PrintBuffer:
  def __init__(self, sep = '\n'):
    self.content = []
    self.sep = sep
  def __str__(self):
    return self.sep.join(self.content)
  def clear(self):
    self.content = []
  def set(self, sep=None):
    if(sep is not None):
      self.sep = sep
  def add(self, msg):
    self.content.append(msg)

def resetLog():
  LogBuffer = []

def saveLog(filename):
  saveAsText('\n'.join(LogBuffer), filename)

def headerfy(mask):
  res = re.sub('\:\d+\.\d+f', '', mask)
  res = re.sub('\:\d+d', '', res)
  return res

def loadAsText(filename, _encoding = 'utf-8'):
  f = codecs.open(filename, 'r', encoding=_encoding)
  content = f.read()
  f.close()
  return content

def saveAsText(content, filename, _encoding='utf-8'):
  f = codecs.open(filename, 'w', encoding=_encoding)
  f.write(content)
  f.close()

def serialise(obj, name):
  f = open(name + '.pkl', 'wb')
  p = pickle.Pickler(f)
  p.fast = True
  p.dump(obj)
  f.close()
  p.clear_memo()

def deserialise(name):
  f = open(name + '.pkl', 'rb')
  p = pickle.Unpickler(f)
  obj = p.load()
  f.close()
  return obj

def getMountedOn():

  if('PARAM_MOUNTEDON' in os.environ):
    res = os.environ['PARAM_MOUNTEDON'] + os.sep
  else:
    res = os.getcwd().split(os.sep)[-0] + os.sep
  return res

#-------------------------------------------------------------------------------------------------------------------------------------------
# General purpose definitions - interface to handle parameter files
#-------------------------------------------------------------------------------------------------------------------------------------------

# Essay Parameters dictionary
EssayParameters = {}

boolean_params = ['PARAM_SAVEIT', 'PARAM_BALANCE', 'PARAM_HIDE_TAGS',
                  'PARAM_BACK2BACKAS', 'PARAM_SKIPFIGURES',
                  'PARAM_FOODTRUCK_NO_FACTORS', 'PARAM_FOODTRUCK_A4ID',
                 ]

integer_params = ['ESSAY_RUNS', 'PARAM_MAXCORES', 'PARAM_MAXLABELS',
                  'PARAM_NUMEXPERTS', 'PARAM_NSPD', 'PARAM_NA', 'PARAM_DATACOPIES',
                  'PARAM_TRANSPOSE',
                 ]

float_params = ['PARAM_TESTFRAC', 'PARAM_ALPHA',]

eval_params = ['PARAM_SOURCEPATH',  'PARAM_TARGETPATH', 'PARAM_GRIDSEARCH',
               'PARAM_PROCEDURES',  'PARAM_DATASETS',   'PARAM_DATAPARS',
               'PARAM_COMPETITORS', 'PARAM_POLYGRID',
               'PARAM_SCHEME', 'PARAM_TTC_TASK', 'PARAM_EXCLUDE', 'PARAM_WARMUP_PASS',
               'PARAM_ALPHA_RANGE', 'PARAM_ROUTE2AGENT',
               'PARAM_WHOQOL_SOURCEPATH',    'PARAM_WHOQOL_TARGETPATH', 'PARAM_WHOQOL_FUZZFACTOR',
               'PARAM_AMPIAB_SOURCEPATH',    'PARAM_AMPIAB_TARGETPATH', 'PARAM_AMPIAB_FUZZFACTOR',
               'PARAM_ELSIO1_SOURCEPATH',    'PARAM_ELSIO1_TARGETPATH', 'PARAM_ELSIO1_FUZZFACTOR',
               'PARAM_FOODTRUCK_SOURCEPATH', 'PARAM_FOODTRUCK_TARGETPATH',
               'PARAM_WATER_SOURCEPATH',     'PARAM_WATER_TARGETPATH',
               'PARAM_PADERBORN_BASEPATH',   'PARAM_PADERBORN_REPOSITORIES',
               'PARAM_ROW_LBLS',    'PARAM_COL_LBLS',   'PARAM_CFG_IDXS',
               'PARAM_MODELORDER', 'PARAM_SHORTNAMES',
              ]

def setupEssayConfig(configFile = ''):

  # initialises the random number generator
  np.random.seed(ECO_SEED)

  # defines default values for some configuration parameters
  setEssayParameter('ESSAY_ESSAYID',  'None')
  setEssayParameter('ESSAY_CONFIGID', 'None')
  setEssayParameter('ESSAY_SCENARIO', 'None')
  setEssayParameter('ESSAY_RUNS',     '1')

  # overrides default values with user-defined configuration
  loadEssayConfig(configFile)

  return listEssayConfig()

def setEssayParameter(param, value):
  """
  Purpose: sets the value of a specific parameter
  Arguments:
  - param: string that identifies the parameter
  - value: its new value
    Premises:
    1) When using inside python code, declare value as string, independently of its true type.
       Example: 'True', '0.32', 'Rastrigin, normalised'
    2) When using parameters in Config files, declare value as if it was a string, but without the enclosing ''.
       Example: True, 0.32, Rastrigin, only Reproduction
  Returns: None
  """

  so_param = param.upper()

  # boolean-valued parameters
  if(so_param in boolean_params):
    so_value = eval(value[0]) if isinstance(value, list) else bool(value)

  # integer-valued parameters
  elif(so_param in integer_params):
    so_value = eval(value[0])

  # floating-point-valued parameters
  elif(so_param in float_params):
    so_value = float(eval(value[0]))

  # parameters that requires eval expansion
  elif(so_param in eval_params):
    so_value = value

  # parameters that represent text
  else:
    so_value = value[0]

  EssayParameters[so_param] = so_value

def getEssayParameter(param):
  return EssayParameters[param.upper()]

class OrderedMultisetDict(OrderedDict):

  def __setitem__(self, key, value):

    try:
      item = self.__getitem__(key)
    except KeyError:
      super(OrderedMultisetDict, self).__setitem__(key, value)
      return

    if isinstance(value, list):
      item.extend(value)
    else:
      item.append(value)

    super(OrderedMultisetDict, self).__setitem__(key, item)

def loadEssayConfig(configFile):

  """
  Purpose: loads essay configuration coded in a essay parameters file
  Arguments:
  - configFile: name and path of the configuration file
  Returns: None, but EssayParameters dictionary is updated
  """

  if(len(configFile) > 0):

    if(os.path.exists(configFile)):

      # initialises the config parser and set a custom dictionary in order to allow multiple entries
      # of a same key (example: several instances of GA_ESSAY_ALLELE
      config = RawConfigParser(dict_type = OrderedMultisetDict)
      config.read(configFile)

      # loads parameters from the ESSAY section
      for param in config.options('ESSAY'):
        setEssayParameter(param, config.get('ESSAY', param))

      # loads parameters from the PROBLEM section
      for param in config.options('PROBLEM'):
        setEssayParameter(param, config.get('PROBLEM', param))

      # expands parameter values that requires evaluation
      # (parameters that hold lists, tuples, dictionaries, and the like)
      for param in eval_params:
        if(param in EssayParameters):
          EssayParameters[param]  = eval(EssayParameters[param][0])

      # checks if configuration is ok
      (check, errors) = checkEssayConfig(configFile)
      if(not check):
        print(errors)
        exit(1)

    else:

      print('*** Warning: Configuration file [{1}] was not found'.format(configFile))

def checkEssayConfig(configFile):

  check = True
  errors = []
  errorMsg = ""

  # insert criteria below
  if(EssayParameters['ESSAY_ESSAYID'] not in EssayParameters['ESSAY_SCENARIO']):
    check = False
    #errors.append("Parameter {0} must respect restriction: {1}\n".format('ESSAY_ESSAYID', 'be part of the ESSAY_SCENARIO identification'))
    param_name = 'ESSAY_ESSAYID'
    restriction = 'be part of the ESSAY_SCENARIO identification'
    errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  if(EssayParameters['ESSAY_CONFIGID'] not in EssayParameters['ESSAY_SCENARIO']):
    check = False
    #errors.append("Parameter {0} must respect restriction: {1}\n".format('ESSAY_CONFIGID', 'be part of the ESSAY_SCENARIO identification'))
    param_name = 'ESSAY_CONFIGID'
    restriction = 'be part of the ESSAY_SCENARIO identification'
    errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  if(EssayParameters['ESSAY_CONFIGID'].lower() not in configFile.lower()):
    check = False
    #errors.append("Parameter {0} must respect restriction: {1}\n".format('ESSAY_CONFIGID', 'be part of the configuration filename'))
    param_name = 'ESSAY_CONFIGID'
    restriction = 'be part of the config filename'
    errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  # summarises errors found
  if(len(errors) > 0):
    separator = "=============================================================================================================================\n"
    errorMsg = separator
    for i in range(0, len(errors)):
      errorMsg = errorMsg + errors[i]
    errorMsg = errorMsg + separator

  return(check, errorMsg)

def getEssayConfig():
  return EssayParameters

# recovers the current essay configuration
def listEssayConfig():

  res = ''
  for e in sorted(EssayParameters.items()):
    res = res + "{0} : {1} (as {2})\n".format(e[0], e[1], type(e[1]))

  return res

