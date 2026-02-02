"""

  This script processes data that was collected by the Fiocruz's ELSI-Brazil project:

  [1] Lima-Costa MF, de Andrade FB, de Souza PRB Jr, Neri AL, Duarte YAO, Castro-Costa E,
      de Oliveira C. The Brazilian Longitudinal Study of Aging (ELSI-Brazil): Objectives
      and Design. Am J Epidemiol. 2018 Jul 1;187(7):1345-1353. doi:10.1093/aje/kwx387
      (1st wave)

  [2] Lima-Costa MF, de Melo Mambrini JV, Bof de Andrade F, de Souza PRB,
      de Vasconcellos MTL, Neri AL, Castro-Costa E, Macinko J, de Oliveira C.
      Cohort Profile: The Brazilian Longitudinal Study of Ageing (ELSI-Brazil).
      Int J Epidemiol. 2023 Feb 8;52(1):e57-e65 doi:10.1093/ije/dyac132
      (2nd wave)

  so it can be loaded to and handled by the Polygrid CLI environment. We thank the authors
  for sharing the data. This script converts participant answers into intrinsic capacity
  domain scores according to the method employed in:

  [3] Márlon J.R. Aliberti, Laiss Bertola, Claudia Szlejf, Déborah Oliveira, Ronaldo D.
      Piovezan, Matteo Cesari, Fabíola Bof de Andrade, Maria Fernanda Lima-Costa, Monica
      Rodrigues Perracini, Cleusa P. Ferri, Claudia K. Suemoto,
      "Validating intrinsic capacity to measure healthy aging in an upper middle-income
      country: Findings from the ELSI-Brazil.", The Lancet Regional Health - Americas,
      Volume 12, 2022, https://doi.org/10.1016/j.lana.2022.100284

  This script creates five datasets, based on the data just described. The first one is
  a multiclass dataset in which each participant is assigned to a class determined by
  their Katz index [4]. The ELSIO1-data dataset has 7175 instances.

  [4] Katz S, Ford AB, Moskowitz RW, Jackson BA, Jaffe MW.
      "Studies of Illness in the Aged: The Index of ADL: A Standardized Measure of
      Biological and Psychosocial Function. JAMA. 1963;185(12):914–919.
      doi:10.1001/jama.1963.03060120024016

  The other four datasets are copies of the previous one except for their assignments,
  which aim to simulate distributional characteristics that were observed in the
  healthcare datasets that are cited below. The assignments in two of these datasets aim
  to simulate characteristics observed in a sample of the data that was collected for
  the article:

  [5] Andrade SC, Marcucci RM, Faria LF, Paschoal SM, Rebustini F, Melo RC.
      Health profile of older adults assisted by the Elderly Caregiver Program of
      the Health Care Network of the City of São Paulo. Einstein (São Paulo). 2020;
      18:eAO5263. http://dx.doi.org/10.31744/einstein_journal/2020AO5263

      AMPI-AB dataset, only instances with referrals informed by the attending
      professional:
        instances          128
        labels              11
        labelsets           15
        single labelsets     8
        cardinality       1.08
        density           0.10

  The other two aim to simulate characteristics of the data described in the Table 3
  of the following article:

  [6] Tavassoli, Neda, Philipe de Souto Barreto, Caroline Berbon, Celine Mathieu,
      Justine de Kerimel, Christine Lafont, Catherine Takeda et al. "Implementation of
      the WHO integrated care for older people (ICOPE) programme in clinical practice:
      a prospective study." The Lancet Healthy Longevity 3, no. 6 (2022): e394-e404.
      https://doi.org/10.1016/S2666-7568(22)00097-6

      ICOPE-FR dataset; these values are inferred from data in Table 3:
        instances          958
        labels              22
        labelsets            ?
        single labelsets     ?
        cardinality       4.54 (as computed by the toulouse.py script)
        density           0.21 (as computed by the toulouse.py script)

  These datasets are created as Bunch objects (same API used in scikit-learn) and are
  serialised in the following files:

  - ELSIO1-data.pkl
    This stores a multiclass dataset, with 7175 samples. It has the domains of the
    WHOIC instrument as features, sex, age, and skin colour as factors, and the target
    corresponds to one of the levels of the Katz score, as described in [4].

  - ELSIO1-ml-11-data.pkl (11 labels)
  - ELSIO1-ml-22-data.pkl (22 labels)
    These store multilabel datasets with 7175 instances, with the same features and
    factors as the previous one, but its labels were obtained from applying fuzzy
    clustering to the features data.

  - ELSIO1-lr-11-data.pkl (11 labels)
  - ELSIO1-lr-22-data.pkl (22 labels)
    These store label ranking datasets with 7175 instances, with the same features and
    factors as the previous ones, but its labels were obtained from applying fuzzy
    clustering to the features data.
"""

import re
import sys
import numpy as np
import pyreadstat as pr
import pandas as pd

from os            import listdir, makedirs, remove
from os.path       import join, exists
from itertools     import chain
from collections   import defaultdict
from numpy.random  import shuffle
from sklearn.utils import Bunch
from customDefs    import getMountedOn, setupEssayConfig, getEssayParameter
from customDefs    import tsprint, stimestamp, saveAsText, saveLog, serialise
from readerDefs    import createView, build_factor_matrix, build_feature_matrix
from readerDefs    import updateAssignments, createMetaDict

from customDefs    import ECO_CAPACITY
from datasets      import ECO_ASSIGNMENT_F, ECO_ASSIGNMENT_R
from datasets      import ECO_DB_MULTILABEL, ECO_DB_LABELRANK

def validate(row):
  # NA stants for 'nao se aplica' and we assume it encodes an invalid answer, which we
  #    believe was the procedure followed by [3] to discard assessments in [1];
  #    evidence that this assumption is correct: we obtain the same number of
  #    participants in [3]
  # NS stands for 'nao sabe' (the participant didn't know the answer)
  # NR stands for 'nao respondeu' (the participant didn't answer the question)
  if(  row['q7'] == 8 or row['q8'] == 8 or row['q9'] == 8 or row['q10'] == 8):
    # 8-NA
    valid = 'no, issue with q7-q10 (cognitive domain)'

  elif(row['q13'] == 88):
    valid = 'no, issue with q13 (cognitive domain)'

  elif(row['q18'] == 8 or row['q19'] == 8 or row['q20'] == 8 or row['q21'] == 8):
    # 9-NS, 10-NR will be mapped to 2 (incorrect)
    valid = 'no, issue with q18-q21 (cognitive domain)'

  elif(row['q14'] == 888):
    # 999-NR will be mapped to 2 (incorrect)
    valid = 'no, issue with q14 (cognitive domain)'

  elif(row['mf33'] in [9666, 9888] or row['mf36'] in [9666, 9888]):
    valid = 'no, issue with mf33-mf38 (locomotor domain)'

  elif(row['mf30'] in [9666, 9888, 9999] or
       row['mf31'] in [9666, 9888, 9999] or
       row['mf32'] in [9666, 9888, 9999]):
    valid = 'no, issue with mf30-mf32 (locomotor domain)'

  elif(row['r2'] in [8, 9] or row['r3'] in [8, 9] or row['r4'] in [8, 9] or
       row['r5'] in [8, 9] or row['r6'] in [8, 9] or row['r7'] in [8, 9] or
       row['r8'] in [8, 9] or row['r9'] in [8, 9]):
    valid = 'no, issue with r2-r9 (psychological domain)'

  elif(row['n74'] == 9 or row['n75'] == 9):
    valid = 'no, issue with n74-n75 (psychological domain)'

  elif(row['n16'] == 9 or row['n6'] == 9 or row['n7'] == 9):
    valid = 'no, issue with n6-n7, n16 (sensory domain)'

  elif(row['mf27'] in [8888, 9555, 9666, 9777, 9888] or
       row['mf28'] in [8888, 9555, 9666, 9777, 9888] or
       row['mf29'] in [8888, 9555, 9666, 9777, 9888]):
    valid = 'no, issue with mf27-mf29 (vitality domain)'

  elif(row['n69'] == 1 and row['n70'] == 99999):
    # n69 == 9-NS/NR will be mapped to 0-No
    valid = 'no, issue with n69 (vitality domain)'

  elif(row['n72'] == 9 or row['n73'] == 9 ):
    valid = 'no, issue with n72-n73 (vitality domain)'

  elif(#row['p40'] == 9 or
       #row['p43'] == 9 or
       #row['p46'] == 9 or #-2, asym
       #row['p49'] == 9 or #-2, repeated
       row['p55'] == 9 or #-1, asym
       #row['p58'] == 9 or #-1, repeated
       False
      ):
    valid = 'no, issue with p40, p43, p46, p49, p55, p58 (ABVDs)'

  elif(row['p41'] == 9 or
       row['p44'] == 9 or
       row['p47'] == 9 or
       row['p49'] == 9 or #-2
       #row['p56'] == 9 or
       row['p58'] == 9 or #-1
       False
       ):
    valid = 'no, issue with p41, p44, p47, p49, p56, p58 (ABVDs)'

  else:
    valid = 'yes'

  return valid

def katzIndex(row):

  dressing     = 1 if (row['p40'], row['p41']) in [(1,8), (2,0), (2,1), (3,0), (3,1)] else 0
  bathing      = 1 if (row['p43'], row['p44']) in [(1,8), (2,0), (2,1), (3,0), (3,1)] else 0
  feeding      = 1 if (row['p46'], row['p47']) in [(1,8), (2,0), (2,1), (3,0), (3,1)] else 0
  toileting    = 1 if (row['p55'], row['p56']) in [(1,8), (2,0), (2,1), (3,0), (3,1)] else 0
  transferring = 1 if  row['p49'] in [1, 2, 3] else 0
  continence   = 1 if  row['p58'] == 1         else 0
  score = bathing + dressing + toileting + feeding + transferring + continence

  return score

def get_factor_labels(row, meta, factor_names, misses):
  res = []
  for variable in factor_names:
    value = row[variable]
    try:
      if(variable == 'idade'):
        if(value < 60):
          label = 'less_than_60'
        elif(60 <= value < 75):
          label = '60-74'
        elif(75 <= value < 90):
          label = '75-89'
        else:
          label = '90_or_more'
      else:
        label = meta.variable_value_labels[variable][value]
    except KeyError:
      if((variable, str(value)) not in misses):
        misses.append((variable, str(value)))
      label = 'missing'
    res.append(label)
  res = ''.join([f'/{label.lower()}' for label in res]) + '/'
  return res

def whoic(row, domain):
  # computes domain scores of the WHO Intrinsic Capacity proposed instrument

  if(domain == 'cognitive'):
    # computes the cognitive domain score
    score = (row['q7'] + row['q8'] + row['q9'] + row['q10'] + # get current date right
             (1 if row['q13'] >= 3 else 0) + # q13 - recalled 3+ words out of 10
             (1 if row['q18'] == 1 else 0) + # q18 - scissors
             (1 if row['q19'] == 1 else 0) + # q19 - banana plant
             (1 if row['q20'] == 1 else 0) + # q20 - current President
             (1 if row['q21'] == 1 else 0) + # q21 - current Vice President
             (1 if row['q14'] >= 8 else 0))  # q14, recalled 8+ animals in 60sec

  elif(domain == 'psychological'):
    # computes the psychological domain score
    # -- depression assessed using the CES-D scale: for r2-r9, 0:No, 1:Yes
    # -- the items are inverted because WHO IC assesses capacity, not deficit
    # -- note that r5 and r7 are positively framed questions, and the rest if the opposite
    depression = ((1 if row['r2'] == 0 else 0)  +  # -r2 - felt depressed most of last week
                  (1 if row['r3'] == 0 else 0)  +  # -r3 - felt difficulty to do things
                  (1 if row['r4'] == 0 else 0)  +  # -r4 - felts sleep not resting enough
                  (1 if row['r5'] == 1 else 0)  +  # +r5 - felt happy most of last week
                  (1 if row['r6'] == 0 else 0)  +  # -r6 - felt lonly most of last week
                  (1 if row['r7'] == 1 else 0)  +  # +r7 - felt pleasure in living
                  (1 if row['r8'] == 0 else 0)  +  # -r8 - felt sad most of last week
                  (1 if row['r9'] == 0 else 0))    # -r9 - felt not able to continue
    score = (depression + # inverted CES-D score
             (1 if row['n74'] <= 3 else 0) + # n74, perceived quality of sleep (regular or >)
             (1 if row['n75'] <  3 else 0))  # n75, taking pils to sleep (once per week or less)

  elif(domain == 'sensory'):
    # computes the sensory domain score
    score = ((1 if row['n16'] <= 3 else 0) + # n16
             (1 if row['n6']  <= 3 else 0) + # n6
             (1 if row['n7']  <= 3 else 0))  # n7

  elif(domain == 'locomotor'):
    # computes the locomotor domain score
    # -- mf33:35, mf36:38, data from walk 3 meters test
    # -- mf33, mf36: minutes; mf34, mf37: seconds, mf35, mf38: centesimals
    # -- mttc_walk3m is the mean time to walk 3m (in sec)
    mttc_walk3m  = (np.mean([row['mf33'], row['mf36']]) * 60 + #
                    np.mean([row['mf34'], row['mf37']]) +
                    np.mean([row['mf35'], row['mf38']]) / 100)

    # using rules adapted from:
    # https://lermagazine.com/article/self-selected-gait-speed-a-critical-clinical-outcome
    gaitspeed = 3/mttc_walk3m # in m/s
    gaitscore = (0 if gaitspeed < 0.4 else # mttc_walk3m > 7.5
                 1 if gaitspeed < 0.6 else # mttc_walk3m > 5.0
                 2 if gaitspeed < 1.0 else # mttc_walk3m > 3.0
                 3)

    # mf30:32, data from balance test
    # mf30: time the participant kept balance with feet side by side, in seconds (10s max)
    # mf31: time the participant kept balance with one foot ahead, in seconds (10s max)
    # mf32: time the participant kept balance with aligned feet, in seconds
    #       (10s max if participant 70+, 30s otherwise)
    balancescore = ( min(row['mf30'] / 10, 1) +
                     min(row['mf31'] / 10, 1) +
                    (min(row['mf32'] / 10, 1) if row['m1'] >= 70 else min(row['mf32'] / 30, 1)))

    score = gaitscore + balancescore

  elif(domain == 'vitality'):
    # computes the vitality domain score
    # -- mf27:29, mean reading from 3x handgrip strength test
    strength = np.mean([row['mf27'], row['mf28'], row['mf29']])

    gender = 'female' if row['m1'] == 0 else 'male'
    # -- mf22: weight measured during interview, mf24: participant informed weight
    # -- mf13: height measured during interview, mf15> participant informed height
    weight = row['mf22'] if row['mf22'] != 99999 else (row['mf24'] if row['mf24'] != 8 else np.nan)
    height = row['mf13'] if row['mf13'] != 99999 else (row['mf15'] if row['mf15'] != 8 else np.nan)
    bmi    = weight/height**2
    if(np.isnan(bmi)):
      # BMI is unknown: assuming measurements were not obtained due to locomotor problems
      #                 selecting the highest level of restriction from:
      # https://www.scielo.br/j/rbepid/a/dhZVDQWSSkkLCWcS6KDZGVp/?lang=en
      if(gender == 'female'):
        gripscore = 0 if strength <= 23.0 else 1
      else:
        gripscore = 0 if strength <= 37.0 else 1
    else:
      # BMI is known: using rules adapted from:
      # https://www.scielo.br/j/rbepid/a/dhZVDQWSSkkLCWcS6KDZGVp/?lang=en
      if(gender == 'female'):
        gripscore = (0 if (strength <= 14.0 and                 bmi <= 23.80) or
                          (strength <= 17.0 and 23.80 < bmi and bmi <= 27.10) or
                          (strength <= 20.0 and 27.10 < bmi and bmi <= 30.80) or
                          (strength <= 23.0 and 30.80 < bmi                 )
                        else 1)
      else:
        gripscore = (0 if (strength <= 21.0 and                 bmi <= 23.12) or
                          (strength <= 25.5 and 23.12 < bmi and bmi <= 25.50) or
                          (strength <= 30.0 and 25.50 < bmi and bmi <= 28.08) or
                          (strength <= 27.0 and 28.08 < bmi                 )
                        else 1)

    #-- n69: unintentional loss of weight last 3mo; 0-No, 1-Yes, 9-don't recall
    #-- n70: how much weight did you lose in the last 3mo? in Kg
    no_unint_weight_loss = (1 if row['n69'] in [0, 9] or
                            (row['n69'] == 1 and row['n70'] < 3) else 0)

    #-- n72: how often found difficulty in carrying on last week?
    #-- n73: how often found routine activities take too much effort last week?
    no_fatigue = 1 if row['n72'] <= 2 else 0
    endurance  = 1 if row['n73'] <= 2 else 0

    score = gripscore + no_unint_weight_loss + no_fatigue + endurance

  return score

def build_feature_matrix_row(df, domains, domain):
  return df.apply(lambda row: whoic(row, domain), axis=1).to_numpy().astype(np.float64)

def convert2Bunch_multiclass(df, meta, descr, subset, domains):

  (factors, features, outcomes, extra) = subset

  ds = df.loc[(df['valid'] == 'yes')]
  caseIDs = ds.index.to_numpy().tolist()
  outcome = 'katz'

  # builds the factor matrix Z
  (factor_names, Z, misses) = build_factor_matrix(ds, meta, factors, get_factor_labels)
  for (variable, value) in misses:
    tsprint(f'** Unexpected level ({value}) for factor {variable}')

  # builds the feature matrix X
  (feature_names, X) = build_feature_matrix(ds, domains, build_feature_matrix_row)
  assert (np.corrcoef(X, rowvar=False) < 0).sum() == 0

  # builds the target matrix Y
  n = ds[outcome].unique().size
  target_names = [str(e) for e in sorted(ds[outcome].unique().tolist())]
  Y = np.array([target_names.index(str(e)) for e in ds[outcome]], dtype=int)
  assert np.isnan(Y).sum() == 0

  # samples the data so we can reduce the size of the dataset
  # ensures the minority classes are represented
  class2inst = defaultdict(list)
  for (i,j) in enumerate(Y):
    class2inst[j].append(i)

  shuffle(class2inst[3])
  shuffle(class2inst[4])
  shuffle(class2inst[5])

  sao  = class2inst[0] + class2inst[1] + class2inst[2]
  sao += class2inst[3][:50]
  sao += class2inst[4][:80]
  sao += class2inst[5][:566]
  shuffle(sao)

  # builds the dataset
  dataset = Bunch(
      caseIDs = [caseID for (i,caseID) in enumerate(caseIDs) if i in sao],
      data    = X[sao],
      target  = Y[sao],
      factors = Z[sao],
      feature_names = feature_names,
      target_names  = target_names,
      factor_names  = factor_names,
      symbol  = ECO_CAPACITY,
      DESCR   = descr,
  )

  # creates a report showing characteristics of the new dataset
  report = []
  (nrows, ncols) = dataset.data.shape
  report.append(f'feature matrix has {nrows} rows and {ncols} columns')
  report.append(f'feature names are {dataset.feature_names}')
  report.append(f'target  names are {dataset.target_names}')
  report.append(f'factor  names are {dataset.factor_names}')

  return (dataset, report)

def main(configfile):

  tsprint(__doc__)
  tsprint(f'Loading parameters from {configfile}')

  setupEssayConfig(configfile)
  sourcepath  = getEssayParameter('PARAM_ELSIO1_SOURCEPATH')
  targetpath  = getEssayParameter('PARAM_ELSIO1_TARGETPATH')
  filename    = getEssayParameter('PARAM_ELSIO1_FILENAME')
  fuzzfactors = getEssayParameter('PARAM_ELSIO1_FUZZFACTOR')
  basename    = filename.split('.')[0].lower()

  # ensures the target folder is available and empty
  if(exists(join(*targetpath))):
    for f in listdir(join(*targetpath)):
      remove(join(*targetpath, f))
  else:
    makedirs(join(*targetpath))

  # reads the dataset file
  tsprint(f'Reading dataset file {filename} from {join(*sourcepath)}')
  (df, meta) = pr.read_dta(join(*sourcepath, filename))

  # reproduces a couple of statistics published in [1], Table 1
  tsprint('-- comparing some dataset statistics with the published data in [1], Table 1:')
  tsprint('  ---------- statistic ----------  \tArticle Dataset')
  tsprint('  # of participants ............. :\t  {0:>5d}\t  {1:>5d}'.format(9412, len(df.index)))
  tsprint('  % of women .................... :\t  {0:2.1f}%\t  {1:2.1f}%'.format(54.0, 100 * (1 - (df['peso_calibrado'] * df['m1']).mean())))
  tsprint('  average age ................... :\t  {0:2.1f}%\t  {1:2.1f}%'.format(62.9, (df['peso_calibrado'] * df['idade']).mean()))

  # creates the metadata dictionary of the dataset
  filename = f'{basename}-meta.csv'
  content = createMetaDict(meta)
  saveAsText('\n'.join(content), join(*targetpath, filename))

  # creates a view with the columns that will be used ahead in pipeline
  tsprint('')
  tsprint('Creating the data view (only fields/columns what will be used)')
  domains = {
             'cognitive':     ['q7', 'q8', 'q9', 'q10', 'q13', 'q18', 'q19', 'q20', 'q21', 'q14',],
             'psychological': ['r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'n74', 'n75',],
             'sensory':       ['n16', 'n6', 'n7',],
             'locomotor':     ['mf33', 'mf34', 'mf35', 'mf36', 'mf37', 'mf38', 'mf30', 'mf31', 'mf32',],
             'vitality':      ['mf27', 'mf28', 'mf29', 'n69', 'n70', 'n72', 'n73',
                               'm1', 'mf22', 'mf24', 'mf13', 'mf15' # gender, weight and height
                              ],
  }
  features = sorted(chain(*domains.values()))
  # although m1:gender is a factor, it will be recovered as a feature for now
  factors  = ['idade', 'e9'] # e9:cor da pele
  outcomes = ['p40', 'p41', 'p43', 'p44', 'p46', 'p47', 'p49', 'p55', 'p56', 'p58',] # ADLs

  # extends the dataset with new columns
  df['valid'] = df.apply(lambda row:  validate(row), axis=1)
  df['katz']  = df.apply(lambda row: katzIndex(row), axis=1)
  extra = ['katz', 'valid']

  subset   = (factors, features, outcomes, extra)
  filename = f'{basename}-view.csv'
  (content, df) = createView(df, subset)
  saveAsText('\n'.join(content), join(*targetpath, filename))
  tsprint(f'-- selected factors : {factors}')
  tsprint(f'-- selected features: {features}')
  tsprint( '   (although m1:gender is a factor, it will be processed as a feature for now)')
  tsprint(f'-- selected outcomes: {outcomes}')
  tsprint(f'-- view with selected columns saved as {join(*targetpath, filename)}')
  tsprint(f'-- view has {len(df.index)} rows')

  for domain in domains:
    tsprint(f'-- Domain: {domain}')
    for item in sorted(domains[domain]):
      tsprint(f'   {item:<4}\t{meta.column_names_to_labels[item]}')

  # now that the original data has been selected, we introduce the missing factor
  # that was recovered as a feature in the data view above
  factors.append('m1')

  #--------------------------------------------------------------------------------------
  # creates a multiclass dataset
  #--------------------------------------------------------------------------------------
  descr = """
   Dataset FioCruz ELSIO1
   ----------------------

   Dataset ELSI-Brazil - Estudo Longitudinal da Saúde dos Idosos Brasileiros
   This dataset contains health assessments of a nation-wide systematic sample of 50+yrs
   individuals in Brazil. Interviews were conducted as part of the Fiocruz's ELSI-Brazil
   project:

   [1] Lima-Costa MF, de Andrade FB, de Souza PRB Jr, Neri AL, Duarte YAO, Castro-Costa E,
       de Oliveira C. The Brazilian Longitudinal Study of Aging (ELSI-Brazil): Objectives
       and Design. Am J Epidemiol. 2018 Jul 1;187(7):1345-1353. doi: 10.1093/aje/kwx387.
       (1st wave)

   [2] Lima-Costa MF, de Melo Mambrini JV, Bof de Andrade F, de Souza PRB,
       de Vasconcellos MTL, Neri AL, Castro-Costa E, Macinko J, de Oliveira C.
       Cohort Profile: The Brazilian Longitudinal Study of Ageing (ELSI-Brazil).
       Int J Epidemiol. 2023 Feb 8;52(1):e57-e65 doi: 10.1093/ije/dyac132.
       (2nd wave)

   (you must register to gain access to it; see https://elsi.cpqrr.fiocruz.br/en/register/)

   The process that converts participant answers into intrinsic capacity scores follows
   the methodology described in:

   [3] Márlon J.R. Aliberti, Laiss Bertola, Claudia Szlejf, Déborah Oliveira, Ronaldo D.
       Piovezan, Matteo Cesari, Fabíola Bof de Andrade, Maria Fernanda Lima-Costa, Monica
       Rodrigues Perracini, Cleusa P. Ferri, Claudia K. Suemoto,
       "Validating intrinsic capacity to measure healthy aging in an upper middle-income
       country: Findings from the ELSI-Brazil.", The Lancet Regional Health - Americas,
       Volume 12, 2022, https://doi.org/10.1016/j.lana.2022.100284

   Each participant is assigned to a class determined by their Katz index [4]:

   [4] Katz S, Ford AB, Moskowitz RW, Jackson BA, Jaffe MW.
       "Studies of Illness in the Aged: The Index of ADL: A Standardized Measure of
       Biological and Psychosocial Function. JAMA. 1963;185(12):914–919.
       doi:10.1001/jama.1963.03060120024016
  """
  tsprint('')
  tsprint(f'Creating a multiclass dataset based on {basename} data, with 6 labels')
  filename = f'{basename}-data'
  (dataset, report) = convert2Bunch_multiclass(df, meta, descr, subset, domains)
  serialise(dataset, join(*targetpath, filename))
  report.append(f'dataset saved as {join(*targetpath, filename)}.pkl')
  tsprint('\n'.join([f'[{stimestamp()}] -- {msg}' for msg in report]), stamp=False)
  #assert (dataset.data.shape[0] == 7175) # according to data published in [3]

  #--------------------------------------------------------------------------------------
  # records distributional characteristics observed in [5, 6], as well as the parameters
  # we use to simulate them based on the feature data of the the AMPI-AB dataset
  #--------------------------------------------------------------------------------------
  observed  = {11: {'data': 'AMPI-AB',  'cardinality': 1.08, 'density': 0.10},
               22: {'data': 'ICOPE-FR', 'cardinality': 4.54, 'density': 0.21},
              }

  simulated = {11: {'maxcareplan': 2,    'fuzzfactor': fuzzfactors[11], },
               22: {'maxcareplan': None, 'fuzzfactor': fuzzfactors[22], },
              }

  #--------------------------------------------------------------------------------------
  # creates a multilabel version of the dataset, with 11 labels
  #--------------------------------------------------------------------------------------
  descr = """
   Dataset FioCruz ELSIO1-ml-11
   ----------------------------

   Dataset ELSI-Brazil - Estudo Longitudinal da Saúde dos Idosos Brasileiros
   This dataset contains health assessments of a nation-wide systematic sample of 50+yrs
   individuals in Brazil. Interviews were conducted as part of the Fiocruz's ELSI-Brazil
   project:

   [1] Lima-Costa MF, de Andrade FB, de Souza PRB Jr, Neri AL, Duarte YAO, Castro-Costa E,
       de Oliveira C. The Brazilian Longitudinal Study of Aging (ELSI-Brazil): Objectives
       and Design. Am J Epidemiol. 2018 Jul 1;187(7):1345-1353. doi: 10.1093/aje/kwx387.
       (1st wave)

   [2] Lima-Costa MF, de Melo Mambrini JV, Bof de Andrade F, de Souza PRB,
       de Vasconcellos MTL, Neri AL, Castro-Costa E, Macinko J, de Oliveira C.
       Cohort Profile: The Brazilian Longitudinal Study of Ageing (ELSI-Brazil).
       Int J Epidemiol. 2023 Feb 8;52(1):e57-e65 doi: 10.1093/ije/dyac132.
       (2nd wave)

   (you must register to gain access to it; see https://elsi.cpqrr.fiocruz.br/en/register/)

   The process that converts participant answers into intrinsic capacity scores follows
   the methodology described in:

   [3] Márlon J.R. Aliberti, Laiss Bertola, Claudia Szlejf, Déborah Oliveira, Ronaldo D.
       Piovezan, Matteo Cesari, Fabíola Bof de Andrade, Maria Fernanda Lima-Costa, Monica
       Rodrigues Perracini, Cleusa P. Ferri, Claudia K. Suemoto,
       "Validating intrinsic capacity to measure healthy aging in an upper middle-income
       country: Findings from the ELSI-Brazil.", The Lancet Regional Health - Americas,
       Volume 12, 2022, https://doi.org/10.1016/j.lana.2022.100284

   Each participant is assigned to one or two labels among 11 labels. This arrangement
   aims to simulate distributional characteristics observed in a sample of the data
   that was collected for the article:

   [5] Andrade SC, Marcucci RM, Faria LF, Paschoal SM, Rebustini F, Melo RC.
       Health profile of older adults assisted by the Elderly Caregiver Program of
       the Health Care Network of the City of São Paulo. Einstein (São Paulo). 2020;
       18:eAO5263. http://dx.doi.org/10.31744/einstein_journal/2020AO5263

       AMPIAB dataset, only instances with referrals informed by the attending healthcare
       professional:
         instances          128
         labels              11
         labelsets           15
         single labelsets     8
         cardinality       1.08
         density           0.10
  """
  (scenario, infix, ngroups) = (ECO_DB_MULTILABEL, 'ml', 11)
  tsprint('')
  tsprint(f'Creating a {scenario} dataset based on {basename} data, with {ngroups} labels')
  filename = f'{basename}-{infix}-{ngroups}-data'

  (dataset, report) = updateAssignments(dataset, descr, scenario, ngroups,
                                        observed, simulated)
  serialise(dataset, join(*targetpath, filename))
  report.append(f'dataset saved as {join(*targetpath, filename)}.pkl')
  tsprint('\n'.join([f'[{stimestamp()}] -- {msg}' for msg in report]), stamp=False)

  #--------------------------------------------------------------------------------------
  # creates a multilabel version of the dataset, with 22 labels
  #--------------------------------------------------------------------------------------
  descr = """
   Dataset FioCruz ELSIO1-ml-22
   ----------------------------

   Dataset ELSI-Brazil - Estudo Longitudinal da Saúde dos Idosos Brasileiros
   This dataset contains health assessments of a nation-wide systematic sample of 50+yrs
   individuals in Brazil. Interviews were conducted as part of the Fiocruz's ELSI-Brazil
   project:

   [1] Lima-Costa MF, de Andrade FB, de Souza PRB Jr, Neri AL, Duarte YAO, Castro-Costa E,
       de Oliveira C. The Brazilian Longitudinal Study of Aging (ELSI-Brazil): Objectives
       and Design. Am J Epidemiol. 2018 Jul 1;187(7):1345-1353. doi: 10.1093/aje/kwx387.
       (1st wave)

   [2] Lima-Costa MF, de Melo Mambrini JV, Bof de Andrade F, de Souza PRB,
       de Vasconcellos MTL, Neri AL, Castro-Costa E, Macinko J, de Oliveira C.
       Cohort Profile: The Brazilian Longitudinal Study of Ageing (ELSI-Brazil).
       Int J Epidemiol. 2023 Feb 8;52(1):e57-e65 doi: 10.1093/ije/dyac132.
       (2nd wave)

   (you must register to gain access to it; see https://elsi.cpqrr.fiocruz.br/en/register/)

   The process that converts participant answers into intrinsic capacity scores follows
   the methodology described in:

   [3] Márlon J.R. Aliberti, Laiss Bertola, Claudia Szlejf, Déborah Oliveira, Ronaldo D.
       Piovezan, Matteo Cesari, Fabíola Bof de Andrade, Maria Fernanda Lima-Costa, Monica
       Rodrigues Perracini, Cleusa P. Ferri, Claudia K. Suemoto,
       "Validating intrinsic capacity to measure healthy aging in an upper middle-income
       country: Findings from the ELSI-Brazil.", The Lancet Regional Health - Americas,
       Volume 12, 2022, https://doi.org/10.1016/j.lana.2022.100284

   Each participant is assigned to at least one label among 22 labels. This arrangement
   aims to simulate distributional characteristics observed in a sample of the data that
   was collected for the article:

   [6] Tavassoli, Neda, Philipe de Souto Barreto, Caroline Berbon, Celine Mathieu,
       Justine de Kerimel, Christine Lafont, Catherine Takeda et al. "Implementation of
       the WHO Integrated Care for Older People (ICOPE) programme in clinical practice:
       a prospective study." The Lancet Healthy Longevity 3, no. 6 (2022): e394-e404.
       https://doi.org/10.1016/S2666-7568(22)00097-6

       ICOPE-FR dataset; these values are inferred from data in Table 3 of the
       article:
         instances          958
         labels              22
         labelsets            ?
         single labelsets     ?
         cardinality       4.54
         density           0.21
  """
  (scenario, infix, ngroups) = (ECO_DB_MULTILABEL, 'ml', 22)
  tsprint('')
  tsprint(f'Creating a {scenario} dataset based on {basename} data, with {ngroups} labels')
  filename = f'{basename}-{infix}-{ngroups}-data'

  (dataset, report) = updateAssignments(dataset, descr, scenario, ngroups,
                                        observed, simulated)
  serialise(dataset, join(*targetpath, filename))
  report.append(f'dataset saved as {join(*targetpath, filename)}.pkl')
  tsprint('\n'.join([f'[{stimestamp()}] -- {msg}' for msg in report]), stamp=False)

  #--------------------------------------------------------------------------------------
  # creates a label ranking version of the dataset, with 11 labels
  #--------------------------------------------------------------------------------------
  descr = """
   Dataset FioCruz ELSIO1-lr-11
   ----------------------------

   Dataset ELSI-Brazil - Estudo Longitudinal da Saúde dos Idosos Brasileiros
   This dataset contains health assessments of a nation-wide systematic sample of 50+yrs
   individuals in Brazil. Interviews were conducted as part of the Fiocruz's ELSI-Brazil
   project:

   [1] Lima-Costa MF, de Andrade FB, de Souza PRB Jr, Neri AL, Duarte YAO, Castro-Costa E,
       de Oliveira C. The Brazilian Longitudinal Study of Aging (ELSI-Brazil): Objectives
       and Design. Am J Epidemiol. 2018 Jul 1;187(7):1345-1353. doi: 10.1093/aje/kwx387.
       (1st wave)

   [2] Lima-Costa MF, de Melo Mambrini JV, Bof de Andrade F, de Souza PRB,
       de Vasconcellos MTL, Neri AL, Castro-Costa E, Macinko J, de Oliveira C.
       Cohort Profile: The Brazilian Longitudinal Study of Ageing (ELSI-Brazil).
       Int J Epidemiol. 2023 Feb 8;52(1):e57-e65 doi: 10.1093/ije/dyac132.
       (2nd wave)

   (you must register to gain access to it; see https://elsi.cpqrr.fiocruz.br/en/register/)

   The process that converts participant answers into intrinsic capacity scores follows
   the methodology described in:

   [3] Márlon J.R. Aliberti, Laiss Bertola, Claudia Szlejf, Déborah Oliveira, Ronaldo D.
       Piovezan, Matteo Cesari, Fabíola Bof de Andrade, Maria Fernanda Lima-Costa, Monica
       Rodrigues Perracini, Cleusa P. Ferri, Claudia K. Suemoto,
       "Validating intrinsic capacity to measure healthy aging in an upper middle-income
       country: Findings from the ELSI-Brazil.", The Lancet Regional Health - Americas,
       Volume 12, 2022, https://doi.org/10.1016/j.lana.2022.100284

   Each participant is assigned to one or two labels among 11 labels. This arrangement
   aims to simulate distributional characteristics observed in a sample of the data
   that was collected for the article:

   [5] Andrade SC, Marcucci RM, Faria LF, Paschoal SM, Rebustini F, Melo RC.
       Health profile of older adults assisted by the Elderly Caregiver Program of
       the Health Care Network of the City of São Paulo. Einstein (São Paulo). 2020;
       18:eAO5263. http://dx.doi.org/10.31744/einstein_journal/2020AO5263

       AMPIAB dataset, only instances with referrals informed by the attending healthcare
       professional:
         instances          128
         labels              11
         labelsets           15
         single labelsets     8
         cardinality       1.08
         density           0.10
  """
  (scenario, infix, ngroups) = (ECO_DB_LABELRANK, 'lr', 11)
  tsprint('')
  tsprint(f'Creating a {scenario} dataset based on {basename} data, with {ngroups} labels')
  filename = f'{basename}-{infix}-{ngroups}-data'

  (dataset, report) = updateAssignments(dataset, descr, scenario, ngroups,
                                        observed, simulated)
  serialise(dataset, join(*targetpath, filename))
  report.append(f'dataset saved as {join(*targetpath, filename)}.pkl')
  tsprint('\n'.join([f'[{stimestamp()}] -- {msg}' for msg in report]), stamp=False)

  #--------------------------------------------------------------------------------------
  # creates a label ranking version of the dataset, with 22 labels
  #--------------------------------------------------------------------------------------
  descr = """
   Dataset FioCruz ELSIO1-lr-22
   ----------------------------

   Dataset ELSI-Brazil - Estudo Longitudinal da Saúde dos Idosos Brasileiros
   This dataset contains health assessments of a nation-wide systematic sample of 50+yrs
   individuals in Brazil. Interviews were conducted as part of the Fiocruz's ELSI-Brazil
   project:

   [1] Lima-Costa MF, de Andrade FB, de Souza PRB Jr, Neri AL, Duarte YAO, Castro-Costa E,
       de Oliveira C. The Brazilian Longitudinal Study of Aging (ELSI-Brazil): Objectives
       and Design. Am J Epidemiol. 2018 Jul 1;187(7):1345-1353. doi: 10.1093/aje/kwx387.
       (1st wave)

   [2] Lima-Costa MF, de Melo Mambrini JV, Bof de Andrade F, de Souza PRB,
       de Vasconcellos MTL, Neri AL, Castro-Costa E, Macinko J, de Oliveira C.
       Cohort Profile: The Brazilian Longitudinal Study of Ageing (ELSI-Brazil).
       Int J Epidemiol. 2023 Feb 8;52(1):e57-e65 doi: 10.1093/ije/dyac132.
       (2nd wave)

   (you must register to gain access to it; see https://elsi.cpqrr.fiocruz.br/en/register/)

   The process that converts participant answers into intrinsic capacity scores follows
   the methodology described in:

   [3] Márlon J.R. Aliberti, Laiss Bertola, Claudia Szlejf, Déborah Oliveira, Ronaldo D.
       Piovezan, Matteo Cesari, Fabíola Bof de Andrade, Maria Fernanda Lima-Costa, Monica
       Rodrigues Perracini, Cleusa P. Ferri, Claudia K. Suemoto,
       "Validating intrinsic capacity to measure healthy aging in an upper middle-income
       country: Findings from the ELSI-Brazil.", The Lancet Regional Health - Americas,
       Volume 12, 2022, https://doi.org/10.1016/j.lana.2022.100284

   Each participant is assigned to at least one label among 22 labels. This arrangement
   aims to simulate distributional characteristics observed in a sample of the data that
   was collected for the article:

   [6] Tavassoli, Neda, Philipe de Souto Barreto, Caroline Berbon, Celine Mathieu,
       Justine de Kerimel, Christine Lafont, Catherine Takeda et al. "Implementation of
       the WHO Integrated Care for Older People (ICOPE) programme in clinical practice:
       a prospective study." The Lancet Healthy Longevity 3, no. 6 (2022): e394-e404.
       https://doi.org/10.1016/S2666-7568(22)00097-6

       ICOPE-FR dataset; these values are inferred from data in Table 3 of the
       article:
         instances          958
         labels              22
         labelsets            ?
         single labelsets     ?
         cardinality       4.54
         density           0.21
  """
  (scenario, infix, ngroups) = (ECO_DB_LABELRANK, 'lr', 22)
  tsprint('')
  tsprint(f'Creating a {scenario} dataset based on {basename} data, with {ngroups} labels')
  filename = f'{basename}-{infix}-{ngroups}-data'

  (dataset, report) = updateAssignments(dataset, descr, scenario, ngroups,
                                        observed, simulated)
  serialise(dataset, join(*targetpath, filename))
  report.append(f'dataset saved as {join(*targetpath, filename)}.pkl')
  tsprint('\n'.join([f'[{stimestamp()}] -- {msg}' for msg in report]), stamp=False)


  # saves the execution log
  logfile = 'readelsi.log'
  tsprint('')
  tsprint('This report was saved to {0}'.format(join(*targetpath, logfile)))
  tsprint('Done.')
  saveLog(join(*targetpath, logfile))

if(__name__ == '__main__'):

  main(sys.argv[1])
