"""

  This script processes quality of life (QoL) assessment data that were collected by the
  Department of Gerontology at UFSCar/BR. As a result, the data can be loaded to and
  handled by the Polygrid CLI environment. We thank the authors for sharing the dataset.
  Details about when, where, and how the data collection was conducted can be found in
  these documents:

  [1] L. J. Lorenzi, P. A. R. Alvarez, P. Bet, P. C. Castro (2022). Digital engagement
      and quality of life of participants at a University of the Third Age.
      Gerontechnology, 21(s), 1-1, https://doi.org/10.4017/gt.2022.21.s.567.opp4

  [2] Termo de Consentimento Livre e Esclarecido para a pesquisa "Envelhecimento ativo na
      Universidade da Terceira Idade" (Protocolo de avaliação + TCLE.pdf)

  To convert participant answers into WHOQOL-BREF domain scores, the original methodology
  proposed by the WHO was employed:

  [3] World Health Organization. WHOQOL-BREF: Introduction, Administration, Scoring and
      generic version of the assessment: Field trial version, December 1996.
      World Health Organization. Division of Mental Health, 1996.
      (see Table 3)

  This script creates five datasets, based on the data just described. All assignments are
  synthetic. The first dataset is a multiclass dataset in which each participant is
  assigned to a class determined by the decision boundary WHOQOL score >= 60, following
  results from:

  [4] Silva PAB, Soares SM, Santos JFG, Silva LB. Cut-off point for WHOQOL-bref as a
      measure of quality of life of older adults. Rev Saúde Pública. 2014Jun; 48(3):390–7.
      https://doi.org/10.1590/S0034-8910.2014048004912

  The other four datasets are copies of the first one except for their assignments, which
  aim to simulate distributional characteristics that were observed in the healthcare
  datasets that are cited below. The assignments of two of these datasets aim to simulate
  characteristics observed in a sample of the data that was collected for the article:

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

  - WHOQOL-data.pkl
    This stores a multiclass dataset, with 100 samples. It has the domains of the
    WHOQOL instrument as features, sex, age, and education as factors, and the target
    corresponds to 'Good QOL' or 'Poor QOL', according to criterion in [4].

  - WHOQOL-ml-11-data.pkl (11 labels)
  - WHOQOL-ml-22-data.pkl (22 labels)
    These store multilabel datasets with 100 instances, with the same features and
    factors as the previous one, but its labels were obtained from applying fuzzy
    clustering to the features data.

  - WHOQOL-lr-11-data.pkl (11 labels)
  - WHOQOL-lr-22-data.pkl (22 labels)
    These store label ranking datasets with 100 instances, with the same features and
    factors as the previous ones, but its labels were obtained from applying fuzzy
    clustering to the features data.
"""

import re
import sys
import numpy as np
import pandas as pd

from os            import listdir, makedirs, remove
from os.path       import join, exists
from itertools     import chain
from sklearn.utils import Bunch

from customDefs    import getMountedOn, setupEssayConfig, getEssayParameter
from customDefs    import tsprint, stimestamp, saveAsText, saveLog, serialise
from datasets      import compute_split_stats
from readerDefs    import createView, build_factor_matrix, build_feature_matrix
from readerDefs    import updateAssignments

from customDefs    import ECO_CAPACITY
from datasets      import ECO_ASSIGNMENT_F, ECO_ASSIGNMENT_R
from datasets      import ECO_DB_MULTILABEL, ECO_DB_LABELRANK

def get_factor_labels(row, meta, factor_names, misses):
  res = []
  for variable in factor_names:
    value = row[variable]
    try:
      if(variable == 'Idade'):
        if(value < 60):
          label = 'less_than_60'
        elif(60 <= value < 75):
          label = '60-74'
        elif(75 <= value < 90):
          label = '75-89'
        else:
          label = '90_or_more'
      elif(variable == 'Gênero'):
        if(value == 'F'):
          label = 'feminino'
        elif(value == 'M'):
          label = 'masculino'
      elif(variable == 'A.1 Escolaridade'):
        if(str(value) == '99'):
          label = 'not informed'
        else:
          label = value
      else:
        raise KeyError
    except KeyError:
      if((variable, str(value)) not in misses):
        misses.append((variable, str(value)))
      label = 'missing'
    res.append(label)
  res = ''.join([f'/{label.lower()}' for label in res]) + '/'
  return res

def whoqol(row, domains, domain):
  # computes domain scores of the WHOQOL instrument
  # -- instruction: observe the minimum number of endorsed items to compute each domain score
  # -- solution ..: scores are computed one by one and checked
  # -- original instruction in [3], Table 3, and SPSS syntax (MEAN command):
  #
  #    COMPUTE DOM1=MEAN.6(Q3,Q4,Q10,Q15,Q16,Q17,Q18)*4.
  #    COMPUTE DOM2=MEAN.5(Q5,Q6,Q7,Q11,Q19,Q26)*4.
  #    COMPUTE DOM3=MEAN.2(Q20,Q21,Q22)*4.
  #    COMPUTE DOM4=MEAN.6(Q8,Q9,Q12,Q13,Q14,Q23,Q24,Q25)*4.

  if(domain != 'sumscore'):

    dom_answers = row[domains[domain]].to_numpy().astype(np.float64)
    endorsed = sum(~np.isnan(dom_answers))
    if(domain == 'physical health'):
      # computes the cognitive domain score
      score = 4 * np.nanmean(dom_answers) if endorsed >= 6 else np.nan

    elif(domain == 'psychological'):
      # computes the psychological domain score
      score = 4 * np.nanmean(dom_answers) if endorsed >= 5 else np.nan

    elif(domain == 'social relationships'):
      # computes the social relationships domain score
      score = 4 * np.nanmean(dom_answers) if endorsed >= 2 else np.nan

    elif(domain == 'environment'):
      # computes the environment domain score
      score = 4 * np.nanmean(dom_answers) if endorsed >= 6 else np.nan

  else:

    all_items = chain(*domains.values())
    missing = sum(np.isnan(row[all_items].to_numpy().astype(np.float64)))
    dom_scores = row[list(domains.keys())].to_numpy().astype(np.float64)
    if(missing > 5 or min(dom_scores) < 4. or max(dom_scores) > 20.):
      score = np.nan
    else:
      score = sum(dom_scores)

  return score

def build_feature_matrix_row(df, domains, domain):
  return df[domain].to_numpy().astype(np.float64)

def convert2Bunch_multiclass(df, meta, descr, subset, domains):
  meta=None # add this to the signature in the 2nd step

  (factors, features, outcomes, extra) = subset

  ds = df.dropna(subset=['sumscore'])
  caseIDs = ds['Identificação'].to_numpy().tolist()
  caseIDs = [caseID.replace('UatiFesc', '') for caseID in caseIDs]
  outcome = 'class'

  # builds the factor matrix Z
  (factor_names, Z, misses) = build_factor_matrix(ds, meta, factors, get_factor_labels)
  for (variable, value) in misses:
    tsprint(f'** No label found for factor level ({variable}, {value})')

  # builds the feature matrix X
  (feature_names, X) = build_feature_matrix(ds, domains, build_feature_matrix_row)
  assert (np.corrcoef(X, rowvar=False) < 0).sum() == 0

  # builds the target matrix Y
  n = ds[outcome].unique().size
  target_names = [str(e) for e in sorted(ds[outcome].unique().tolist())]
  Y = np.array([target_names.index(str(e)) for e in ds[outcome]], dtype=int)
  assert np.isnan(Y).sum() == 0

  # builds the dataset
  dataset = Bunch(
      caseIDs = caseIDs,
      data    = X,
      target  = Y,
      factors = Z,
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

  tsprint(f'Loading parameters from {configfile}')

  setupEssayConfig(configfile)
  sourcepath  = getEssayParameter('PARAM_WHOQOL_SOURCEPATH')
  targetpath  = getEssayParameter('PARAM_WHOQOL_TARGETPATH')
  filename    = getEssayParameter('PARAM_WHOQOL_FILENAME')
  fuzzfactors = getEssayParameter('PARAM_WHOQOL_FUZZFACTOR')
  basename    = filename.split('.')[0].lower()

  # ensures the target folder is available and empty
  if(exists(join(*targetpath))):
    for f in listdir(join(*targetpath)):
      remove(join(*targetpath, f))
  else:
    makedirs(join(*targetpath))

  # reads the dataset file
  tsprint(f'Reading dataset file {filename} from {join(*sourcepath)}')
  df = pd.read_excel(join(*sourcepath, filename))

  # discards assessments without properly documented formal consent;
  df = df.loc[df['TCLE'] == 'SIM']

  # reproduces a couple of statistics published in [1], Table 1
  aux1 = len(df[df['A.1 Escolaridade'].isin(['Ensino Superior', 'Ensino Técnico'])].index)/len(df.index)
  aux2 = len(df[df['A.1 Escolaridade'].isin(['Fundamental Incompleto',])].index)/len(df.index)
  aux3 = len(df.loc[df['H3.a'] >= 2.].index)/len(df.index)
  tsprint(__doc__)
  tsprint('-- comparing some statistics with the published data in [1], Table 1:')
  tsprint('  ---------- statistic ----------  \tArticle  Sample')
  tsprint('  # of participants ............. :\t  {0:>5d}\t  {1:>5d}'.format(107, len(df.index)))
  tsprint('  average age ................... :\t   {0:4.1f}\t   {1:4.1f}'.format(67.1, df['Idade'].mean()))
  tsprint('  higher or technical education . :\t  {0:4.1f}%\t  {1:4.1f}%'.format(39.2, 100 * aux1))
  tsprint('  incomplete elementary education :\t  {0:4.1f}%\t  {1:4.1f}%'.format(10.3, 100 * aux2))
  tsprint('  digitally engaged ............. :\t  {0:4.1f}%\t  {1:4.1f}%'.format(81.3, 100 * aux3))

  # creates a view with the columns that will be used ahead in the pipeline
  tsprint('')
  tsprint('Creating the data view (only fields/columns what will be used)')

  domains = {
            'physical health':      ['I.3','I.4','I.10','I.15','I.16','I.17','I.18'],
            'psychological':        ['I.5','I.6','I.7','I.11','I.19','I.26'],
            'social relationships': ['I.20','I.21','I.22'],
            'environment':          ['I.8','I.9','I.12','I.13','I.14','I.23','I.24','I.25'],
  }

  questions = {
            'I.1':  'How would you rate your quality of life?',
            'I.2':  'How satisfied are you with your health?',
            'I.3':  'To what extent do you feel that physical pain prevents you from doing what you need to do?',
            'I.4':  'How much do you need any medical treatment to function in your daily life?',
            'I.5':  'How much do you enjoy life?',
            'I.6':  'To what extent do you feel your life to be meaningful?',
            'I.7':  'How well are you able to concentrate?',
            'I.8':  'How safe do you feel in your daily life?',
            'I.9':  'How healthy is your physical environment?',
            'I.10': 'Do you have enough energy for everyday life?',
            'I.11': 'Are you able to accept your bodily appearance?',
            'I.12': 'Have you enough money to meet your needs?',
            'I.13': 'How available to you is the information that you need in your day-to-day life?',
            'I.14': 'To what extent do you have the opportunity for leisure activities?',
            'I.15': 'How well are you able to get around?',
            'I.16': 'How satisfied are you with your sleep?',
            'I.17': 'How satisfied are you with your ability to perform your daily living activities?',
            'I.18': 'How satisfied are you with your capacity for work?',
            'I.19': 'How satisfied are you with yourself?',
            'I.20': 'How satisfied are you with your personal relationships?',
            'I.21': 'How satisfied are you with your sex life?',
            'I.22': 'How satisfied are you with the support you get from your friends?',
            'I.23': 'How satisfied are you with the conditions of your living place?',
            'I.24': 'How satisfied are you with your access to health services?',
            'I.25': 'How satisfied are you with your transport?',
            'I.26': 'How often do you have negative feelings such as blue mood, despair, anxiety, depression?',
  }

  features = list(chain(*domains.values()))
  factors  = ['Idade', 'Gênero', 'A.1 Escolaridade']

  # extends the dataset with new columns
  for domain in domains:
    df[domain] = df.apply(lambda row: whoqol(row, domains, domain),  axis=1)
  df['sumscore'] = df.apply(lambda row: whoqol(row, domains, 'sumscore'),  axis=1)
  extra = ['Identificação', 'TCLE', 'H3.a', 'sumscore'] + list(domains.keys())

  """
  Each participant is assigned to a class which is defined by the following rule proposed
  in [4]: GOOD QOL if the WHOQOL sumscore >= cutoff, POOR QOL otherwise. Some comments
  are in place:
  -- In [4], Figure 2 suggests the authors used sumscores in the 0-100 scale, but we
     adopted the 0-80 scale in the beginning of our project;
  -- They proposed a cutoff at sumscore = 60, which corresponds to 48 in our scale;
  -- We tried cutoff = 48, but this lead to #(Good,Poor)=(91,9) in our sample;
  -- Although the observed imbalance could be justified in principle by pointing out that
     the individuals engaged in UATI/FESC activities are predominantly priviledged from a
     functional perspective (i.e., most can come and and go back home independently of the
     help of others), we decided to keep the cutoff = 60, as it leads to a less imbalanced
     result, which is interesting for our running example in Chapter 4 of the thesis.
  """
  whoqol_cutoff = 60
  df['class'] = df.apply(lambda row: 'Good QOL' if row['sumscore'] >= whoqol_cutoff else 'Poor QOL', axis=1)
  outcomes = ['class']

  subset   = (factors, features, outcomes, extra)
  filename = '{0}-view.csv'.format(basename)
  (content, df) = createView(df, subset, dropna=False)
  saveAsText('\n'.join(content), join(*targetpath, filename))
  tsprint(f'-- selected factors : {factors}')
  tsprint(f'-- selected features: {features}')
  tsprint(f'-- selected outcomes: {outcomes}')
  tsprint(f'-- view with selected columns saved as {join(*targetpath, filename)}')
  tsprint(f'-- view has {len(df.index)} rows')

  for domain in domains:
    tsprint(f'-- Domain: {domain}')
    for item in sorted(domains[domain]):
      tsprint(f'   {item:<4}\t{questions[item]}')

  #--------------------------------------------------------------------------------------
  # creates a multiclass dataset
  #--------------------------------------------------------------------------------------
  descr = """
   Dataset DG-UFSCar/FESC WHOQOL
   -----------------------------

   This dataset contains quality of life assessments of individuals attending courses
   offered by the UATI/FESC (Universidade Aberta à Terceira Idade/Fundação Educacional
   São Carlos), which were collected by researchers affiliated to the Department of
   Gerontology at the UFSCar/BR (Universidade Federal de São Carlos/Brazil) in October
   2019 as part of the research program entitled "Envelhecimento Ativo na Universidade
   da Terceira Idade":

   [1] L. J. Lorenzi, P. A. R. Alvarez, P. Bet, P. C. Castro (2022). Digital engagement
       and quality of life of participants at a University of the Third Age.
       Gerontechnology, 21(s), 1-1, https://doi.org/10.4017/gt.2022.21.s.567.opp4

   [2] Termo de Consentimento Livre e Esclarecido para a pesquisa "Envelhecimento ativo
       na Universidade da Terceira Idade" (Protocolo de avaliação + TCLE.pdf)

   The process that converts participant answers into quality of life scores follows the
   methodology described in:

   [3] World Health Organization. WHOQOL-BREF: Introduction, Administration, Scoring and
       generic version of the assessment: Field trial version, December 1996.
       World Health Organization. Division of Mental Health, 1996. (see Table 3)

   Each participant is assigned to a class determined by a single decision boundary defined
   on the WHOQOL sumscore ('Good QOL' if sumscore >= 60, 'Poor QOL' otherwise). This cutoff
   value is based on results published in:

   [4] Silva PAB, Soares SM, Santos JFG, Silva LB. Cut-off point for WHOQOL-BREF as a
       measure of quality of life of older adults. Rev Saúde Pública. 2014Jun; 48(3):390–7.
       https://doi.org/10.1590/S0034-8910.2014048004912
  """

  tsprint('')
  tsprint(f'Creating a multiclass dataset based on {basename} data, with 2 labels')
  filename = f'{basename}-data'
  meta = None
  (dataset, report) = convert2Bunch_multiclass(df, meta, descr, subset, domains)
  serialise(dataset, join(*targetpath, filename))
  report.append(f'dataset saved as {join(*targetpath, filename)}.pkl')
  tsprint('\n'.join([f'[{stimestamp()}] -- {msg}' for msg in report]), stamp=False)
  #assert (nrows == 107) # according to data published in the results of [1]

  #--------------------------------------------------------------------------------------
  # records distributional characteristics observed in [5, 6], as well as the parameters
  # we use to simulate them based on the feature data of the the WHOQOL dataset
  #--------------------------------------------------------------------------------------
  observed  = {11: {'data': 'AMPI-AB',  'cardinality': 1.08, 'density': 0.10},
               22: {'data': 'ICOPE-FR', 'cardinality': 4.54, 'density': 0.21},
              }

  simulated = {11: {'maxcareplan': 2,    'fuzzfactor': fuzzfactors[11], },
               22: {'maxcareplan': None, 'fuzzfactor': fuzzfactors[22], },
              }

  #--------------------------------------------------------------------------------------
  # creates a multilabel version of the dataset, with 11 labels, simulating [5]
  #--------------------------------------------------------------------------------------
  descr="""
   Dataset DG-UFSCar/FESC WHOQOL-ML-11
   -----------------------------------

   This dataset contains quality of life assessments of individuals attending courses
   offered by the UATI/FESC (Universidade Aberta à Terceira Idade/Fundação Educacional
   São Carlos), which were collected by researchers affiliated to the Department of
   Gerontology at the UFSCar/BR (Universidade Federal de São Carlos/Brazil) in October
   2019 as part of the research program entitled "Envelhecimento Ativo na Universidade
   da Terceira Idade":

   [1] L. J. Lorenzi, P. A. R. Alvarez, P. Bet, P. C. Castro (2022). Digital engagement
       and quality of life of participants at a University of the Third Age.
       Gerontechnology, 21(s), 1-1, https://doi.org/10.4017/gt.2022.21.s.567.opp4

   [2] Termo de Consentimento Livre e Esclarecido para a pesquisa "Envelhecimento ativo
       na Universidade da Terceira Idade" (Protocolo de avaliação + TCLE.pdf)

   The process that converts participant answers into quality of life scores follows the
   methodology described in:

   [3] World Health Organization. WHOQOL-BREF: Introduction, Administration, Scoring and
       generic version of the assessment: Field trial version, December 1996.
       World Health Organization. Division of Mental Health, 1996. (see Table 3)

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

  (scenario, infix) = (ECO_DB_MULTILABEL, 'ml')
  ngroups = 11
  tsprint('')
  tsprint(f'Creating a {scenario} dataset based on {basename} data, with {ngroups} labels')
  filename = f'{basename}-{infix}-{ngroups}-data'

  (dataset, report) = updateAssignments(dataset, descr, scenario, ngroups,
                                        observed, simulated)
  serialise(dataset, join(*targetpath, filename))
  report.append(f'dataset saved as {join(*targetpath, filename)}.pkl')
  tsprint('\n'.join([f'[{stimestamp()}] -- {msg}' for msg in report]), stamp=False)

  #--------------------------------------------------------------------------------------
  # creates a multilabel version of the dataset, with 22 labels, simulating [6]
  #--------------------------------------------------------------------------------------
  descr="""
   Dataset DG-UFSCar/FESC WHOQOL-ML-22
   -----------------------------------

   This dataset contains quality of life assessments of individuals attending courses
   offered by the UATI/FESC (Universidade Aberta à Terceira Idade/Fundação Educacional
   São Carlos), which were collected by researchers affiliated to the Department of
   Gerontology at the UFSCar/BR (Universidade Federal de São Carlos/Brazil) in October
   2019 as part of the research program entitled "Envelhecimento Ativo na Universidade
   da Terceira Idade":

   [1] L. J. Lorenzi, P. A. R. Alvarez, P. Bet, P. C. Castro (2022). Digital engagement
       and quality of life of participants at a University of the Third Age.
       Gerontechnology, 21(s), 1-1, https://doi.org/10.4017/gt.2022.21.s.567.opp4

   [2] Termo de Consentimento Livre e Esclarecido para a pesquisa "Envelhecimento ativo
       na Universidade da Terceira Idade" (Protocolo de avaliação + TCLE.pdf)

   The process that converts participant answers into quality of life scores follows the
   methodology described in:

   [3] World Health Organization. WHOQOL-BREF: Introduction, Administration, Scoring and
       generic version of the assessment: Field trial version, December 1996.
       World Health Organization. Division of Mental Health, 1996. (see Table 3)

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

  (scenario, infix) = (ECO_DB_MULTILABEL, 'ml')
  ngroups = 22
  tsprint('')
  tsprint(f'Creating a {scenario} dataset based on {basename} data, with {ngroups} labels')
  filename = f'{basename}-{infix}-{ngroups}-data'

  (dataset, report) = updateAssignments(dataset, descr, scenario, ngroups,
                                        observed, simulated)
  serialise(dataset, join(*targetpath, filename))
  report.append(f'dataset saved as {join(*targetpath, filename)}.pkl')
  tsprint('\n'.join([f'[{stimestamp()}] -- {msg}' for msg in report]), stamp=False)

  #--------------------------------------------------------------------------------------
  # creates a label ranking version of the dataset, with 11 labels, simulating [5]
  #--------------------------------------------------------------------------------------
  descr="""
   Dataset DG-UFSCar/FESC WHOQOL-LR-11
   -----------------------------------

   This dataset contains quality of life assessments of individuals attending courses
   offered by the UATI/FESC (Universidade Aberta à Terceira Idade/Fundação Educacional
   São Carlos), which were collected by researchers affiliated to the Department of
   Gerontology at the UFSCar/BR (Universidade Federal de São Carlos/Brazil) in October
   2019 as part of the research program entitled "Envelhecimento Ativo na Universidade
   da Terceira Idade":

   [1] L. J. Lorenzi, P. A. R. Alvarez, P. Bet, P. C. Castro (2022). Digital engagement
       and quality of life of participants at a University of the Third Age.
       Gerontechnology, 21(s), 1-1, https://doi.org/10.4017/gt.2022.21.s.567.opp4

   [2] Termo de Consentimento Livre e Esclarecido para a pesquisa "Envelhecimento ativo
       na Universidade da Terceira Idade" (Protocolo de avaliação + TCLE.pdf)

   The process that converts participant answers into quality of life scores follows the
   methodology described in:

   [3] World Health Organization. WHOQOL-BREF: Introduction, Administration, Scoring and
       generic version of the assessment: Field trial version, December 1996.
       World Health Organization. Division of Mental Health, 1996. (see Table 3)

   Each participant is assigned to one or two labels among 11 labels. This arrangement
   aims to simulate distributional characteristics observed in a sample of the data that
   was collected for the article:

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

  (scenario, infix) = (ECO_DB_LABELRANK, 'lr')
  ngroups = 11
  tsprint('')
  tsprint(f'Creating a {scenario} dataset based on {basename} data, with {ngroups} labels')
  filename = f'{basename}-{infix}-{ngroups}-data'

  (dataset, report) = updateAssignments(dataset, descr, scenario, ngroups,
                                        observed, simulated)
  serialise(dataset, join(*targetpath, filename))
  report.append(f'dataset saved as {join(*targetpath, filename)}.pkl')
  tsprint('\n'.join([f'[{stimestamp()}] -- {msg}' for msg in report]), stamp=False)

  #--------------------------------------------------------------------------------------
  # creates a label ranking version of the dataset, with 22 labels, simulating [6]
  #--------------------------------------------------------------------------------------
  descr="""
   Dataset DG-UFSCar/FESC WHOQOL-LR-22
   -----------------------------------

   This dataset contains quality of life assessments of individuals attending courses
   offered by the UATI/FESC (Universidade Aberta à Terceira Idade/Fundação Educacional
   São Carlos), which were collected by researchers affiliated to the Department of
   Gerontology at the UFSCar/BR (Universidade Federal de São Carlos/Brazil) in October
   2019 as part of the research program entitled "Envelhecimento Ativo na Universidade
   da Terceira Idade":

   [1] L. J. Lorenzi, P. A. R. Alvarez, P. Bet, P. C. Castro (2022). Digital engagement
       and quality of life of participants at a University of the Third Age.
       Gerontechnology, 21(s), 1-1, https://doi.org/10.4017/gt.2022.21.s.567.opp4

   [2] Termo de Consentimento Livre e Esclarecido para a pesquisa "Envelhecimento ativo
       na Universidade da Terceira Idade" (Protocolo de avaliação + TCLE.pdf)

   The process that converts participant answers into quality of life domain scores uses
   the data and follows the method described in:

   [3] World Health Organization. WHOQOL-BREF: Introduction, Administration, Scoring and
       generic version of the assessment: Field trial version, December 1996.
       World Health Organization. Division of Mental Health, 1996. (see Table 3)

   Each participant is assigned to at least one label among 22 labels. This arrangement
   aims to simulate distributional characteristics observed in a sample of the data that
   was collected for the article:

   [6] Tavassoli, Neda, Philipe de Souto Barreto, Caroline Berbon, Celine Mathieu,
       Justine de Kerimel, Christine Lafont, Catherine Takeda et al. "Implementation of
       the WHO integrated care for older people (ICOPE) programme in clinical practice:
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

  (scenario, infix) = (ECO_DB_LABELRANK, 'lr')
  ngroups = 22
  tsprint('')
  tsprint(f'Creating a {scenario} dataset based on {basename} data, with {ngroups} labels')
  filename = f'{basename}-{infix}-{ngroups}-data'

  (dataset, report) = updateAssignments(dataset, descr, scenario, ngroups,
                                        observed, simulated)
  serialise(dataset, join(*targetpath, filename))
  report.append(f'dataset saved as {join(*targetpath, filename)}.pkl')
  tsprint('\n'.join([f'[{stimestamp()}] -- {msg}' for msg in report]), stamp=False)


  # saves the execution log
  logfile = 'readwhoqol.log'
  tsprint('')
  tsprint('This report was saved as {0}'.format(join(*targetpath, logfile)))
  tsprint('Done.')
  saveLog(join(*targetpath, logfile))

if(__name__ == '__main__'):

  main(sys.argv[1])
