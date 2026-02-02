import re
import numpy as np

from customDefs    import tsprint, stimestamp #, saveAsText, saveLog, serialise
from datasets      import assign_groups, compute_split_stats


from datasets      import ECO_ASSIGNMENT_F, ECO_ASSIGNMENT_R
from datasets      import ECO_DB_MULTILABEL, ECO_DB_LABELRANK

def createMetaDict(meta, regexstr='.*'):

  mask    = '{0}\t{1}\t{2}\t{3}\t{4}'
  header  = mask.format('item', 'value', 'description of value', 'item score', 'item phrasing')
  content = [header]

  pattern = re.compile(regexstr)
  for variable in meta.column_names:
    if(pattern.match(variable)):
      if(variable in meta.variable_value_labels):
        for value in meta.variable_value_labels[variable]:
          buffer = mask.format(variable,
                               value,
                               meta.variable_value_labels[variable][value],
                               -1,
                               meta.column_names_to_labels[variable])
          content.append(buffer)
      else:
        buffer = mask.format(variable,
                             'not documented',
                             'not documented',
                             -1,
                             meta.column_names_to_labels[variable])
        content.append(buffer)

  return content

def createView(df, subset, dropna=True):

  (factors, features, outcomes, extra) = subset

  mask    = '\t'.join(['{' + str(i) + '}' for i in range(len(factors+features+outcomes+extra) + 1)])
  header  = mask.format('subject', *factors, *features, *outcomes, *extra)
  content = [header]

  ds = df[factors+features+outcomes+extra]
  if(dropna):
    ds = ds.dropna(subset=features+outcomes)

  for (subjectID, row) in ds.iterrows():
    factor_vals  = [row[variable] for variable in factors]
    feature_vals = [row[variable] for variable in features]
    outcome_vals = [row[variable] for variable in outcomes]
    extra_vals   = [row[variable] for variable in extra]
    buffer = mask.format(subjectID, *factor_vals, *feature_vals, *outcome_vals, *extra_vals)
    content.append(buffer)

  return (content, ds)

def build_factor_matrix(df, meta, factors, get_factor_labels):
  factor_names = factors
  misses = []
  f = lambda row: get_factor_labels(row, meta, factor_names, misses)
  Z = df[factors].apply(f, axis=1).to_numpy()
  return (factor_names, Z, misses)

def build_feature_matrix(df, domains, build_feature_matrix_row):
  m = len(df.index)
  d = len(domains)
  feature_names = []
  X = np.empty((m,d))
  for (j, domain) in enumerate(domains):
  #for (j, domain) in enumerate(sorted(domains)):
    feature_names.append(domain)
    X[:,j] = build_feature_matrix_row(df, domains, domain) #df[domain].to_numpy().astype(np.float64)
  assert np.isnan(X).sum() == 0
  return (feature_names, X)

def updateAssignments(dataset, descr, scenario, ngroups, observed, simulated):

  if(  scenario == ECO_DB_MULTILABEL):
    assignmethod = ECO_ASSIGNMENT_F
  elif(scenario == ECO_DB_LABELRANK):
    assignmethod = ECO_ASSIGNMENT_R

  procedure   = (assignmethod, )
  maxcareplan = simulated[ngroups]['maxcareplan']
  fuzzfactor  = simulated[ngroups]['fuzzfactor']
  SEED = None

  # applies fuzzy clustering on the features data to create synthetic assignments
  assignpars = (procedure, False, None, maxcareplan, SEED)
  (Y, U) = assign_groups(dataset, ngroups, params=assignpars, fuzzfactor=fuzzfactor)
  (cardinality, label_counts, imbalance, labelsets, single_labelsets, density) = \
     compute_split_stats(Y, scenario)
  (nrows, ncols) = dataset.data.shape

  # updates the dataset with the a description and assignments
  dataset.DESCR = descr
  dataset.target_names = ['Class {0}'.format(j) for j in range(ngroups)]
  dataset.target = Y

  # validates the updates made in the dataset
  assert (dataset.target.shape[1] == ngroups)
  tsprint(f"-------------------------- {observed[ngroups]['cardinality']}, {cardinality}, {abs(observed[ngroups]['cardinality'] - cardinality)}")
  assert (abs(observed[ngroups]['cardinality'] - cardinality) <= 0.011)
  tsprint(f"-------------------------- {observed[ngroups]['density']}, {density}, {abs(observed[ngroups]['density'] - density)}")
  assert (abs(observed[ngroups]['density'] - density) <= 0.011)

  # creates a report showing characteristics of the new dataset
  report = []
  report.append(f'feature matrix has {nrows} rows and {ncols} columns')
  report.append(f'feature names are {dataset.feature_names}')
  report.append(f'target  names are {dataset.target_names}')
  report.append(f'factor  names are {dataset.factor_names}')
  report.append(f'cardinality .....: {cardinality:2.2f}')
  report.append(f'class imbalance .: {imbalance:2.2f}')
  report.append(f'#labelsets ......: {len(labelsets):4d}')
  report.append(f'#single labelsets: {len(single_labelsets):4d}')
  report.append(f'density .........: {density:2.2f}')

  return (dataset, report)

