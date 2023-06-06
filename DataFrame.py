#! /usr/bin/python3
# %% 
import numpy
import pandas
import matplotlib.pyplot
from sklearn.preprocessing import OneHotEncoder
from collections import Counter

# %% 
def DataFrame_FieldDifference(dataFrame_source: pandas.DataFrame, dataFrame_target: pandas.DataFrame) -> set[str]:
  return set(dataFrame_target)-set(dataFrame_source)
# %% 
def DataFrame_Extraction_Cases(dataFrame: pandas.DataFrame, fields: list[str]) -> list[list]: 
  """Return the cases of the fields in the data set, every field's cases will order by frequency of occurrence from highest to lowest

  Args:
    dataFrame (pandas.DataFrame): target dataset
    fields (list[str]): target fields

  Returns:
    list[list[str]]: fields cases
  """
  fieldsCases = []
  for field in fields:
    counter = Counter(dataFrame[field])
    fieldsCases += [list(pandas.DataFrame({"case": list(counter), "count": [counter[case] for case in list(counter)]}).sort_values("count", ascending=False)["case"])]
  return fieldsCases
# %% 
def DataFrame_Extraction_OneHotEncode_Cases(dataFrame: pandas.DataFrame, fields: list[str], fieldsCases: list[list[str]]) -> pandas.DataFrame: 
  """Upgrade the specified field content case to a new field

  Args:
    dataFrame (pandas.DataFrame): target dataset
    fields (list[str]): target fields
    fieldsCases (list[list[str]]): target fields cases

  Raises:
    ValueError: len(fields) != len(fieldsCases)

  Returns:
    pandas.DataFrame: result dataset
  """
  if len(fields)!=len(fieldsCases): 
    raise ValueError("len(fields) != len(fieldsCases)")

  for fieldIndex in range(len(fields)):
    # Create fieldCase's Mapping rule and fieldCasesName
    caseMapping = {}
    for caseIndex in range(len(fieldsCases[fieldIndex])):
      caseMapping[fieldsCases[fieldIndex][caseIndex]] = caseIndex
      # fix `fieldsCases[fieldIndex][caseIndex]` collision
      fieldsCases[fieldIndex][caseIndex] = fields[fieldIndex] + "_" + fieldsCases[fieldIndex][caseIndex]
      while fieldsCases[fieldIndex][caseIndex] in dataFrame:
        fieldsCases[fieldIndex][caseIndex] += '_'
    # fix `fieldsMappingName` collision
    fieldsMappingName = fields[fieldIndex]+"_MappingCode"
    while fieldsMappingName in dataFrame:
      fieldsMappingName += '_'
    # Add a field to record Mapping Code
    dataFrame[fieldsMappingName] = dataFrame[fields[fieldIndex]].map(caseMapping)
    # Post conversion information
    attackType_OneHotEncoder = OneHotEncoder(dtype=int).fit_transform(dataFrame[[fieldsMappingName]]).toarray()
    # Add one-hot encoding field
    dataFrame[fieldsCases[fieldIndex]] = attackType_OneHotEncoder
    # Remove redundant fields
    dataFrame = dataFrame.drop([fields[fieldIndex]], axis=1)
    dataFrame = dataFrame.drop([fieldsMappingName], axis=1)
  return dataFrame
def DataFrame_Extraction_OneHotEncode(dataFrame: pandas.DataFrame, fields: list[str]) -> pandas.DataFrame: 
  """Upgrade the specified field content case to a new field

  Args:
    dataFrame (pandas.DataFrame): target dataset
    fields (list[str]): target fields

  Returns:
    pandas.DataFrame: result dataset
  """
  return DataFrame_Extraction_OneHotEncode_Cases(dataFrame, fields, DataFrame_Extraction_Cases(dataFrame, fields))
# %%
def DataFrame_Filter_Percentile(dataFrame: pandas.DataFrame, fields: list[str], round=1, borderCropping=4, borderPercentile=[25,75], whisker=1.5, whiskers=[1.5, 1.5], plotDisplay=False, plotsDisplay=False) -> pandas.DataFrame: 
  """In the specified column, keep at least the specified percentile range, extend the range of retained values and filter by this range. Defaults parameters have been set to common IQR mode.

  Args:
    dataFrame (pandas.DataFrame): target dataset.
    fields (list[str]): target field
    round (int, optional): Filter rounds. Defaults to 1.
    borderCropping (int, optional): number of equal parts. This setting will override B. Defaults to borderPercentile.
    borderPercentile (list, optional): Percentile intervals that must be retained. If borderCropping is set, this setting will have no effect. Defaults to [25,75].
    whisker (float, optional): Extended magnification retention range. This setting will override whiskers. Defaults to 1.5.
    whiskers (list, optional): Extended magnification of upper and lower retention range. If whisker is set, this setting will have no effect. Defaults to [1.5, 1.5].
    plotDisplay (bool, optional): Individually display the distribution of box plots before and after treatment in each column. Defaults to False.
    plotsDisplay (bool, optional): Display the distribution of box plots before and after treatment in each column. Defaults to False.

  Returns:
    pandas.DataFrame: result dataset
  """
  # borderCropping  will override borderPercentile
  if borderCropping!=4:
    borderPercentile[0] = 100/borderCropping
    borderPercentile[1] = 100-borderPercentile[0]
  # whisker  will override whiskers
  if whisker!=1.5:
    whiskers[0] = whisker
    whiskers[1] = whisker
  # show percentile distribution
  if plotsDisplay and len(fields)!=0:
    matplotlib.pyplot.figure(figsize=(len(fields),5))
    matplotlib.pyplot.boxplot(dataFrame[fields], showmeans=True, labels=fields)
    matplotlib.pyplot.title("Boxplot Before percentile Filtering")
    matplotlib.pyplot.show()
  # remove outliers use percentile
  for field in fields:
    if plotDisplay:
      matplotlib.pyplot.figure(figsize=(2,5))
      matplotlib.pyplot.boxplot(dataFrame[field], showmeans=True, labels=[field])
      matplotlib.pyplot.title(f"{field} Boxplot Before percentile Filtering")
      matplotlib.pyplot.show()
    for i in range(round):
      lower, upper = numpy.percentile(dataFrame[field],borderPercentile[0]), numpy.percentile(dataFrame[field],borderPercentile[1])
      percentile = upper - lower
      dataFrame = dataFrame[(dataFrame[field] > lower-whiskers[0]*percentile) & (dataFrame[field] <= upper+whiskers[1]*percentile)]
    if plotDisplay:
      matplotlib.pyplot.figure(figsize=(2,5))
      matplotlib.pyplot.boxplot(dataFrame[field], showmeans=True, labels=[field])
      matplotlib.pyplot.title(f"{field} Boxplot After percentile Filtering")
      matplotlib.pyplot.show()
  # show percentile distribution
  if plotsDisplay and len(fields)!=0:
    matplotlib.pyplot.figure(figsize=(len(fields),5))
    matplotlib.pyplot.boxplot(dataFrame[fields], showmeans=True, labels=fields)
    matplotlib.pyplot.title("Boxplot Before percentile Filtering")
    matplotlib.pyplot.show()
  return dataFrame
def DataFrame_Filter_Quartile(dataFrame: pandas.DataFrame, fields: list[str], round=1, plotDisplay=False, plotsDisplay=False) -> pandas.DataFrame: 
  """Use IQR rules to filter specified fields.

  Args:
    dataFrame (pandas.DataFrame): target dataset.
    fields (list[str]): target field
    round (int, optional): Filter rounds. Defaults to 1.
    plotDisplay (bool, optional): Individually display the distribution of box plots before and after treatment in each column. Defaults to False.
    plotsDisplay (bool, optional): Display the distribution of box plots before and after treatment in each column. Defaults to False.

  Returns:
    pandas.DataFrame: result dataset
  """
  return DataFrame_Filter_Percentile(dataFrame, fields, round, plotDisplay=plotDisplay, plotsDisplay=plotsDisplay)
# %%
def DataFrame_Filter_NormalDistribution(dataFrame: pandas.DataFrame, fields: list[str], round=1, stddevRange=2, stddevRanges=[2, 2], plotDisplay=False, plotsDisplay=False) -> pandas.DataFrame: 
  """Use the rule of normal distribution in the specified field, and retain the items whose distance from the mean is less than the specified standard deviation

  Args:
    dataFrame (pandas.DataFrame): target dataset
    fields (list[str]): target field
    round (int, optional): Filter rounds. Defaults to 1.
    stddevRange (int, optional): Extended Standard Deviation Multiplier. This setting will override stddevRanges. Defaults to 2.
    stddevRanges (list, optional): Standard deviation upper and lower extension magnification. If stddevRange is set, this setting will have no effect. Defaults to [2, 2].
    plotDisplay (bool, optional): Individually display the distribution of box plots before and after treatment in each column. Defaults to False.
    plotsDisplay (bool, optional): Display the distribution of box plots before and after treatment in each column. Defaults to False.

  Returns:
    pandas.DataFrame: _description_
  """
  # stddevRange will override stddevRanges
  if stddevRange!=2:
    stddevRanges[0] = stddevRange
    stddevRanges[1] = stddevRange
  # show percentile distribution
  if plotDisplay and len(fields)!=0:
    pass
    # matplotlib.pyplot.figure(figsize=(len(fields),5))
    # matplotlib.pyplot.boxplot(dataFrame[fields], showmeans=True, labels=fields)
    # matplotlib.pyplot.title("Boxplot Before NormalDistribute Filtering")
    # matplotlib.pyplot.show()
  # remove outliers use NormalDistribute
  for field in fields:
    if plotsDisplay:
      pass
      # matplotlib.pyplot.figure(figsize=(2,5))
      # matplotlib.pyplot.boxplot(dataFrame[field], showmeans=True, labels=[field])
      # matplotlib.pyplot.title(f"{field} Boxplot Before NormalDistribute Filtering")
      # matplotlib.pyplot.show()
    for i in range(round): 
      mean = dataFrame[field].mean()
      lower, upper = mean-(dataFrame[field].std()*stddevRanges[0]), mean+(dataFrame[field].std()*stddevRanges[1])
      dataFrame = dataFrame[(dataFrame[field] > lower) & (dataFrame[field] <= upper)]
    if plotsDisplay:
      pass
      # matplotlib.pyplot.figure(figsize=(2,5))
      # matplotlib.pyplot.boxplot(dataFrame[field], showmeans=True, labels=[field])
      # matplotlib.pyplot.title(f"{field} Boxplot After NormalDistribute Filtering")
      # matplotlib.pyplot.show()
  # show percentile distribution
  if plotDisplay and len(fields)!=0:
    pass
    # matplotlib.pyplot.figure(figsize=(len(fields),5))
    # matplotlib.pyplot.boxplot(dataFrame[fields], showmeans=True, labels=fields)
    # matplotlib.pyplot.title("Boxplot Before NormalDistribute Filtering")
    # matplotlib.pyplot.show()
  return dataFrame
  
# %%
def DataFrame_Process_Normalization(dataFrame: pandas.DataFrame, fields: list[str]) -> pandas.DataFrame: 
  """_summary_

  Args:
    dataFrame (pandas.DataFrame): target dataset
    fields (list[str]): target fields

  Returns:
    pandas.DataFrame: result dataset
  """
  for field in fields:
    dataFrame[field] = (dataFrame[field] - dataFrame[field].min())/(dataFrame[field].max() - dataFrame[field].min())
  return dataFrame
# %%
def DataFrame_Extraction_Filter_NormalDistribution(dataFrame: pandas.DataFrame, fields: list[str], field_analyzes: list, stddevRange=2, stddevRanges=[2, 2]) -> list[list]:
  """Obtain keywords from the content of each field through the specified analysis method, and then extract keywords whose occurrence frequency deviation from the mean is less than the specified standard deviation

  Args:
    dataFrame (pandas.DataFrame): source DataFrame
    fields (list[str]): target fields
    field_analyzes (list): List of Field Analysis Methods
    stddevRange (int, optional): Extended Standard Deviation Multiplier. This setting will override stddevRanges. Defaults to 2.
    stddevRanges (list, optional): Standard deviation upper and lower extension magnification. If stddevRange is set, this setting will have no effect. Defaults to [2, 2].

  Raises:
      ValueError: different length

  Returns:
      list[list]: The extracted list, the first level is the field, and the second level is the content
  """
  if len(fields)!=len(field_analyzes):
    raise ValueError("len(fields) != len(field_analyzes)")
  # stddevRange will override stddevRanges
  if stddevRange!=2:
    stddevRanges[0] = stddevRange
    stddevRanges[1] = stddevRange
  result = []
  for fieldIndex in range(len(fields)):
    # Extract data
    population = []
    for domain in dataFrame[fields[fieldIndex]].dropna().to_list():
      population += field_analyzes[fieldIndex](domain)
    # Merge content
    case = list(set(population))
    # Statistical frequency
    count = numpy.array([population.count(case) for case in case])
    # Computes the standard deviation of frequency
    meanDiff = count-count.mean()
    # Normal distribution extraction field index
    keys = []
    for key_index in numpy.where((meanDiff<(stddevRanges[1]*count.std()))&(meanDiff>(-stddevRanges[0]*count.std())))[0]:
      keys += [case[key_index]]
    result += [keys]
  return result