#! /usr/bin/python3
# %% 
import numpy
import pandas
import tqdm
import matplotlib.pyplot
from sklearn.preprocessing import OneHotEncoder
from collections import Counter

# %% 
def FieldDifference(dataFrame_source: pandas.DataFrame, dataFrame_target: pandas.DataFrame) -> set[str]:
  return set(dataFrame_target)-set(dataFrame_source)
# %% 
def Extraction_Cases(dataFrame: pandas.DataFrame, fields: list[str], use_notebook=False) -> list[list]: 
  """Return the cases of the fields in the data set, every field's cases will order by frequency of occurrence from highest to lowest

  Args:
    dataFrame (pandas.DataFrame): target dataFrame
    fields (list[str]): target fields
    use_notebook (bool, optional): If True, use tqdm_notebook for progress bar visualization. Defaults to False.

  Returns:
    list[list[str]]: fields cases
  """
  tqdm_func = tqdm.tqdm_notebook if use_notebook else tqdm.tqdm
  print(f"DataFrame.Extraction_Cases: {fields}")
  progressBar_0 = tqdm_func(total=len(fields), unit="field", desc="Fields")
  fieldsCases = []
  for field in fields:
    counter = Counter(dataFrame[field])
    fieldsCases += [list(pandas.DataFrame({"case": list(counter), "count": [counter[case] for case in list(counter)]}).sort_values("count", ascending=False)["case"])]
    progressBar_0.update(1)
  progressBar_0.close()
  return fieldsCases
# %% 
def Extraction_OneHotEncode_merged_(dataFrame: pandas.DataFrame, fields: list[str], fieldsCases: list[list[str]], use_notebook=False) -> pandas.DataFrame: 
  """Upgrade the specified field content case to a new field

  Args:
    dataFrame (pandas.DataFrame): target dataFrame
    fields (list[str]): target fields
    fieldsCases (list[list[str]]): target fields cases
    use_notebook (bool, optional): If True, use tqdm_notebook for progress bar visualization. Defaults to False.

  Raises:
    ValueError: different length: len(fields) != len(fieldsCases)

  Returns:
    pandas.DataFrame: result dataFrame
  """
  if len(fields)!=len(fieldsCases): 
    raise ValueError("different length: len(fields) != len(fieldsCases)")
  tqdm_func = tqdm.tqdm_notebook if use_notebook else tqdm.tqdm

  print(f"DataFrame.Extraction_OneHotEncode: {fields}")
  progressBar_0 = tqdm_func(total=len(fields), unit="field", desc="Fields")
  for fieldIndex in range(len(fields)):
    # Create fieldCase's Mapping rule and fieldCasesName
    progressBar_1 = tqdm_func(total=len(fieldsCases[fieldIndex]), unit="case", desc=f"Field {fieldIndex+1}/{len(fields)}")
    caseMapping = {}
    for caseIndex in range(len(fieldsCases[fieldIndex])):
      caseMapping[fieldsCases[fieldIndex][caseIndex]] = caseIndex
      # fix `fieldsCases[fieldIndex][caseIndex]` collision
      fieldsCases[fieldIndex][caseIndex] = fields[fieldIndex] + "_" + fieldsCases[fieldIndex][caseIndex]
      while fieldsCases[fieldIndex][caseIndex] in dataFrame:
        fieldsCases[fieldIndex][caseIndex] += '_'
      progressBar_1.update(1)
    progressBar_1.close()
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
    progressBar_0.update(1)
  progressBar_0.close()
  return dataFrame
def Extraction_OneHotEncode_merged(dataFrame: pandas.DataFrame, fields: list[str], use_notebook=False) -> pandas.DataFrame: 
  """Upgrade the specified field content case to a new field

  Args:
    dataFrame (pandas.DataFrame): target dataFrame
    fields (list[str]): target fields
    use_notebook (bool, optional): If True, use tqdm_notebook for progress bar visualization. Defaults to False.

  Returns:
    pandas.DataFrame: result dataFrame
  """
  return Extraction_OneHotEncode_merged_(dataFrame, fields, Extraction_Cases(dataFrame, fields), use_notebook)
# %%
def Extraction_Element_merged(dataFrame: pandas.DataFrame, fields: list[str], parses: list, elementsList: list[set[str]], use_notebook=False) -> pandas.DataFrame:
  """After the specified field is divided according to the specified rule, the existence of the specified element is extracted.

  Args:
    dataFrame (pandas.DataFrame): target dataFrame
    fields (list[str]): target fields
    parse (list): functions to convert the target field
    elements (list[set[str]]): The set of elements to extract
    use_notebook (bool, optional): If True, use tqdm_notebook for progress bar visualization. Defaults to False.

  Raises:
    ValueError: different length: (len(fields)!=len(parses) or len(fields)!=len(elements))

  Returns:
    pandas.DataFrame: result dataFrame
  """
  if len(fields)!=len(parses) or len(fields)!=len(elementsList): 
    raise ValueError("different length: (len(fields)!=len(parses) or len(fields)!=len(elements))")
  tqdm_func = tqdm.tqdm_notebook if use_notebook else tqdm.tqdm
  
  print(f"DataFrame.Extraction_Element: {fields}")
  progressBar_0 = tqdm_func(total=len(fields), unit="field", desc="Fields")
  for field, parse, elements in zip(fields, parses, elementsList):
    # Create new field to record the presence of each elements
    elements_fieldName = dict()
    for element in elements:
      element_fieldName = field + "_" + element
      # collision prevention
      while element_fieldName in dataFrame:
        element_fieldName += "_"
      # create and record field
      elements_fieldName[element] = element_fieldName
    
    elements_dataFrame = pandas.DataFrame(columns=elements_fieldName.values())

    # Perform data extraction and update the new field
    progressBar_1 = tqdm_func(total=len(dataFrame), unit="row", desc=f"Field {fields.index(field)+1}/{len(fields)}")
    for index, row in dataFrame.iterrows():
      data = parse(str(row[field]))
      feature = [1 if element in data else 0 for element in elements]
      elements_dataFrame.loc[index] = feature
      progressBar_1.update(1)
    progressBar_1.close()

    # Concatenate the temporary DataFrame with the original DataFrame
    dataFrame = pandas.concat([dataFrame, elements_dataFrame], axis=1)

    progressBar_0.update(1)
  progressBar_0.close()
  return dataFrame

def Extraction_Element_(dataFrame: pandas.DataFrame, fields: list[str], parses: list, elementsList: list[set[str]], use_notebook=False) -> pandas.DataFrame:
  """After the specified field is divided according to the specified rule, the existence of the specified element is extracted.

  Args:
    dataFrame (pandas.DataFrame): target dataFrame
    fields (list[str]): target fields
    parse (list): functions to convert the target field
    elements (list[set[str]]): The set of elements to extract
    use_notebook (bool, optional): If True, use tqdm_notebook for progress bar visualization. Defaults to False.

  Raises:
    ValueError: different length: (len(fields)!=len(parses) or len(fields)!=len(elements))

  Returns:
    pandas.DataFrame: result dataFrame
  """
  tqdm_func = tqdm.tqdm_notebook if use_notebook else tqdm.tqdm
  if len(fields)!=len(parses) or len(fields)!=len(elementsList): 
    raise ValueError("different length: (len(fields)!=len(parses) or len(fields)!=len(elements))")
  result_list = list()
  print(f"DataFrame.Extraction_Element: {fields}")
  progressBar_0 = tqdm_func(total=len(fields), unit="field", desc="Fields")
  for field, parse, elements in zip(fields, parses, elementsList):
    # Create new field to record the presence of each elements
    elements_fieldName = dict()
    for element in elements:
      element_fieldName = field + "_" + element
      # collision prevention
      while element_fieldName in dataFrame:
        element_fieldName += "_"
      # create and record field
      elements_fieldName[element] = element_fieldName
    
    # elements_dataFrame = pandas.DataFrame(columns=elements_fieldName.values())
    elements_data = []

    # Perform data extraction and update the new field
    progressBar_1 = tqdm_func(total=len(dataFrame), unit="row", desc=f"Field {fields.index(field)+1}/{len(fields)}")
    for index, row in dataFrame.iterrows():
      data = parse(str(row[field]))
      feature = [1 if element in data else 0 for element in elements]
      # elements_dataFrame.loc[index] = feature
      elements_data.append(feature)
      progressBar_1.update(1)
    progressBar_1.close()

    # # Concatenate the temporary DataFrame with the original DataFrame
    # dataFrame = pandas.concat([dataFrame, elements_dataFrame], axis=1)

    result_array = numpy.array(elements_data)
    result_df = pandas.DataFrame(result_array, columns=sum(elements_fieldName.values(), []))
    result_list.append(result_df)

    progressBar_0.update(1)
  progressBar_0.close()

  return result_list

def Extraction_Element_elementBatch_rowBatch(dataFrame: pandas.DataFrame, fields: list[str], parses: list, elementsList: list[set[str]], elementBatch_size=256, rowBatch_size=65536, use_notebook=False, ignore=None) -> pandas.DataFrame:
  """Extract elements from a DataFrame using element batches and row batches

  This function takes a pandas DataFrame and performs element extraction based on provided fields, parse functions, and element lists. It utilizes element batches and row batches to process the data efficiently. The function follows the following steps:

  1. Input validation: The function checks if the lengths of the `fields`, `parses`, and `elementsList` parameters are the same. If they differ, a ValueError is raised.

  2. Progress bar setup: If the `use_notebook` parameter is set to True, the function uses `tqdm_notebook` from the tqdm library for progress bar visualization. Otherwise, it uses the `tqdm` function.

  3. Iterating over fields: The function iterates over the fields defined by the `fields`, `parses`, and `elementsList` parameters. For each field, it prepares element field names and initializes a list to store the resulting DataFrames for element batches.

  4. Processing element batches: The function divides the elements into batches based on the `elementBatch_size` parameter and processes them one batch at a time. Within each element batch, the function splits the data into row batches based on the `rowBatch_size` parameter. It then iterates over the row batches and applies the parse function to extract features for each row. If the value of a field is equal to the `ignore` parameter, the row is skipped.

  5. Creating result DataFrames: The extracted features are stored in a list, and at the end of each element batch, a DataFrame is created using the collected features. The resulting DataFrames for each element batch are appended to a list.

  6. Returning the result: The final result list containing the DataFrames for each element batch is returned.

  Args:
    dataFrame (pandas.DataFrame): The input DataFrame to extract elements from.
    fields (list[str]): A list of field names present in the DataFrame.
    parses (list): A list of parse functions corresponding to each field.
    elementsList (list[set[str]]): A list of element sets for each field.
    elementBatch_size (int, optional): The number of elements to process in each batch. Defaults to 256.
    rowBatch_size (int, optional): The number of rows to process in each batch. Defaults to 65536.
    use_notebook (bool, optional): If True, use tqdm_notebook for progress bar visualization. Defaults to False.
    ignore (Any, optional): Value to ignore during element extraction. Rows with this value in the corresponding field will be skipped. Defaults to None.

  Raises:
    ValueError: If the lengths of the `fields`, `parses`, and `elementsList` parameters are not the same.

  Returns:
    pandas.DataFrame: A DataFrame containing the extracted elements for each element batch.
  """
  if len(fields) != len(parses) or len(fields) != len(elementsList):
    raise ValueError("different length: (len(fields)!=len(parses) or len(fields)!=len(elements))")
  tqdm_func = tqdm.tqdm_notebook if use_notebook else tqdm.tqdm

  print(f"DataFrame.Extraction_Element: {fields}")
  result_list = []  # List to store the resulting DataFrames

  progressBar_0 = tqdm_func(total=len(fields), unit="field", desc="Fields")
  for field_idx, (field, parse, elements) in enumerate(zip(fields, parses, elementsList), start=1):
    elements_fieldName = dict()
    for element in elements:
      element_fieldName = field + "_" + element
      while element_fieldName in dataFrame:
        element_fieldName += "_"
      elements_fieldName[element] = element_fieldName

    result_df_list = []  # List to store DataFrames for each batch

    # Batch processing for elements
    num_element_batches = int(numpy.ceil(len(elements) / elementBatch_size))
    progressBar_1 = tqdm_func(total=num_element_batches, unit="element batch", desc=f"Fields {field_idx}/{len(fields)}")
    for element_batch_idx in range(num_element_batches):
      start_element_idx = element_batch_idx * elementBatch_size
      end_element_idx = min((element_batch_idx + 1) * elementBatch_size, len(elements))
      element_batch = elements[start_element_idx:end_element_idx]  # Get a batch of elements

      # Batch processing for rows
      elements_data = []  # List to store parsed data for each batch of elements
      num_row_batches = int(numpy.ceil(len(dataFrame) / rowBatch_size))
      progressBar_2 = tqdm_func(total=num_row_batches, unit="row batch", desc=f"Element Batch {element_batch_idx+1}/{num_element_batches}")
      for row_batch_idx in range(num_row_batches):
        start_row_idx = row_batch_idx * rowBatch_size
        end_row_idx = min((row_batch_idx + 1) * rowBatch_size, len(dataFrame))
        row_batch_data = dataFrame.iloc[start_row_idx:end_row_idx]  # Get a batch of rows

        for index, row in row_batch_data.iterrows():
          if row[field] != ignore:
            data = parse(str(row[field]))
            feature = [1 if element in data else 0 for element in element_batch]
            elements_data.append(feature)  # Append parsed data for each row in the batch

        del row_batch_data  # Delete row batch data to free memory

        progressBar_2.update(1)
      progressBar_2.close()

      result_array = numpy.array(elements_data)
      columns = list(elements_fieldName.values())[:len(element_batch)]  # Select columns matching the batch
      result_df = pandas.DataFrame(result_array, columns=columns)
      result_df_list.append(result_df)  # Append DataFrame for the batch to the list
      del elements_data, result_array  # Delete elements data and result array DataFrame to release memory

      progressBar_1.update(1)
    progressBar_1.close()

    result_df = pandas.concat(result_df_list, ignore_index=True)  # Concatenate DataFrames for all batches
    result_list.append(result_df)  # Append the final DataFrame to the result list
    del result_df_list, result_df  # Delete result DataFrame and result DataFrame list to free memory

    progressBar_0.update(1)
  progressBar_0.close()

  return result_list
# %%
def Filter_Percentile(dataFrame: pandas.DataFrame, fields: list[str], round=1, borderCropping=4, borderPercentile=[25,75], whisker=1.5, whiskers=[1.5, 1.5], plotDisplay=False, plotsDisplay=False, use_notebook=False) -> pandas.DataFrame: 
  """In the specified column, keep at least the specified percentile range, extend the range of retained values and filter by this range. Defaults parameters have been set to common IQR mode.

  Args:
    dataFrame (pandas.DataFrame): target dataFrame.
    fields (list[str]): target field
    round (int, optional): Filter rounds. Defaults to 1.
    borderCropping (int, optional): number of equal parts. This setting will override B. Defaults to borderPercentile.
    borderPercentile (list, optional): Percentile intervals that must be retained. If borderCropping is set, this setting will have no effect. Defaults to [25,75].
    whisker (float, optional): Extended magnification retention range. This setting will override whiskers. Defaults to 1.5.
    whiskers (list, optional): Extended magnification of upper and lower retention range. If whisker is set, this setting will have no effect. Defaults to [1.5, 1.5].
    plotDisplay (bool, optional): Individually display the distribution of box plots before and after treatment in each column. Defaults to False.
    plotsDisplay (bool, optional): Display the distribution of box plots before and after treatment in each column. Defaults to False.
    use_notebook (bool, optional): If True, use tqdm_notebook for progress bar visualization. Defaults to False.

  Returns:
    pandas.DataFrame: result dataFrame
  """
  tqdm_func = tqdm.tqdm_notebook if use_notebook else tqdm.tqdm
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
  print(f"DataFrame.Filter_Percentile: {fields}")
  progressBar_0 = tqdm_func(total=len(fields), unit="field", desc="Fields")
  for field in fields:
    if plotDisplay:
      matplotlib.pyplot.figure(figsize=(2,5))
      matplotlib.pyplot.boxplot(dataFrame[field], showmeans=True, labels=[field])
      matplotlib.pyplot.title(f"{field} Boxplot Before percentile Filtering")
      matplotlib.pyplot.show()
    progressBar_1 = tqdm_func(total=round, unit="round", desc=f"Field {fields.index(field)+1}/{len(fields)}")
    for i in range(round):
      lower, upper = numpy.percentile(dataFrame[field],borderPercentile[0]), numpy.percentile(dataFrame[field],borderPercentile[1])
      percentile = upper - lower
      dataFrame = dataFrame[(dataFrame[field] > lower-whiskers[0]*percentile) & (dataFrame[field] <= upper+whiskers[1]*percentile)]
      progressBar_1.update(1)
    progressBar_1.close()
    if plotDisplay:
      matplotlib.pyplot.figure(figsize=(2,5))
      matplotlib.pyplot.boxplot(dataFrame[field], showmeans=True, labels=[field])
      matplotlib.pyplot.title(f"{field} Boxplot After percentile Filtering")
      matplotlib.pyplot.show()
    progressBar_0.update(1)
  progressBar_0.close()
  # show percentile distribution
  if plotsDisplay and len(fields)!=0:
    matplotlib.pyplot.figure(figsize=(len(fields),5))
    matplotlib.pyplot.boxplot(dataFrame[fields], showmeans=True, labels=fields)
    matplotlib.pyplot.title("Boxplot Before percentile Filtering")
    matplotlib.pyplot.show()
  return dataFrame
def Filter_Quartile(dataFrame: pandas.DataFrame, fields: list[str], round=1, plotDisplay=False, plotsDisplay=False, use_notebook=False) -> pandas.DataFrame: 
  """Use IQR rules to filter specified fields.

  Args:
    dataFrame (pandas.DataFrame): target dataFrame.
    fields (list[str]): target field
    round (int, optional): Filter rounds. Defaults to 1.
    plotDisplay (bool, optional): Individually display the distribution of box plots before and after treatment in each column. Defaults to False.
    plotsDisplay (bool, optional): Display the distribution of box plots before and after treatment in each column. Defaults to False.
    use_notebook (bool, optional): If True, use tqdm_notebook for progress bar visualization. Defaults to False.

  Returns:
    pandas.DataFrame: result dataFrame
  """
  return Filter_Percentile(dataFrame, fields, round, plotDisplay=plotDisplay, plotsDisplay=plotsDisplay, use_notebook=use_notebook)
# %%
def Filter_NormalDistribution(dataFrame: pandas.DataFrame, fields: list[str], round=1, stddevRange=2, stddevRanges=[2, 2], plotDisplay=False, plotsDisplay=False, use_notebook=False) -> pandas.DataFrame: 
  """Use the rule of normal distribution in the specified field, and retain the items whose distance from the mean is less than the specified standard deviation

  Args:
    dataFrame (pandas.DataFrame): target dataFrame
    fields (list[str]): target field
    round (int, optional): Filter rounds. Defaults to 1.
    stddevRange (int, optional): Extended Standard Deviation Multiplier. This setting will override stddevRanges. Defaults to 2.
    stddevRanges (list, optional): Standard deviation upper and lower extension magnification. If stddevRange is set, this setting will have no effect. Defaults to [2, 2].
    plotDisplay (bool, optional): Individually display the distribution of box plots before and after treatment in each column. Defaults to False.
    plotsDisplay (bool, optional): Display the distribution of box plots before and after treatment in each column. Defaults to False.
    use_notebook (bool, optional): If True, use tqdm_notebook for progress bar visualization. Defaults to False.

  Returns:
    pandas.DataFrame: result dataFrame
  """
  tqdm_func = tqdm.tqdm_notebook if use_notebook else tqdm.tqdm
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
  print(f"DataFrame.Filter_NormalDistribution: {fields}")
  progressBar_0 = tqdm_func(total=len(fields), unit="field", desc="Fields")
  for field in fields:
    if plotsDisplay:
      pass
    progressBar_1 = tqdm_func(total=round, unit="round", desc=f"Field {fields.index(field)+1}/{len(fields)}")
    for i in range(round): 
      mean = dataFrame[field].mean()
      lower, upper = mean-(dataFrame[field].std()*stddevRanges[0]), mean+(dataFrame[field].std()*stddevRanges[1])
      dataFrame = dataFrame[(dataFrame[field] > lower) & (dataFrame[field] <= upper)]
      progressBar_1.update(1)
    progressBar_1.close()
    if plotsDisplay:
      pass
      # matplotlib.pyplot.figure(figsize=(2,5))
      # matplotlib.pyplot.boxplot(dataFrame[field], showmeans=True, labels=[field])
      # matplotlib.pyplot.title(f"{field} Boxplot After NormalDistribute Filtering")
      # matplotlib.pyplot.show()
    progressBar_0.update(1)
  progressBar_0.close()
  # show percentile distribution
  if plotDisplay and len(fields)!=0:
    pass
    # matplotlib.pyplot.figure(figsize=(len(fields),5))
    # matplotlib.pyplot.boxplot(dataFrame[fields], showmeans=True, labels=fields)
    # matplotlib.pyplot.title("Boxplot Before NormalDistribute Filtering")
    # matplotlib.pyplot.show()
  return dataFrame
  
# %%
def Process_Normalization(dataFrame: pandas.DataFrame, fields: list[str], use_notebook=False) -> pandas.DataFrame: 
  """_summary_

  Args:
    dataFrame (pandas.DataFrame): target dataFrame
    fields (list[str]): target fields
    use_notebook (bool, optional): If True, use tqdm_notebook for progress bar visualization. Defaults to False.

  Returns:
    pandas.DataFrame: result dataFrame
  """
  tqdm_func = tqdm.tqdm_notebook if use_notebook else tqdm.tqdm
  print(f"DataFrame.Process_Normalization: {fields}")
  progressBar_0 = tqdm_func(total=len(fields), unit="field", desc="Fields")
  for field in fields:
    dataFrame[field] = (dataFrame[field] - dataFrame[field].min())/(dataFrame[field].max() - dataFrame[field].min())
    progressBar_0.update(1)
  progressBar_0.close()
  return dataFrame
# %%
def Extraction_Filter_NormalDistribution(dataFrame: pandas.DataFrame, fields: list[str], field_analyzes: list, stddevRange=2, stddevRanges=[2, 2], use_notebook=False) -> list[list]:
  """Obtain keywords from the content of each field through the specified analysis method, and then extract keywords whose occurrence frequency deviation from the mean is less than the specified standard deviation

  Args:
    dataFrame (pandas.DataFrame): source DataFrame
    fields (list[str]): target fields
    field_analyzes (list): List of Field Analysis Methods
    stddevRange (int, optional): Extended Standard Deviation Multiplier. This setting will override stddevRanges. Defaults to 2.
    stddevRanges (list, optional): Standard deviation upper and lower extension magnification. If stddevRange is set, this setting will have no effect. Defaults to [2, 2].
    use_notebook (bool, optional): If True, use tqdm_notebook for progress bar visualization. Defaults to False.

  Raises:
    ValueError: different length

  Returns:
    list[list]: The extracted list, the first level is the field, and the second level is the content
  """
  if len(fields)!=len(field_analyzes):
    raise ValueError("len(fields) != len(field_analyzes)")
  tqdm_func = tqdm.tqdm_notebook if use_notebook else tqdm.tqdm
  # stddevRange will override stddevRanges
  if stddevRange!=2:
    stddevRanges[0] = stddevRange
    stddevRanges[1] = stddevRange
  result = []
  print(f"DataFrame.Extraction_Filter_NormalDistribution: {fields}")
  progressBar_0 = tqdm_func(total=len(fields), unit="field", desc="Fields")
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
    progressBar_0.update(1)
  progressBar_0.close()
  return result