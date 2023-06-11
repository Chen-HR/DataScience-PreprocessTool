#! /usr/bin/python3
# %% 
# from typing import List, Dict, Callable
import typing 
import numpy
import numpy as np
import pandas
import pandas as pd
import tqdm
import matplotlib.pyplot
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, LabelBinarizer
from collections import Counter

# %% 
def FieldDifference(dataFrame_source: pandas.DataFrame, dataFrame_target: pandas.DataFrame) -> set[str]:
  return set(dataFrame_target)-set(dataFrame_source) # type: ignore
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
  fieldsCases = []
  print(f"DataFrame.Extraction_Cases: {fields}")
  progressBar_0 = tqdm_func(total=len(fields), unit="field", desc="Fields")
  for field in fields:
    counter = Counter(dataFrame[field])
    fieldsCases += [list(pandas.DataFrame({"case": list(counter), "count": [counter[case] for case in list(counter)]}).sort_values("count", ascending=False)["case"])]
    progressBar_0.update(1)
  progressBar_0.close()
  return fieldsCases
# %%
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, LabelBinarizer

# def Extraction_LabelEncoder(dataFrame: pandas.DataFrame, fields: list[str]) -> list[pandas.DataFrame]: 
#   encoded_dataframes = []
#   for field in fields:
#     encoder = LabelEncoder()
#     encoded_labels = encoder.fit_transform(dataFrame[field])
#     encoded_dataframe = pandas.DataFrame({field + '_Encoded': encoded_labels})
#     encoded_dataframes.append(encoded_dataframe)
#   return encoded_dataframes

# def Extraction_OneHotEncoder(dataFrame: pandas.DataFrame, fields: list[str]) -> list[pandas.DataFrame]: 
#   encoded_dataframes = []
#   for field in fields:
#     encoder = OneHotEncoder(dtype=int)
#     encoded_features = encoder.fit_transform(dataFrame[[field]]).toarray()
#     feature_names = [field + '_' + str(category) for category in encoder.categories_[0]]
#     encoded_dataframe = pandas.DataFrame(encoded_features, columns=feature_names)
#     encoded_dataframes.append(encoded_dataframe)
#   return encoded_dataframes

# def Extraction_OrdinalEncoder(dataFrame: pandas.DataFrame, fields: list[str]) -> list[pandas.DataFrame]: 
#   encoded_dataframes = []
#   for field in fields:
#     encoder = OrdinalEncoder()
#     encoded_features = encoder.fit_transform(dataFrame[[field]])
#     encoded_dataframe = pandas.DataFrame({field + '_Encoded': encoded_features.flatten()})
#     encoded_dataframes.append(encoded_dataframe)
#   return encoded_dataframes

# def Extraction_LabelBinarizer(dataFrame: pandas.DataFrame, fields: list[str]) -> list[pandas.DataFrame]: 
#   encoded_dataframes = []
#   for field in fields:
#     encoder = LabelBinarizer()
#     binary_matrix = encoder.fit_transform(dataFrame[field])
#     feature_names = [field + '_' + str(category) for category in encoder.classes_]
#     encoded_dataframe = pandas.DataFrame(binary_matrix, columns=feature_names)
#     encoded_dataframes.append(encoded_dataframe)
#   return encoded_dataframes

def add_collision_suffix(field_name: str, existing_columns: set):
  new_field_name = field_name
  counter = 1
  while new_field_name in existing_columns:
    new_field_name = field_name + '_' + str(counter)
    counter += 1
  return new_field_name

def Extraction_LabelEncoder(dataFrame: pandas.DataFrame, fields: list[str], enable_prefix=True, prefix=None, anti_collision=True) -> list[pandas.DataFrame]:
  encoded_dataframes = []
  existing_columns = set(dataFrame.columns)
  for field in fields:
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(dataFrame[field])
    
    if enable_prefix:
      field_prefix = prefix or field
      encoded_field_name = field_prefix + '_Encoded'
    else:
      encoded_field_name = field + '_Encoded'
    
    if anti_collision:
      encoded_field_name = add_collision_suffix(encoded_field_name, existing_columns | set([field]))
    
    encoded_dataframe = pd.DataFrame({encoded_field_name: encoded_labels})
    encoded_dataframes.append(encoded_dataframe)
    existing_columns.add(encoded_field_name)
  
  return encoded_dataframes

def Extraction_OneHotEncoder(dataFrame: pandas.DataFrame, fields: list[str], enable_prefix=True, prefix=None, anti_collision=True) -> list[pandas.DataFrame]:
  encoded_dataframes = []
  existing_columns = set(dataFrame.columns)
  for field in fields:
    encoder = OneHotEncoder(dtype=int)
    encoded_features = encoder.fit_transform(dataFrame[[field]]).toarray() # type: ignore
    feature_names = [str(category) for category in encoder.categories_[0]] # type: ignore
    
    if enable_prefix:
      field_prefix = prefix or field
      feature_names = [field_prefix + '_' + str(category) for category in encoder.categories_[0]] # type: ignore
    
    if anti_collision:
      feature_names = [add_collision_suffix(name, existing_columns) for name in feature_names]
    
    encoded_dataframe = pd.DataFrame(encoded_features, columns=feature_names)
    encoded_dataframes.append(encoded_dataframe)
    existing_columns.update(feature_names)
  
  return encoded_dataframes

def Extraction_OrdinalEncoder(dataFrame: pandas.DataFrame, fields: list[str], enable_prefix=True, prefix=None, anti_collision=True) -> list[pandas.DataFrame]:
  encoded_dataframes = []
  existing_columns = set(dataFrame.columns)
  for field in fields:
    encoder = OrdinalEncoder()
    encoded_features = encoder.fit_transform(dataFrame[[field]])
    
    if enable_prefix:
      field_prefix = prefix or field
      encoded_field_name = field_prefix + '_Encoded'
    else:
      encoded_field_name = field + '_Encoded'
    
    if anti_collision:
      encoded_field_name = add_collision_suffix(encoded_field_name, existing_columns | set([field]))
    
    encoded_dataframe = pd.DataFrame({encoded_field_name: encoded_features.flatten()})
    encoded_dataframes.append(encoded_dataframe)
    existing_columns.add(encoded_field_name)
  
  return encoded_dataframes

def Extraction_LabelBinarizer(dataFrame: pandas.DataFrame, fields: list[str], enable_prefix=True, prefix=None, anti_collision=True) -> list[pandas.DataFrame]:
  encoded_dataframes = []
  existing_columns = set(dataFrame.columns)
  for field in fields:
    encoder = LabelBinarizer()
    binary_matrix = encoder.fit_transform(dataFrame[field])
    feature_names = [str(category) for category in encoder.classes_]
    
    if enable_prefix:
      field_prefix = prefix or field
      feature_names = [field_prefix + '_' + str(category) for category in encoder.classes_]
    
    if anti_collision:
      feature_names = [add_collision_suffix(name, existing_columns) for name in feature_names]
    
    encoded_dataframe = pd.DataFrame(binary_matrix, columns=feature_names)
    encoded_dataframes.append(encoded_dataframe)
    existing_columns.update(feature_names)
  
  return encoded_dataframes

# %%
def Extraction_OneHotEncode_merged_cases(dataFrame: pandas.DataFrame, fields: list[str], fieldsCases: list[list[str]], use_notebook=False) -> pandas.DataFrame: 
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

  print(f"DataFrame.Extraction_OneHotEncode_merged_cases: {fields}")
  progressBar_0 = tqdm_func(total=len(fields), unit="field", desc="Fields")
  for fieldIndex in range(len(fields)):
    # Create fieldCase's Mapping rule and fieldCasesName
    caseMapping = {}
    progressBar_1 = tqdm_func(total=len(fieldsCases[fieldIndex]), unit="case", desc=f"Field {fieldIndex+1}/{len(fields)}")
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
    attackType_OneHotEncoder = OneHotEncoder(dtype=int).fit_transform(dataFrame[[fieldsMappingName]]).toarray() # type: ignore
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
  return Extraction_OneHotEncode_merged_cases(dataFrame, fields, Extraction_Cases(dataFrame, fields), use_notebook)
# %%
# def Extraction_Element_merged(dataFrame: pandas.DataFrame, fields: list[str], parses: list, elementsList: list[set[str]], use_notebook=False) -> pandas.DataFrame:
#   """After the specified field is divided according to the specified rule, the existence of the specified element is extracted.

#   Args:
#     dataFrame (pandas.DataFrame): target dataFrame
#     fields (list[str]): target fields
#     parse (list): functions to convert the target field
#     elements (list[set[str]]): The set of elements to extract
#     use_notebook (bool, optional): If True, use tqdm_notebook for progress bar visualization. Defaults to False.

#   Raises:
#     ValueError: different length: (len(fields)!=len(parses) or len(fields)!=len(elements))

#   Returns:
#     pandas.DataFrame: result dataFrame
#   """
#   if len(fields)!=len(parses) or len(fields)!=len(elementsList): 
#     raise ValueError("different length: (len(fields)!=len(parses) or len(fields)!=len(elements))")
#   tqdm_func = tqdm.tqdm_notebook if use_notebook else tqdm.tqdm
  
#   print(f"DataFrame.Extraction_Element: {fields}")
#   progressBar_0 = tqdm_func(total=len(fields), unit="field", desc="Fields")
#   for field, parse, elements in zip(fields, parses, elementsList):
#     # Create new field to record the presence of each elements
#     elements_fieldName = dict()
#     for element in elements:
#       element_fieldName = field + "_" + element
#       # collision prevention
#       while element_fieldName in dataFrame:
#         element_fieldName += "_"
#       # create and record field
#       elements_fieldName[element] = element_fieldName
    
#     elements_dataFrame = pandas.DataFrame(columns=list(elements_fieldName.values()))

#     # Perform data extraction and update the new field
#     progressBar_1 = tqdm_func(total=len(dataFrame), unit="row", desc=f"Field {fields.index(field)+1}/{len(fields)}")
#     for index, row in dataFrame.iterrows():
#       data = parse(str(row[field]))
#       feature = [1 if element in data else 0 for element in elements]
#       elements_dataFrame.loc[index] = feature # type: ignore
#       progressBar_1.update(1)
#     progressBar_1.close()

#     # Concatenate the temporary DataFrame with the original DataFrame
#     dataFrame = pandas.concat([dataFrame, elements_dataFrame], axis=1)

#     progressBar_0.update(1)
#   progressBar_0.close()
#   return dataFrame

# def Extraction_Element_(dataFrame: pandas.DataFrame, fields: list[str], parses: list, elementsList: list[set[str]], use_notebook=False) -> list[pandas.DataFrame]:
#   """After the specified field is divided according to the specified rule, the existence of the specified element is extracted.

#   Args:
#     dataFrame (pandas.DataFrame): target dataFrame
#     fields (list[str]): target fields
#     parse (list): functions to convert the target field
#     elements (list[set[str]]): The set of elements to extract
#     use_notebook (bool, optional): If True, use tqdm_notebook for progress bar visualization. Defaults to False.

#   Raises:
#     ValueError: different length: (len(fields)!=len(parses) or len(fields)!=len(elements))

#   Returns:
#     pandas.DataFrame: result dataFrame
#   """
#   tqdm_func = tqdm.tqdm_notebook if use_notebook else tqdm.tqdm
#   if len(fields)!=len(parses) or len(fields)!=len(elementsList): 
#     raise ValueError("different length: (len(fields)!=len(parses) or len(fields)!=len(elements))")
#   result_list = list()
#   print(f"DataFrame.Extraction_Element: {fields}")
#   progressBar_0 = tqdm_func(total=len(fields), unit="field", desc="Fields")
#   for field, parse, elements in zip(fields, parses, elementsList):
#     # Create new field to record the presence of each elements
#     elements_fieldName = dict()
#     for element in elements:
#       element_fieldName = field + "_" + element
#       # collision prevention
#       while element_fieldName in dataFrame:
#         element_fieldName += "_"
#       # create and record field
#       elements_fieldName[element] = element_fieldName
    
#     # elements_dataFrame = pandas.DataFrame(columns=elements_fieldName.values())
#     elements_data = []

#     # Perform data extraction and update the new field
#     progressBar_1 = tqdm_func(total=len(dataFrame), unit="row", desc=f"Field {fields.index(field)+1}/{len(fields)}")
#     for index, row in dataFrame.iterrows():
#       data = parse(str(row[field]))
#       feature = [1 if element in data else 0 for element in elements]
#       # elements_dataFrame.loc[index] = feature
#       elements_data.append(feature)
#       progressBar_1.update(1)
#     progressBar_1.close()

#     # # Concatenate the temporary DataFrame with the original DataFrame
#     # dataFrame = pandas.concat([dataFrame, elements_dataFrame], axis=1)

#     result_array = numpy.array(elements_data)
#     result_df = pandas.DataFrame(result_array, columns=sum(elements_fieldName.values(), []))
#     result_list.append(result_df)

#     progressBar_0.update(1)
#   progressBar_0.close()

#   return result_list

def Extraction_Element_(dataFrame: pandas.DataFrame, fields: list[str], parses: list, elementsList: list[list[str]], elementBatch_size=1, rowBatch_size=65536, use_notebook=False) -> list[pandas.DataFrame]:
  """Extract specified elements from the specified fields of a DataFrame using batch processing.
  
  The function divides the data into smaller batches specified by the `elementBatch_size` and `rowBatch_size` parameters.
  
  Each batch is processed independently, and the resulting DataFrames are concatenated at the end.
  
  By processing the data in smaller batches, memory usage is reduced since only a portion of the data is processed at a time.
  
  This function supports progress bars using the `tqdm` library, with the option to use `tqdm_notebook` in Jupyter notebooks.
  
  
  (OpenAI(ChatGPT3.5) assisted production at 2023/06/08)

  Args:
    dataFrame (pandas.DataFrame): The target DataFrame to extract elements from.
    fields (list[str]): The list of field names to be processed.
    parses (list): The list of parse functions corresponding to each field.
    elementsList (list[set[str]]): The list of element sets to extract from each field.
    elementBatch_size (int, optional): The size of the element batches to process. Defaults to 1.
    rowBatch_size (int, optional): The size of the row batches to process. Defaults to 65536.
    use_notebook (bool, optional): Whether to use `tqdm_notebook` for progress bars in Jupyter notebooks. Defaults to False.

  Raises:
    ValueError: If the lengths of the `fields`, `parses`, and `elementsList` parameters are not the same.

  Returns:
    pandas.DataFrame: A list of DataFrames containing the extracted elements for each element batch.
  """
  if len(fields) != len(parses) or len(fields) != len(elementsList):
    raise ValueError("different length: (len(fields)!=len(parses) or len(fields)!=len(elements))")

  print(f"DataFrame.Extraction_Element: {fields}")
  result_list = []  # List to store the resulting DataFrames

  tqdm_func = tqdm.tqdm_notebook if use_notebook else tqdm.tqdm

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
    progressBar_1 = tqdm_func(total=num_element_batches, unit="element batch", desc=f"Field {field_idx}/{len(fields)}")
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
          data = parse(str(row[field]))
          feature = [1 if element in data else 0 for element in element_batch]
          elements_data.append(feature)  # Append parsed data for each row in the batch

        del row_batch_data  # Delete row batch data to free memory

        progressBar_2.update(1)
      progressBar_2.close()

      columns = [elements_fieldName[element] for element in element_batch]
      result_df = pandas.DataFrame(numpy.array(elements_data), columns=columns)
      result_df_list.append(result_df)  # Append DataFrame for the batch to the list
      del elements_data  # Delete elements data to release memory

      progressBar_1.update(1)
    progressBar_1.close()

    result_list.append(result_df_list)  # Append the list of DataFrames for the field
    del result_df_list  # Delete result DataFrame list to free memory

    progressBar_0.update(1)
  progressBar_0.close()

  return result_list
def Extraction_Element(dataFrame: pandas.DataFrame, field_data: list[dict[str, object]], elementBatch_size=1, rowBatch_size=65536, use_notebook=False) -> list[pandas.DataFrame]:
  if len(field_data) == 0:
    raise ValueError("field_data cannot be empty")

  print(f"DataFrame.Extraction_Element: {field_data}")
  result_list = []  # List to store the resulting DataFrames

  tqdm_func = tqdm.tqdm_notebook if use_notebook else tqdm.tqdm

  progressBar_0 = tqdm_func(total=len(field_data), unit="field", desc="Fields:")
  for field_idx, field_info in enumerate(field_data, start=1):
    field: str | object | None = field_info.get("field")
    parse: object | None = field_info.get("parse")
    elements: list[str] | object | None = field_info.get("elements")

    elements_fieldName = dict()
    for element in elements: # type: ignore
      element_fieldName = field + "_" + element # type: ignore
      while element_fieldName in dataFrame:
        element_fieldName += "_"
      elements_fieldName[element] = element_fieldName

    # result_df_list = []  # List to store DataFrames for each batch

    # Batch processing for elements
    num_element_batches = int(numpy.ceil(len(elements) / elementBatch_size)) # type: ignore
    progressBar_1 = tqdm_func(total=len(elements), unit="element", desc=f"Field {field_idx}/{len(field_data)}: Elements:") # type: ignore
    for element_batch_idx in range(num_element_batches):
      start_element_idx = element_batch_idx * elementBatch_size
      end_element_idx = min((element_batch_idx + 1) * elementBatch_size, len(elements)) # type: ignore
      element_batch = elements[start_element_idx:end_element_idx]  # type: ignore # Get a batch of elements

      # Batch processing for rows
      elements_data = []  # List to store parsed data for each batch of elements
      num_row_batches = int(numpy.ceil(len(dataFrame) / rowBatch_size))
      progressBar_2 = tqdm_func(total=len(dataFrame), unit="row", desc=f"Field {field_idx}/{len(field_data)}: Element {element_batch_idx*elementBatch_size}/{len(elements)}: Rows:") # type: ignore
      for row_batch_idx in range(num_row_batches):
        start_row_idx = row_batch_idx * rowBatch_size
        end_row_idx = min((row_batch_idx + 1) * rowBatch_size, len(dataFrame))
        row_batch_data = dataFrame.iloc[start_row_idx:end_row_idx]  # Get a batch of rows

        for index, row in row_batch_data.iterrows():
          data: list[str] = parse(str(row[field])) # type: ignore
          feature = [1 if element in data else 0 for element in element_batch]
          elements_data.append(feature)  # Append parsed data for each row in the batch

        del row_batch_data  # Delete row batch data to free memory

        progressBar_2.update(end_row_idx-start_row_idx)
      progressBar_2.close()

      columns = [elements_fieldName[element] for element in element_batch]
      result_df = pandas.DataFrame(numpy.array(elements_data), columns=columns)
      # result_df_list.append(result_df)  # Append DataFrame for the batch to the list
      result_list.append(result_df)  # Append the list of DataFrames for the field
      del elements_data  # Delete elements data to release memory

      # print(f"{elementBatch_size} if ({(element_batch_idx+1)*elementBatch_size} < {len(elements)}) else {(len(elements)-element_batch_idx*elementBatch_size)}") # type: ignore
      progressBar_1.update(elementBatch_size if ((element_batch_idx+1)*elementBatch_size < len(elements)) else (len(elements)-element_batch_idx*elementBatch_size)) # type: ignore
    progressBar_1.close()

    # result_list.append(result_df)  # Append the list of DataFrames for the field
    # result_list.append(result_df_list[0])  # Append the list of DataFrames for the field
    # del result_df_list  # Delete result DataFrame list to free memory

    progressBar_0.update(1)
  progressBar_0.close()

  return result_list
# %%
def Filter_Percentile(dataFrame: pandas.DataFrame, fields: list[str], round=1, borderCropping=4, borderPercentile=[25.0,75.0], whisker=1.5, whiskers=[1.5, 1.5], plotDisplay=False, plotsDisplay=False, use_notebook=False) -> pandas.DataFrame: 
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
def Filter_NormalDistribution(dataFrame: pandas.DataFrame, fields: list[str], round=1, stddevRange=2.0, stddevRanges=[2.0, 2.0], plotDisplay=False, plotsDisplay=False, use_notebook=False) -> pandas.DataFrame: 
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
def Filter_Features(features: list[pandas.DataFrame], target: pandas.DataFrame, threshold=0.8, use_notebook=False) -> list[pandas.DataFrame]:
  # Step 0: Check the length of feature and target
  for feature in features:
    if len(feature) != len(target):
      raise ValueError(f"Length of feature and target should be the same. {len(feature)} != {len(target)} : {feature}")
  tqdm_func = tqdm.tqdm_notebook if use_notebook else tqdm.tqdm

  result_list: list[pandas.DataFrame] = []

  for feature in features:
    feature_df = pd.DataFrame(feature)  # Convert feature to DataFrame if necessary

    # Combine feature and target into a single DataFrame
    data = pd.concat([feature_df, target], axis=1)

    # Step 1: Remove rows without data in feature and target
    data.dropna(subset=feature_df.columns, inplace=True)

    # Step 2: Filter features based on correlation threshold
    corr_matrix = data.corr().abs()  # Calculate absolute correlation matrix
    correlated_features = set()  # Set to store highly correlated features

    # Find highly correlated features
    for i in range(len(corr_matrix.columns)):
      for j in range(i):
        if corr_matrix.iloc[i, j] > threshold: # type: ignore
          colname = corr_matrix.columns[i]
          correlated_features.add(colname)

    # Filter out non-existent column names
    correlated_features = correlated_features.intersection(feature_df.columns)

    # Step 3: Return the filtered feature DataFrame
    result_list.append(feature_df.drop(columns=correlated_features)) # type: ignore

  return result_list
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
def Extraction_Filter_NormalDistribution_(dataFrame: pandas.DataFrame, fields: list[str], field_analyzes: list, stddevRange=2.0, stddevRanges=[2.0, 2.0], use_notebook=False) -> list[list]:
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
    progressBar_1 = tqdm_func(total=len(dataFrame[fields[fieldIndex]].dropna().to_list()), unit="feature", desc=f"Field {fieldIndex+1}/{len(fields)}")
    for domain in dataFrame[fields[fieldIndex]].dropna().to_list():
      population += field_analyzes[fieldIndex](domain)
      progressBar_1.update(1)
    progressBar_1.close()
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
def Extraction_Filter_NormalDistribution(dataFrame: pandas.DataFrame, field_data: list[dict[str, object]], stddevRange=2.0, stddevRanges=[2.0, 2.0], use_notebook=False) -> list[list]:
  tqdm_func = tqdm.tqdm_notebook if use_notebook else tqdm.tqdm
  # stddevRange will override stddevRanges
  if stddevRange!=2:
    stddevRanges[0] = stddevRange
    stddevRanges[1] = stddevRange
  result = []
  print(f"DataFrame.Extraction_Filter_NormalDistribution: {[field_info.get('field') for field_info in field_data]}")
  progressBar_0 = tqdm_func(total=len(field_data), unit="field", desc="Fields")
  # for fieldIndex in range(len(fields)):
  for field_idx, field_info in enumerate(field_data, start=1):
    field: str | object | None = field_info.get("field")
    parse: object | None = field_info.get("parse")
    # Extract data
    population = []
    progressBar_1 = tqdm_func(total=len(dataFrame[field].dropna().to_list()), unit="feature", desc=f"Field {field_idx+1}/{len(field_data)}")
    for domain in dataFrame[field].dropna().to_list():
      population += parse(domain) # type: ignore
      progressBar_1.update(1)
    progressBar_1.close()
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