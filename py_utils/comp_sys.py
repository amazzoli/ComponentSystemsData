import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, lil_matrix


class ComponentSystem(object):

  """
  Class storing all the properties of a component system and computing the
  main statistical properties
  """

  def __init__(self, label, objects, components, count_sparse, check_consistency=False):
    """
    Args:
    - objects: pandas table with the list of objects.
    - components: pandas table with the list of components.
    - sparse_counts: pandas table with the sparse counts (i,j,n coordinate form).
    """
    
    self.label = label
    self.objects = objects
    self.components = components

    # Constructing the sparse lil_matrix. It is probably more efficient to pass
    # from the coo_matrix given the way we store the values in count_sparse
    row_col_indexes = count_sparse['object_id'], count_sparse['component_id']
    self.sparse_mat = coo_matrix((count_sparse['count'].values, row_col_indexes))
    self.sparse_mat = self.sparse_mat.tolil().transpose()

    if check_consistency:
      self.check_consistency()


  def __str__(self):
    return '"' + self.label + '" component-system with ' + str(len(self.objects)) + " objects and "  + str(len(self.components)) + " components."
    
    
  def comps_in_obj(self, obj_id):
    """
    Returns the table of the components in the object id
    """
    obj_array = self.sparse_mat[:,obj_id].toarray().T[0]
    obj_comps = np.where(obj_array > 0)[0]
    obj_comp_table = self.components.loc[obj_comps].drop(['abundance', 'occurrence'], axis=1)
    obj_comp_table['count'] = obj_array[obj_comps]
    return obj_comp_table.sort_values('count', ascending=False).copy()


  def objs_of_comp(self, comp_id):
    """
    Returns the table of the objects in which the component is present
    """
    comp_array = self.sparse_mat[comp_id].toarray()[0]
    comp_objs = np.where(comp_array > 0)[0]
    comp_obj_table = self.objects.loc[comp_objs].drop(['size', 'vocabulary'], axis=1)
    comp_obj_table['count'] = comp_array[comp_objs]
    return comp_obj_table.sort_values('count', ascending=False).copy()


  def check_consistency(self):
    """
    Check if the identifiers and the summation of rows and columns in the objects
    and component tables are consistent with the sparse count matrix.
    """

    self.consistent = True

    # Checking the number of objects
    sp_sizes = np.array(self.sparse_mat.sum(axis=0))[0]
    no_zero_objects = self.objects[self.objects['size'] > 0] # Objects with size 0 are not checked
    if len(sp_sizes) != max(no_zero_objects.index) + 1:
      print('The number of objects in the sparse matrix is different than in the objects table')
      self.consistent = False

    # Checking that the sparse matrix size matche with the one in the table
    sizes_series = pd.DataFrame(sp_sizes)
    sizes_series[1] = sizes_series.index.map(no_zero_objects['size']).fillna(0)
    sizes_series[2] = sizes_series[0] != sizes_series[1]
    n_no_match  = np.sum(sizes_series[2])
    if n_no_match > 0:
      print('Sizes reported in the table do not match the ones in the sparse matrix')
      self.consistent = False

    # Checking the number of components
    sp_abundances = np.array(self.sparse_mat.sum(axis=1)).T[0]
    no_zero_components = self.components[self.components['abundance'] > 0]
    if len(sp_abundances) != max(no_zero_components.index) + 1:
      print('The number of components in the sparse matrix is different than in the components table')
      self.consistent = False

    # Checking that the sparse matrix abundances matche with the one in the table
    ab_series = pd.DataFrame(sp_abundances)
    ab_series[1] = ab_series.index.map(no_zero_components['abundance']).fillna(0)
    ab_series[2] = ab_series[0] != ab_series[1]
    n_no_match  = np.sum(ab_series[2])
    if n_no_match > 0:
      print('Abundances reported in the table do not match the ones in the sparse matrix')
      self.consistent = False

    if self.consistent:
      print('The tables are consistent')




def read_metadata(repo_folder='/content/ComponentSystemsData/'):
  """
  Return a pandas table with the metadata of the datasets. The repo_folder is
  set for the analysis in a colab notebook, change the folder if you are running
  the code locally.
  """

  return pd.read_csv(repo_folder + 'metadata.tsv', sep='\t', index_col=0)



def load_system(label, repo_folder='/content/ComponentSystemsData/'):
  """
  Return a ComponentSystem object for the given label. 
  Labels are listed in the metadata.tsv file. 
  The repo_folder is set for the analysis in a colab notebook, change the folder 
  if you are running the code locally.
  """

  metadata = read_metadata(repo_folder)
  if label not in metadata.index:
    raise ValueError('Label {} not found in metadata.tsv'.format(label))

  data_folder = 'datasets/{}/data/'.format(label)
  objects = pd.read_csv(repo_folder + data_folder + 'objects.tsv', sep='\t', index_col=0)
  components = pd.read_csv(repo_folder + data_folder + 'components.tsv', sep='\t', index_col=0)
  count_sparse = pd.read_csv(repo_folder + data_folder + 'count_sparse.zip', sep='\t', compression='zip')

  return ComponentSystem(label, objects, components, count_sparse)


