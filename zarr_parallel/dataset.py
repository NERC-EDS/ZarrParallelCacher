__author__    = "Daniel Westwood"
__contact__   = "daniel.westwood@stfc.ac.uk"
__copyright__ = "Copyright 2026 United Kingdom Research and Innovation"

from typing import Union, Callable
import xarray as xr

import hashlib

class ZarrParallelDataset:
    description = 'Xarray-like task container that can render selectors for the ZarrParallelAssembler'

    def __init__(
            self, 
            croissant_file: Union[str,None] = None,
            uri: Union[str,None] = None, 
            engine: Union[str,None] = None,
            pre_transforms: Union[list,None] = None,
            variable_transforms: Union[list,None] = None,
            array_object: xr.DataArray = None,
            **kwargs):
        
        if croissant_file is None and uri is None:
            raise ValueError(
                'A data source must be provided (croissant_file or uri)'
            )
        
        self.croissant_file = croissant_file
        if croissant_file:
            self._access_croissant(croissant_file)
        
        self.kwargs = kwargs
        self.uri    = uri
        self.engine = engine

        self._await_method = None
        self._await_variable = None

        self._lock = False

        if array_object is not None:
            self._arr = array_object
        else:
            self._arr = xr.open_dataset(self.uri, engine=self.engine, **self.kwargs)

        # Could be used by the dataloader when caching selectors to recombine selectors from the same source.
        self._common_transform_hash = hashlib.md5(str(pre_transforms).encode('utf-8')).hexdigest()

        self.pre_transforms = pre_transforms or []
        self.variable_transforms = variable_transforms or None

    def lock(self):
        """
        When locked this object behaves like a dictionary, with an additional container for the array.
        """
        self._lock = True

    def unlock(self):
        """
        When unlocked this object behaves like an xarray dataset that can be sliced as needed."""
        self._lock = False

    def get_array(self):
        return self._arr

    def apply_cfunc(
            self, 
            xarray_method: Callable = None,
            copy: bool = False,
            **kwargs
        ) -> Union[None, object]:

        if xarray_method is None:
            if self._await_method is None:
                raise ValueError(
                    'No method specified'
                )
            xarray_method = self._await_method
            self._await_method = None

        xarray_kwargs = {}
        if xarray_method == 'isel' or xarray_method == 'sel':
            for k, v in kwargs.items():
                if isinstance(v,list):
                    xarray_kwargs[k] = slice(*v)
                else:
                    xarray_kwargs[k] = v

        if self._await_variable is None:
            new_arr        = getattr(self._arr, xarray_method)(**xarray_kwargs)
            pre_transforms = self.pre_transforms + [{"type": xarray_method} | kwargs]
        else:
            new_arr        = getattr(self._arr[self._await_variable], xarray_method)(**xarray_kwargs)
            variable_transforms = self.variable_transforms + [{"type": xarray_method} | kwargs]
            self._await_variable = None

        if copy:
            return ZarrParallelDataset(
                uri=self.uri,
                engine=self.engine,
                pre_transforms=pre_transforms,
                variable_transforms=variable_transforms,
                array_object=new_arr)
        else:
            self._arr = new_arr
            self.pre_transforms = pre_transforms
            self.variable_transforms = variable_transforms
    
    def __getattr__(self, attr):
        if self._lock:
            return getattr(self._get_templated_selector, attr)

        self._await_method = attr
        return self.apply_cfunc
    
    def __getitem__(self, variable):

        if self._lock:
            return self._get_templated_selector[variable]

        self._await_variable = variable
        return self

    def __repr__(self):
        return self._arr.__repr__()
    
    def _get_variables(self):
        return [v for v in self._arr.variables if v not in self._arr.dims]
    
    def _get_templated_selector(self):
        """
        Get template for selector in current state
        """
        
        vars = self._get_variables()
        return {
            "uri": self.uri,
            "engine":self.engine,
            "common": {
                "pre_transforms": self.pre_transforms,
                "pre_transform_rule": "append",  # or "override" to replace common pre-transforms with variable-specific ones
            },
            "variables": {v : self.variable_transforms for v in vars}
        }
    
    def assemble_selector(self, chunks: dict, vars: Union[list,None] = None):
        """
        Assemble a compatible selector object based on the TaskContainer's tasks.
        """
        
        # Will either render all variables with no transforms or one variable with many transforms.
        # Either way this should abide by the normal selector rules.

        selector = self._get_templated_selector()

        selector['common']['chunks'] = chunks
        if vars is not None:
            selector['variables'] = {v : self.variable_transforms for v in vars}

        return selector
    
def combine_selectors(selectors):

    merged_selector = selectors[0]

    for s in selectors[1:]:

        # Verify required values are equal
        for key in ['uri','engine']:
            if s[key] != merged_selector[key]:
                raise ValueError(
                    'Incompatible selectors: '
                    f'{s[key]}, {merged_selector[key]}')
        
        #Merge transforms, merge variables