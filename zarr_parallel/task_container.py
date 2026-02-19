from typing import Union, Callable
import xarray as xr

class TaskContainer:
    description = 'Xarray-like task container that can render selectors for the ZarrParallelAssembler'

    def __init__(
            self, 
            uri: str, 
            var: str, 
            engine: Union[str,None] = None,
            pre_transforms: list = None,
            array_object: xr.DataArray = None,
            **kwargs):
        
        self.kwargs = kwargs
        self.uri    = uri
        self.engine = engine

        self.var = var

        self._await_method = None

        print(array_object)
        if array_object is not None:
            self._arr = array_object
        else:
            self._arr = xr.open_dataset(self.uri, engine=self.engine, **self.kwargs)[var]

        self.pre_transforms = pre_transforms or []

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

        new_arr        = getattr(self._arr, xarray_method)(**xarray_kwargs)
        pre_transforms = self.pre_transforms + [{"type": xarray_method} | kwargs]

        if copy:
            return TaskContainer(
                uri=self.uri,
                var=self.var,
                engine=self.engine,
                pre_transforms=pre_transforms,
                array_object=new_arr)
        else:
            self._arr = new_arr
            self.pre_transforms = pre_transforms
    
    def __getattr__(self, attr):

        self._await_method = attr
        return self.apply_cfunc

    def __repr__(self):
        return self._arr.__repr__()
    
    def assemble_selector(self, chunks: dict):
        """
        Assemble a compatible selector object based on the TaskContainer's tasks.
        """
        
        return {
            "uri": self.uri,
            "engine":self.engine,
            "common": {
                "pre_transforms": self.pre_transforms,
                "pre_transform_rule": "append",  # or "override" to replace common pre-transforms with variable-specific ones
                "chunks": chunks,  # Example chunking strategy, can be adjusted as needed,
            },
            "variables": {self.var : {}}
        }
    
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