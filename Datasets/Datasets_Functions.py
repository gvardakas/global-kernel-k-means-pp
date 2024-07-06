import importlib

def function_get_dataset(dataset_name, module_name, batch_size):
    module = importlib.import_module(module_name)
    function = getattr(module, "get_dataset")
    
    return function(dataset_name, batch_size)
        
        
        
        
        
        
        
