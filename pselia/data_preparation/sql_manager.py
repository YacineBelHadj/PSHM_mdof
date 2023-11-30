
from pselia.config_elia import get_data_path
class QueryBuilder:
    table_names = {'psd_original': ['processed_data', 'metadata'],
                   'psd_notch': ['notch_vas', 'original_psd']}
    def __init__(self):
        self.table = None
        self.columns = []
        self.conditions = []
    
    def set_path(self, setting_name:str,type_of_data:str ='psd_original'):
        if type_of_data not in self.table_names.keys():
            raise ValueError(f"Invalid type_of_data: {type_of_data}. \
                             Must be one of {self.table_names.keys()}")
        self.path = get_data_path(setting_name, type_of_data)
        self.type_of_data = type_of_data
        return self
    
    def set_table(self, table:str ='processed_data'):
        if table not in self.table_names[self.type_of_data]:
            raise ValueError(f"Invalid table: {table}. \
                             Must be one of {self.table_names[self.type_of_data]}")  
        self.table = table
        return self

    def add_column(self, column):
        self.columns.append(column)
        return self
    
    def add_condition(self, condition):
        self.conditions.append(condition)
        return self

    def build(self):
        query = f"SELECT {', '.join(self.columns)} FROM {self.table}"
        if self.conditions:
            query += f" WHERE {' AND '.join(self.conditions)}"
        return query

