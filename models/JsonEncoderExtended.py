import pandas as pd
from numpy import float32, float64, ndarray, int32, int64, bool_, isinf, isneginf, isnan
from django.core.serializers.json import DjangoJSONEncoder
from SharedConstants import DatasetColumnDataType


class JsonEncoderExtended(DjangoJSONEncoder):
    """Extended JSON encoder for handling pandas, numpy, and custom types"""

    # convert everything not compatible with Django/sql - to make able to write the json inside the django-mysql table
    def default(self, o):
        #raise ValueError
         if pd.isna(o):
             return None
         elif isinstance(o, float32) or isinstance(o, float64):
             if isinf(o) or isneginf(o) or isnan(o):
                 return None
             else:
                 return float(o)
         elif isinstance(o, int32) or isinstance(o, int64):
             return int(o)
         elif isinstance(o, ndarray):
             return o.tolist()
         elif isinstance(o, bool_):
             return bool(o)
         elif isinstance(o, DatasetColumnDataType):
             return o._name_
         else:
             return super().default(o)


