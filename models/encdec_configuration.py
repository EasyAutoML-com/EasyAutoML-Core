from sys import getsizeof
import json
from django.db import models

# JSON Encoder
try:
    from models.JsonEncoderExtended import JsonEncoderExtended
except ImportError:
    from django.core.serializers.json import DjangoJSONEncoder
    JsonEncoderExtended = DjangoJSONEncoder


class EncDecConfiguration(models.Model):
    """
    This table contains EncDec configuration of the machine as JSON object
    """

    enc_dec_config = models.JSONField(default=dict, blank=True, null=True, encoder=JsonEncoderExtended)

    class Meta:
        app_label = 'models'
        verbose_name = "EncDec configuration"
        verbose_name_plural = "EncDec configurations"
        db_table = "machine_encdecconfiguration"

    def __str__(self):
        return (f"Size: {getsizeof(self.enc_dec_config)} \n" + json.dumps(self.enc_dec_config)) \
            if self.enc_dec_config else "undefined"
