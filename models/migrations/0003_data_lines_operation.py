# Migration to create DataLinesOperation table
from django.db import migrations, models


class Migration(migrations.Migration):
    
    dependencies = [
        ('models', '0002_machine'),
    ]
    
    operations = [
        migrations.CreateModel(
            name='DataLinesOperation',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('machine_id', models.IntegerField()),
                ('date_time', models.DateTimeField(auto_now_add=True)),
                ('is_added_for_learning', models.BooleanField(blank=True, null=True)),
                ('is_added_for_solving', models.BooleanField(blank=True, null=True)),
            ],
            options={
                'db_table': 'DataLinesOperation',
                'verbose_name': 'Data Lines Operation',
                'verbose_name_plural': 'Data Lines Operations',
                'indexes': [
                    models.Index(fields=['machine_id'], name='data_lines_machine_id_idx'),
                ],
            },
        ),
    ]



