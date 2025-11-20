# Migration to create Team table
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    
    dependencies = [
        ('models', '0003_data_lines_operation'),
    ]
    
    operations = [
        # Create Team table
        migrations.CreateModel(
            name='Team',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=200, verbose_name='Team name')),
                ('url', models.URLField(default=None, null=True, unique=True)),
                ('admin_user', models.ForeignKey(
                    blank=True,
                    null=True,
                    on_delete=django.db.models.deletion.CASCADE,
                    to=settings.AUTH_USER_MODEL
                )),
                ('permission', models.OneToOneField(
                    null=True,
                    on_delete=django.db.models.deletion.CASCADE,
                    to='auth.permission'
                )),
            ],
            options={
                'db_table': 'Team',
                'verbose_name': 'Team',
                'verbose_name_plural': 'Teams',
                'indexes': [
                    models.Index(fields=['name'], name='team_name_idx'),
                ],
            },
        ),
        # Create many-to-many relationship table for Team users
        migrations.CreateModel(
            name='Team_Users',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('team_id', models.BigIntegerField()),
                ('user_id', models.BigIntegerField()),
            ],
            options={
                'db_table': 'Team_users',
            },
        ),
    ]

