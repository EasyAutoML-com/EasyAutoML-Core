# Generated migration for models app to create all required tables
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        # No dependencies - just create User table directly
        # ('team', '0001_initial'),  # Commented out: team migrations disabled in tests
        # ('server', '0001_initial'),  # Commented out: server migrations disabled in tests
    ]

    operations = [
        # User model - This creates the user table for AUTH_USER_MODEL
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('password', models.CharField(max_length=128, verbose_name='password')),
                ('last_login', models.DateTimeField(blank=True, null=True, verbose_name='last login')),
                ('email', models.EmailField(max_length=254, null=True, unique=True)),
                ('first_name', models.CharField(blank=True, max_length=140, null=True)),
                ('last_name', models.CharField(blank=True, max_length=140, null=True)),
                ('user_profile', models.TextField(blank=True, null=True)),
                ('time_format', models.CharField(default='24H', max_length=3)),
                ('date_format', models.CharField(default='DMY', max_length=3)),
                ('date_separator', models.CharField(default='/', max_length=1)),
                ('datetime_separator', models.CharField(default=' ', max_length=1)),
                ('decimal_separator', models.CharField(default=',', max_length=1)),
                ('coupons_activated_date', models.JSONField(blank=True, default=dict)),
                ('is_super_admin', models.BooleanField(default=False)),
                ('is_superuser', models.BooleanField(default=False, verbose_name='superuser status')),
                ('is_staff', models.BooleanField(default=False, verbose_name='staff status')),
                ('is_active', models.BooleanField(default=True, verbose_name='active')),
                ('date_joined', models.DateTimeField(default=django.utils.timezone.now, verbose_name='date joined')),
                ('coupon_balance', models.DecimalField(decimal_places=10, default=0, max_digits=32, null=True)),
                ('user_balance', models.DecimalField(decimal_places=10, default=0, max_digits=32, null=True)),
                ('user_ixioo_balance', models.DecimalField(decimal_places=10, default=0, max_digits=32, null=True)),
                ('last_billing_time', models.DateTimeField(blank=True, null=True)),
            ],
            options={
                'db_table': 'user',
                'swappable': 'AUTH_USER_MODEL',
            },
        ),
        # Create User-Group relationship
        migrations.CreateModel(
            name='User_Groups',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user_id', models.BigIntegerField()),
                ('group_id', models.BigIntegerField()),
            ],
            options={
                'db_table': 'user_groups',
            },
        ),
        # Create User permissions relationship  
        migrations.CreateModel(
            name='User_User_Permissions',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user_id', models.BigIntegerField()),
                ('permission_id', models.BigIntegerField()),
            ],
            options={
                'db_table': 'user_user_permissions',
            },
        ),
        
        # Machine table lock
        migrations.CreateModel(
            name='MachineTableLockWrite',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('table_name', models.CharField(max_length=255, unique=True)),
                ('locked_at', models.DateTimeField(default=django.utils.timezone.now)),
            ],
            options={
                'db_table': 'machine_table_lock_write',
                'verbose_name': 'Machine Table Lock Write',
                'verbose_name_plural': 'Machine Table Lock Writes',
            },
        ),
        
        # NN Model
        migrations.CreateModel(
            name='NNModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('nn_model', models.BinaryField(blank=True, max_length=750000000, null=True)),
            ],
            options={
                'db_table': 'nn_model',
                'verbose_name': 'NN Model',
                'verbose_name_plural': 'NN Models',
            },
        ),
        
        # EncDec Configuration
        migrations.CreateModel(
            name='EncDecConfiguration',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('enc_dec_config', models.JSONField(blank=True, default=dict, null=True)),
            ],
            options={
                'db_table': 'encdec_configuration',
                'verbose_name': 'EncDec configuration',
                'verbose_name_plural': 'EncDec configurations',
            },
        ),
    ]
