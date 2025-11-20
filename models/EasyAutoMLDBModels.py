from pathlib import Path
import sys
import os


class EasyAutoMLDBModels:
    """
    Function is working with django models.
    Main purpose is to provide access to all shared Django models

    DESCRIPTION:    Provide Django object instance to give Access to all table models in Django EasyAutoML.com database
                    Now uses the shared models package for better separation of concerns

    USAGE:  		To work with Django EasyAutoML.com DBModel and it's tables. Writing or reading database data.
                        More simple way than writing MySQL queries.
    """

    def __init__(self):
        # Use centralized models from /models/ directory
        project_root = Path(__file__).absolute().parent.parent
        models_path = str(project_root / "models")
        
        if models_path not in sys.path:
            sys.path.append(models_path)

        import django
        
        # Set Django settings for standalone backend
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")
        
        # Only call django.setup() if Django hasn't been set up yet
        try:
            # Check if Django apps are already configured
            from django.apps import apps
            if not apps.ready:
                django.setup()
        except (ImportError, RuntimeError) as e:
            # If Django is currently being set up (reentrant call), skip it
            if "populate() isn't reentrant" in str(e):
                pass  # Django is already being set up, continue with imports
            else:
                # If Django isn't configured yet, try to set it up
                try:
                    django.setup()
                except RuntimeError as setup_error:
                    if "populate() isn't reentrant" in str(setup_error):
                        pass  # Django is already being set up, continue with imports
                    else:
                        raise

        # Import database connections only after Django setup
        from django.db import connection, connections
        
        # Import from centralized models package
        from models import (
            User, Team, Machine,
            EasyAutoMLLogger, Graph, NNModel, DataLinesOperation, 
            MachineTableLockWrite, EncDecConfiguration
        )
        # Set placeholders for models that don't exist yet
        Server = None
        Operation = None
        ConsultingRequest = None
        Work = None
        Billing = None
        MachineBilling = None

        # Store database connections - delay cursor creation to avoid hanging
        self.connections = connections
        self.cursor = None  # Lazy initialize on first use
        
        # Register all models as instance attributes for backward compatibility
        self.User = User
        self.Machine = Machine
        self.MachineTableLockWrite = MachineTableLockWrite
        self.DataLinesOperation = DataLinesOperation
        self.EasyAutoMLLogger = EasyAutoMLLogger
        self.Server = Server  # None - model doesn't exist yet
        self.Operation = Operation  # None - model doesn't exist yet
        self.Team = Team
        self.Work = Work  # None - model doesn't exist yet
        
        # Optional models that may not be available in all environments
        if 'Graph' in locals():
            self.Graph = Graph
        if 'NNModel' in locals():
            self.NNModel = NNModel
        if 'ConsultingRequest' in locals():
            self.ConsultingRequest = ConsultingRequest
        if 'EncDecConfiguration' in locals():
            self.EncDecConfiguration = EncDecConfiguration
        if 'Billing' in locals():
            self.Billing = Billing
    
    @property
    def logger(self):
        """Return a logger instance"""
        return self.EasyAutoMLLogger()
    
    def get_cursor(self):
        """Lazily get database cursor on first use"""
        if self.cursor is None:
            from django.db import connection
            try:
                self.cursor = connection.cursor()
            except Exception as e:
                # If connection fails, log and return None
                print(f"Warning: Could not get database cursor: {e}")
                return None
        return self.cursor


