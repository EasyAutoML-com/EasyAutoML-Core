#!/usr/bin/env python3
"""
Dependency Testing Module for AI Project

This module tests all imports and dependencies to identify missing packages
and configuration issues that prevent tests from running.

Usage:
    python test_dependencies.py
"""

import sys
import os
import importlib
import traceback
from typing import Dict, List, Tuple, Optional

class DependencyTester:
    """Test all project dependencies and imports."""
    
    def __init__(self):
        self.results = {
            'core_python': {},
            'django': {},
            'project_apps': {},
            'ai_modules': {},
            'test_modules': {},
            'missing_packages': [],
            'configuration_issues': []
        }
        
    def test_core_python_modules(self) -> Dict[str, bool]:
        """Test core Python modules that should always be available."""
        print("🔍 Testing Core Python Modules...")
        
        core_modules = [
            'os', 'sys', 'json', 'datetime', 'collections',
            'itertools', 'functools', 'typing', 'pathlib',
            'unittest', 'pytest', 'pandas', 'numpy'
        ]
        
        for module in core_modules:
            try:
                importlib.import_module(module)
                self.results['core_python'][module] = True
                print(f"  ✅ {module}")
            except ImportError as e:
                self.results['core_python'][module] = False
                self.results['missing_packages'].append(module)
                print(f"  ❌ {module}: {e}")
        
        return self.results['core_python']
    
    def test_django_modules(self) -> Dict[str, bool]:
        """Test Django and related modules."""
        print("\n🔍 Testing Django Modules...")
        
        django_modules = [
            'django',
            'django.conf',
            'django.db',
            'django.test',
            'django.contrib.auth',
            'django.contrib.contenttypes',
            'rest_framework',
            'rest_framework.authtoken'
        ]
        
        for module in django_modules:
            try:
                importlib.import_module(module)
                self.results['django'][module] = True
                print(f"  ✅ {module}")
            except ImportError as e:
                self.results['django'][module] = False
                self.results['missing_packages'].append(module)
                print(f"  ❌ {module}: {e}")
        
        return self.results['django']
    
    def test_project_apps(self) -> Dict[str, bool]:
        """Test project-specific Django apps."""
        print("\n🔍 Testing Project Apps...")
        
        # Add WWW to path if not already there
        www_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'WWW')
        if www_path not in sys.path:
            sys.path.insert(0, www_path)
        
        project_apps = [
            'machine',
            'user', 
            'eamllogger',
            'api',
            'core',
            'consulting',
            'dashboard',
            'message',
            'server',
            'team',
            'billing',
            'documentation'
        ]
        
        for app in project_apps:
            try:
                importlib.import_module(app)
                self.results['project_apps'][app] = True
                print(f"  ✅ {app}")
            except ImportError as e:
                self.results['project_apps'][app] = False
                self.results['missing_packages'].append(app)
                print(f"  ❌ {app}: {e}")
        
        return self.results['project_apps']
    
    def test_ai_modules(self) -> Dict[str, bool]:
        """Test AI module imports."""
        print("\n🔍 Testing AI Modules...")
        
        ai_modules = [
            'ML',
            'ML.Machine',
            'models.EasyAutoMLDBModels',
            'SharedConstants',
            'ML.EncDec',
            'ML.FeatureEngineeringConfiguration',
            'ML.InputsColumnsImportance',
            'ML.MachineDataConfiguration',
            'ML.MachineEasyAutoML',
            'ML.NNConfiguration',
            'ML.NNEngine',
            'ML.SolutionFinder',
            'ML.SolutionScore'
        ]
        
        for module in ai_modules:
            try:
                importlib.import_module(module)
                self.results['ai_modules'][module] = True
                print(f"  ✅ {module}")
            except ImportError as e:
                self.results['ai_modules'][module] = False
                self.results['missing_packages'].append(module)
                print(f"  ❌ {module}: {e}")
            except Exception as e:
                # Handle Django configuration errors
                self.results['ai_modules'][module] = False
                error_msg = str(e)
                if 'AUTH_USER_MODEL' in error_msg or 'models.User' in error_msg:
                    self.results['configuration_issues'].append(f"Django AUTH_USER_MODEL issue: {error_msg}")
                    print(f"  ⚠️  {module}: Django configuration error - {error_msg}")
                else:
                    self.results['missing_packages'].append(module)
                    print(f"  ❌ {module}: {e}")
        
        return self.results['ai_modules']
    
    def test_test_modules(self) -> Dict[str, bool]:
        """Test test module imports."""
        print("\n🔍 Testing Test Modules...")
        
        test_modules = [
            'tests',
            'tests.conftest',
            'tests.unit',
            'tests.unit.test_encdec',
            'tests.unit.test_machine',
            'tests.unit.test_machine_data_configuration',
            'tests.unit.test_feature_engineering_configuration',
            'tests.unit.test_inputs_columns_importance',
            'tests.unit.test_eaml_db_models',
            'tests.unit.test_machine_easy_automl',
            'tests.unit.test_nn_configuration',
            'tests.unit.test_nn_engine',
            'tests.unit.test_solution_finder',
            'tests.unit.test_solution_score'
        ]
        
        for module in test_modules:
            try:
                importlib.import_module(module)
                self.results['test_modules'][module] = True
                print(f"  ✅ {module}")
            except ImportError as e:
                self.results['test_modules'][module] = False
                self.results['missing_packages'].append(module)
                print(f"  ❌ {module}: {e}")
        
        return self.results['test_modules']
    
    def test_django_configuration(self) -> bool:
        """Test Django configuration and setup."""
        print("\n🔍 Testing Django Configuration...")
        
        try:
            import django
            from django.conf import settings
            
            print(f"  Django version: {django.get_version()}")
            
            if not settings.configured:
                print("  ⚠️  Django settings not configured")
                return False
            
            print("  ✅ Django settings configured")
            
            # Test specific settings
            required_settings = [
                'INSTALLED_APPS',
                'DATABASES',
                'AUTH_USER_MODEL',
                'SECRET_KEY'
            ]
            
            for setting in required_settings:
                if hasattr(settings, setting):
                    print(f"  ✅ {setting} is set")
                else:
                    print(f"  ❌ {setting} is missing")
                    self.results['configuration_issues'].append(f"Missing setting: {setting}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Django configuration error: {e}")
            self.results['configuration_issues'].append(f"Django config error: {e}")
            return False
    
    def test_specific_missing_packages(self) -> List[str]:
        """Test for specific packages that are commonly missing."""
        print("\n🔍 Testing Specific Missing Packages...")
        
        common_missing = [
            'colorlog',
            'django_components',
            'django_crontab',
            'django_extensions',
            'allauth',
            'widget_tweaks',
            'ckeditor',
            'ckeditor_uploader'
        ]
        
        missing = []
        for package in common_missing:
            try:
                importlib.import_module(package)
                print(f"  ✅ {package}")
            except ImportError:
                missing.append(package)
                print(f"  ❌ {package} - MISSING")
        
        return missing
    
    def test_django_path_issue(self) -> bool:
        """Test if Django apps can be found in the correct path."""
        print("\n🔍 Testing Django Path Configuration...")
        
        # Check if WWW directory exists and contains Django apps
        www_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'WWW')
        
        if not os.path.exists(www_path):
            print(f"  ❌ WWW directory not found at: {www_path}")
            self.results['configuration_issues'].append("WWW directory not found")
            return False
        
        print(f"  ✅ WWW directory found at: {www_path}")
        
        # Check for specific Django apps
        django_apps = ['machine', 'user', 'eamllogger']
        for app in django_apps:
            app_path = os.path.join(www_path, app)
            if os.path.exists(app_path):
                print(f"  ✅ {app} app directory found")
            else:
                print(f"  ❌ {app} app directory not found")
                self.results['configuration_issues'].append(f"{app} app directory not found")
        
        return True
    
    def generate_requirements_suggestion(self) -> str:
        """Generate a suggested requirements.txt based on missing packages."""
        print("\n📋 Generating Requirements Suggestion...")
        
        # Common packages that might be needed
        suggested_packages = [
            'django>=4.0',
            'djangorestframework',
            'django-crontab',
            'django-components',
            'django-extensions',
            'mock-generator',
            'django-allauth',
            'django-widget-tweaks',
            'django-ckeditor',
            'colorlog',
            'pandas',
            'numpy',
            'pytest',
            'pytest-django',
            'pytest-cov'
        ]
        
        requirements_content = "# Suggested requirements.txt\n"
        requirements_content += "# Install with: pip install -r requirements.txt\n\n"
        
        for package in suggested_packages:
            requirements_content += f"{package}\n"
        
        return requirements_content
    
    def run_all_tests(self) -> Dict:
        """Run all dependency tests."""
        print("🚀 Starting Comprehensive Dependency Testing")
        print("=" * 60)
        
        # Run all test categories
        self.test_core_python_modules()
        self.test_django_modules()
        self.test_project_apps()
        self.test_django_path_issue()
        self.test_ai_modules()
        self.test_test_modules()
        self.test_django_configuration()
        missing_packages = self.test_specific_missing_packages()
        
        # Generate summary
        self.generate_summary()
        
        return self.results
    
    def generate_summary(self):
        """Generate a comprehensive summary of test results."""
        print("\n" + "=" * 60)
        print("📊 DEPENDENCY TEST SUMMARY")
        print("=" * 60)
        
        # Count successes and failures
        total_tests = 0
        total_passed = 0
        
        for category, results in self.results.items():
            if isinstance(results, dict):
                category_total = len(results)
                category_passed = sum(1 for success in results.values() if success)
                total_tests += category_total
                total_passed += category_passed
                
                print(f"\n{category.upper().replace('_', ' ')}:")
                print(f"  Passed: {category_passed}/{category_total} ({category_passed/category_total*100:.1f}%)")
        
        print(f"\nOVERALL: {total_passed}/{total_tests} ({total_passed/total_tests*100:.1f}%)")
        
        # Show missing packages
        if self.results['missing_packages']:
            print(f"\n❌ MISSING PACKAGES ({len(self.results['missing_packages'])}):")
            for package in set(self.results['missing_packages']):
                print(f"  - {package}")
        
        # Show configuration issues
        if self.results['configuration_issues']:
            print(f"\n⚠️  CONFIGURATION ISSUES ({len(self.results['configuration_issues'])}):")
            for issue in self.results['configuration_issues']:
                print(f"  - {issue}")
        
        # Generate requirements suggestion
        if self.results['missing_packages']:
            print(f"\n💡 SOLUTION:")
            print("  1. Install missing packages:")
            print("     pip install " + " ".join(set(self.results['missing_packages'])))
            print("\n  2. Or install from requirements.txt:")
            print("     pip install -r requirements.txt")
            
            # Save requirements suggestion
            requirements_content = self.generate_requirements_suggestion()
            try:
                with open('suggested_requirements.txt', 'w') as f:
                    f.write(requirements_content)
                print(f"\n  3. Suggested requirements.txt saved to: suggested_requirements.txt")
            except Exception as e:
                print(f"  Could not save requirements file: {e}")
        
        print(f"\n🎯 NEXT STEPS:")
        if total_passed == total_tests:
            print("  ✅ All dependencies are available!")
            print("  🚀 You can now run: pytest tests/unit/ -v")
        else:
            print("  📦 Install missing packages first")
            print("  🔧 Fix configuration issues")
            print("  🧪 Then run: pytest tests/unit/ -v")

def main():
    """Main function to run dependency tests."""
    tester = DependencyTester()
    results = tester.run_all_tests()
    
    # Return exit code based on results
    if results['missing_packages'] or results['configuration_issues']:
        print(f"\n⚠️  Dependency issues found. Check the summary above.")
        return 1
    else:
        print(f"\n🎉 All dependencies are available!")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
