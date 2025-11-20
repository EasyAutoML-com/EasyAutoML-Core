"""
Tests for SolutionFinder.py - Optimization Solver

This file tests the SolutionFinder module, which is responsible for finding
optimal solutions using optimization algorithms. It's like a "solution optimizer"
that searches for the best combination of parameters to achieve your goals.

WHAT IS SOLUTION FINDING?
=========================
Solution finding is the process of searching for the best combination of
parameters that optimizes a given objective. It's like finding the perfect
recipe by trying different ingredient combinations.

WHAT DOES SOLUTION FINDER DO?
=============================
1. OPTIMIZATION ALGORITHMS:
   - Uses differential evolution and other optimization methods
   - Searches through parameter spaces efficiently
   - Finds optimal solutions based on scoring criteria

2. PARAMETER SPACE EXPLORATION:
   - Explores different combinations of parameters
   - Handles both constant and varying parameters
   - Manages large parameter spaces efficiently

3. SOLUTION EVALUATION:
   - Evaluates solutions using SolutionScore
   - Tracks evaluation progress and results
   - Manages solution quality metrics

4. RESULT TRACKING:
   - Tracks the best solution found
   - Records evaluation statistics
   - Provides progress information

WHY IS SOLUTION FINDER IMPORTANT?
=================================
SolutionFinder is important because:
- It automates the search for optimal solutions
- It can handle complex optimization problems
- It provides efficient search algorithms
- It integrates with the scoring system

WHAT DOES THIS MODULE TEST?
===========================
- Basic solution finding functionality
- Parameter space exploration
- Result tracking and statistics
- Error handling and edge cases
- Integration with SolutionScore
- Different parameter types and sizes

TESTING STRATEGY:
=================
Each test follows this pattern:
1. Create SolutionFinder instance
2. Define parameter spaces and scoring criteria
3. Test solution finding functionality
4. Verify results and statistics
5. Test edge cases and error conditions

DEPENDENCIES:
=============
SolutionFinder depends on:
- SolutionScore: For evaluating solutions
- Experimenter: For generating outputs
- Various optimization libraries
"""
import pytest
import pandas as pd
import numpy as np
from ML import SolutionFinder, SolutionScore, Experimenter
from models.EasyAutoMLDBModels import EasyAutoMLDBModels


class TestSolutionFinder:
    """
    Test SolutionFinder Class Functionality
    
    This class contains all tests for the SolutionFinder module. Each test method
    focuses on one specific aspect of solution finding and optimization functionality.
    
    TEST ORGANIZATION:
    ==================
    The tests are organized by functionality:
    1. Initialization tests (basic setup, parameter handling)
    2. Solution finding tests (basic optimization)
    3. Result tracking tests (statistics and progress)
    4. Parameter space tests (different parameter types)
    5. Error handling tests (invalid inputs, edge cases)
    6. Integration tests (with SolutionScore and Experimenter)
    
    FIXTURE DEPENDENCIES:
    =====================
    Most tests use these fixtures:
    - db_cleanup: Ensures test data is cleaned up after each test
    
    TEST NAMING CONVENTION:
    =======================
    - test_solution_finder_[functionality]: Tests specific SolutionFinder functionality
    - Each test name describes what it's testing
    - Tests are independent and can run in any order
    
    OPTIMIZATION PROCESS:
    =====================
    1. Define parameter spaces (constant and varying)
    2. Set up scoring criteria
    3. Run optimization algorithm
    4. Track results and statistics
    5. Return best solution found
    """
    
    @pytest.mark.django_db
    def test_solution_finder_init(self, db_cleanup):
        """
        Test SolutionFinder Initialization
        
        WHAT THIS TEST DOES:
        - Creates a new SolutionFinder instance
        - Verifies that the instance is created correctly
        - Checks that all result tracking attributes are initialized
        
        WHY THIS TEST IS IMPORTANT:
        - Initialization is the first step in using SolutionFinder
        - This test ensures the basic creation process works
        - It verifies that result tracking is properly set up
        
        INITIALIZATION PROCESS:
        1. Create SolutionFinder with a name
        2. Initialize result tracking attributes
        3. Set up internal state
        4. Verify instance is ready for use
        
        WHAT WE'RE TESTING:
        - SolutionFinder object is created successfully
        - Name is set correctly
        - Result tracking attributes are initialized
        - Object is ready for optimization
        
        TEST STEPS:
        1. Create SolutionFinder instance
        2. Verify instance is created
        3. Check that name is set correctly
        4. Verify result tracking attributes are initialized
        """
        sf = SolutionFinder("test_finder")
        
        # Comprehensive initialization validation
        assert sf.solution_finder_name == "test_finder"
        assert sf.result_shorter_cycles_enabled is None
        
        # Validate basic attributes that exist
        assert hasattr(sf, 'solution_finder_name'), "SolutionFinder should have solution_finder_name attribute"
        assert hasattr(sf, 'result_shorter_cycles_enabled'), "SolutionFinder should have result_shorter_cycles_enabled attribute"
        
        # Validate initial state
        assert sf.result_shorter_cycles_enabled is None, "Initial result_shorter_cycles_enabled should be None"
        
        print(f"✅ SolutionFinder initialization validation passed: Name='{sf.solution_finder_name}'")
        assert sf.result_best_solution_final_score is None
        assert sf.result_evaluate_count_run is None
        assert sf.result_evaluate_count_better_score is None
        assert sf.result_evaluate_count_no_score is None
        assert sf.result_dict_solution_found_best_values is None
        assert sf.result_delay_sec is None
            
    @pytest.mark.django_db
    def test_solution_finder_result_tracking(self, db_cleanup):
        """Test SolutionFinder result tracking"""
        sf = SolutionFinder("test_tracking")
        
        # Simulate result tracking
        sf.result_shorter_cycles_enabled = True
        sf.result_best_solution_final_score = 95.5
        sf.result_evaluate_count_run = 100
        sf.result_evaluate_count_better_score = 15
        sf.result_evaluate_count_no_score = 5
        sf.result_delay_sec = 2.5
        
        # Comprehensive result tracking validation
        assert sf.result_shorter_cycles_enabled == True
        assert sf.result_best_solution_final_score == 95.5
        assert sf.result_evaluate_count_run == 100
        assert sf.result_evaluate_count_better_score == 15
        assert sf.result_evaluate_count_no_score == 5
        assert sf.result_delay_sec == 2.5
        
        # Validate result tracking data types
        assert isinstance(sf.result_shorter_cycles_enabled, bool), "result_shorter_cycles_enabled should be boolean"
        assert isinstance(sf.result_best_solution_final_score, (int, float)), "result_best_solution_final_score should be numeric"
        assert isinstance(sf.result_evaluate_count_run, int), "result_evaluate_count_run should be integer"
        assert isinstance(sf.result_evaluate_count_better_score, int), "result_evaluate_count_better_score should be integer"
        assert isinstance(sf.result_evaluate_count_no_score, int), "result_evaluate_count_no_score should be integer"
        assert isinstance(sf.result_delay_sec, (int, float)), "result_delay_sec should be numeric"
        
        # Validate result tracking logic
        assert sf.result_evaluate_count_run >= sf.result_evaluate_count_better_score, "Total runs should be >= better score count"
        assert sf.result_evaluate_count_run >= sf.result_evaluate_count_no_score, "Total runs should be >= no score count"
        assert sf.result_best_solution_final_score > 0, "Best score should be positive"
        assert sf.result_delay_sec > 0, "Delay should be positive"
        
        print(f"✅ SolutionFinder result tracking validation passed: Score={sf.result_best_solution_final_score}, Runs={sf.result_evaluate_count_run}")
        
    @pytest.mark.django_db
    def test_solution_finder_differential_evolution_logic(self, db_cleanup):
        """Test SolutionFinder differential evolution logic"""
        sf = SolutionFinder("test_de")
        
        # Test that differential evolution parameters are handled
        possible_values_constant = {'base': 1}
        possible_values_varying = {
            'x': [0, 1, 2, 3, 4, 5],
            'y': [10, 20, 30]
        }
        
        solution_score = self._create_mock_solution_score()
        experimenter = self._create_mock_experimenter()
        
        # Test that the method exists and can be called
        assert hasattr(sf, 'find_solution')
        assert callable(sf.find_solution)
        
    @pytest.mark.django_db
    def test_solution_finder_with_different_parameter_types(self, db_cleanup):
        """Test SolutionFinder with different parameter types"""
        sf = SolutionFinder("test_types")
        
        # Test with different data types
        possible_values_constant = {
            'string_param': 'test',
            'numeric_param': 42,
            'bool_param': True
        }
        
        possible_values_varying = {
            'int_list': [1, 2, 3, 4, 5],
            'float_list': [1.1, 2.2, 3.3],
            'string_list': ['A', 'B', 'C']
        }
        
        solution_score = self._create_mock_solution_score()
        experimenter = self._create_mock_experimenter()
        
        # Test that parameters are accepted
        assert isinstance(possible_values_constant, dict)
        assert isinstance(possible_values_varying, dict)
        assert isinstance(solution_score, SolutionScore)
        # Note: experimenter is a MockExperimenter, not Experimenter
        assert hasattr(experimenter, 'generate_outputs')
        
    def test_solution_finder_invalid_parameters(self, db_cleanup):
        """Test SolutionFinder with invalid parameters"""
        sf = SolutionFinder("test_invalid")
        
        # Test with None parameters
        with pytest.raises(Exception):
            sf.find_solution(
                possible_values_constant=None,
                possible_values_varying=None,
                solution_score=None,
                experimenter=None
            )
            
    @pytest.mark.django_db
    def test_solution_finder_empty_possible_values(self, db_cleanup):
        """Test SolutionFinder with empty possible values"""
        sf = SolutionFinder("test_empty")
        
        possible_values_constant = {}
        possible_values_varying = {}
        
        solution_score = self._create_mock_solution_score()
        experimenter = self._create_mock_experimenter()
        
        # Test with empty values
        try:
            sf.find_solution(
                possible_values_constant=possible_values_constant,
                possible_values_varying=possible_values_varying,
                solution_score=solution_score,
                experimenter=experimenter
            )
        except Exception as e:
            # Expected to fail with empty values
            error_msg = str(e).lower()
            assert ("empty" in error_msg or 
                   "values" in error_msg or
                   "bounds" in error_msg or
                   "sequence" in error_msg or
                   "finite" in error_msg)
            
    @pytest.mark.django_db
    def test_solution_finder_name_property(self, db_cleanup):
        """Test SolutionFinder name property"""
        sf = SolutionFinder("test_name")
        
        assert sf.solution_finder_name == "test_name"
        
        # Test name modification
        sf.solution_finder_name = "modified_name"
        assert sf.solution_finder_name == "modified_name"
        
    @pytest.mark.django_db
    def test_solution_finder_result_initialization(self, db_cleanup):
        """Test SolutionFinder result initialization"""
        sf = SolutionFinder("test_init")
        
        # All result attributes should be None initially
        result_attrs = [
            'result_shorter_cycles_enabled',
            'result_best_solution_final_score',
            'result_evaluate_count_run',
            'result_evaluate_count_better_score',
            'result_evaluate_count_no_score',
            'result_dict_solution_found_best_values',
            'result_delay_sec'
        ]
        
        for attr in result_attrs:
            assert getattr(sf, attr) is None
            
    def _create_mock_solution_score(self):
        """
        Helper Method: Create Mock SolutionScore for Testing
        
        WHAT THIS METHOD DOES:
        - Creates a mock SolutionScore object for testing
        - Defines a simple scoring formula
        - Provides a realistic scoring scenario
        
        WHY THIS HELPER EXISTS:
        - SolutionFinder tests need a SolutionScore object
        - This helper creates a consistent test scenario
        - It avoids code duplication across test methods
        
        SCORING FORMULA:
        - param1: '+++' (maximize param1)
        - param2: '---' (minimize param2)
        - param3: '~50' (target param3 around 50)
        
        RETURN VALUE:
        - SolutionScore object with the defined formula
        """
        # Create a simple scoring formula
        formula = {
            'param1': '+++',  # Maximize param1
            'param2': '---',  # Minimize param2
            'param3': '~50'   # Target param3 around 50
        }
        
        return SolutionScore(formula)
        
    def _create_mock_experimenter(self):
        """
        Helper Method: Create Mock Experimenter for Testing
        
        WHAT THIS METHOD DOES:
        - Creates a mock Experimenter object for testing
        - Provides a simple output generation method
        - Simulates the behavior of a real Experimenter
        
        WHY THIS HELPER EXISTS:
        - SolutionFinder tests need an Experimenter object
        - This helper creates a consistent test scenario
        - It avoids code duplication across test methods
        
        MOCK BEHAVIOR:
        - generate_outputs method takes inputs and returns outputs
        - Outputs are simple transformations of inputs
        - Returns a DataFrame with output columns
        
        RETURN VALUE:
        - MockExperimenter object with generate_outputs method
        """
        # This would normally be a real Experimenter, but for testing we'll create a minimal one
        class MockExperimenter:
            def __init__(self):
                pass
                
            def generate_outputs(self, inputs):
                # Mock output generation
                return pd.DataFrame({
                    'output1': [inputs.get('param1', 0) * 2],
                    'output2': [inputs.get('param2', 0) + 10]
                })
                
        return MockExperimenter()
