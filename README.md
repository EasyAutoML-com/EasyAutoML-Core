# 🧠 EasyAutoML Core – Open-Source AutoML Engine

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange.svg)](https://www.tensorflow.org/)
[![Django](https://img.shields.io/badge/Django-3.2-green.svg)](https://www.djangoproject.com/)
[![License](https://img.shields.io/badge/license-AGPLv3-blue.svg)](LICENSE.md)
[![GitHub](https://img.shields.io/badge/GitHub-EasyAutoML--Core-black.svg)](https://github.com/EasyAutoML-com/EasyAutoML-Core)

**EasyAutoML Core** is a powerful, production-ready AutoML engine that automates machine learning from data ingestion to model deployment. It's designed for developers, data scientists, and organizations who want to integrate advanced machine learning capabilities into their applications with minimal code.

Licensed under the [GNU AGPLv3 License](LICENSE.md) — © 2025 Laurent Bruère / [EasyAutoML.com](https://easyautoml.com)

---

## 🎯 What is EasyAutoML Core?

**EasyAutoML Core** is the intelligent heart of the EasyAutoML ecosystem — a standalone Python library that handles:

✅ **Automatic Model Selection** – Chooses the best ML algorithm for your data  
✅ **Progressive Learning** – Starts with experimenters, evolves to neural networks  
✅ **Feature Engineering** – Automatically creates and optimizes features  
✅ **Neural Network Engine** – Built-in CNN support with TensorFlow/PyTorch  
✅ **Multi-Modal Learning** – Combines numeric, text, and categorical data  
✅ **Production Ready** – Battle-tested with Django integration

### Use Cases

- 📊 **Predictive Analytics**: Sales forecasting, demand prediction, risk assessment
- 🔮 **Regression & Classification**: Any supervised learning task
- 📝 **Text + Numeric Fusion**: Customer sentiment + behavior analysis
- 🔄 **Continuous Learning**: Models that improve as data grows
- 🚀 **Rapid Prototyping**: From Excel to trained model in minutes


---

## 📋 Table of Contents

- [What is EasyAutoML Core?](#-what-is-easyautoml-core)
- [Quick Start](#-quick-start)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Usage Examples](#-usage-examples)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Documentation](#-documentation)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Quick Start

Get up and running in 3 steps:

### 1. Clone & Install

```bash
# Clone the repository
git clone https://github.com/EasyAutoML-com/EasyAutoML-Core.git
cd EasyAutoML-Core

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Create Your First Model

```python
from ML.MachineEasyAutoML import MachineEasyAutoML
import pandas as pd

# Load your training data (DataFrame with inputs and output column)
training_data = pd.DataFrame({
    "temperature": [20, 25, 30, 35],
    "humidity": [60, 65, 70, 75],
    "sales": [100, 120, 140, 160]  # This is what we want to predict
})

# Create and train a model
model = MachineEasyAutoML(
    machine_name="sales_predictor",
    access_user_id=1  # Required for database tracking
)

# Train the model
model.learn_this_inputs_outputs(training_data)

# Make predictions on new data
new_data = pd.DataFrame({
    "temperature": [28],
    "humidity": [68]
})
predictions = model.do_predict(new_data)
print(f"Predicted sales: {predictions}")
```

### 3. Verify Installation

```bash
# Run the test suite
python -m pytest "tests/Tests All AI modules/" -v

# Or test a specific module
python -m pytest "tests/Tests All AI modules/unit/test_machine.py" -v
```

**That's it!** You now have a working AutoML engine. Check out the [Usage Examples](#-usage-examples) for more advanced scenarios.

---

## ✨ Key Features

### 🧠 Core Capabilities

| Feature | Description |
|---------|-------------|
| **Progressive Learning** | Start with experimenters, automatically evolve to neural networks |
| **Auto Feature Engineering** | Handles missing data, normalization, encoding, and feature generation |
| **Multi-Modal Intelligence** | Combines CNNs, LLMs, and traditional ML for text + numeric data |
| **Genetic Optimization** | Evolves model architectures and hyperparameters automatically |
| **Self-Tuning Pipeline** | Learns from previous runs to continuously improve |

### 🔍 Model Types Supported

- ✅ **Regression** – Predict continuous values (sales, prices, quantities)
- ✅ **Classification** – Categorize data into classes (customer segments, risk levels)
- ✅ **Text Analysis** – Process and analyze unstructured text
- ✅ **Mixed-Data Models** – Combine numerical and textual features

### 🛠️ Developer-Friendly

- 🐍 **Pure Python** – Easy integration into any Python project
- 📦 **Minimal Dependencies** – TensorFlow/PyTorch, Django, scikit-learn
- 🔌 **Django Integration** – Built-in models and database support
- 📊 **Excel/CSV Ready** – Load data directly from spreadsheets
- 🧪 **Fully Tested** – Comprehensive test suite included

### 📊 Model Evaluation

- Real-time performance metrics (Accuracy, F1, ROC-AUC, RMSE)
- Feature importance analysis
- Confusion matrices and visualizations
- Exportable model reports

---

## 📦 Installation

### Prerequisites

- **Python 3.9+** (Required - [Download Python](https://www.python.org/downloads/))
- **Git** ([Download Git](https://git-scm.com/downloads))
- **Virtual Environment** (Highly recommended)

### Step-by-Step Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/EasyAutoML-com/EasyAutoML-Core.git
cd EasyAutoML-Core
```

#### 2. Create Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages including:
- TensorFlow 2.17 (neural network engine)
- PyTorch 2.8 (alternative neural network backend)
- Django 3.2 (web framework & ORM)
- scikit-learn (machine learning utilities)
- pandas, numpy (data processing)
- And more...

#### 4. Set Up Database (Optional)

For development, SQLite is used by default (no setup needed). For production or advanced features:

```bash
# Set environment variables (create a .env file)
# See OPEN_SOURCE_CHECKLIST.md for all variables
```

#### 5. Verify Installation

```bash
# Quick test
python -c "from ML.MachineEasyAutoML import MachineEasyAutoML; print('✅ Installation successful!')"

# Run full test suite
python -m pytest "tests/Tests All AI modules/" -v
```

### Troubleshooting

**TensorFlow Installation Issues:**
```bash
# Try installing TensorFlow separately first
pip install tensorflow==2.17
```

**Windows Long Path Issues:**
```powershell
# Enable long paths in Windows
git config --global core.longpaths true
```

**Import Errors:**
- Make sure you're in the project root directory
- Ensure your virtual environment is activated
- Check that Python 3.9+ is being used: `python --version`

---

## 💻 Usage Examples

### Example 1: Simple Regression Model

Train a model to predict sales based on temperature and humidity:

```python
import pandas as pd
from ML.MachineEasyAutoML import MachineEasyAutoML

# Load your training data from Excel
training_data = pd.read_excel("sales_data.xlsx")
# Expected columns: temperature, humidity, sales

# Create and train the model
model = MachineEasyAutoML(
    machine_name="sales_predictor",
    access_user_id=1  # Required for database tracking
)

# Train the model (automatically detects 'sales' as output)
model.learn_this_inputs_outputs(training_data)

# Make predictions on new data
new_data = pd.DataFrame({
    "temperature": [28, 32],
    "humidity": [65, 70]
})
predictions = model.do_predict(new_data)
print(f"Predicted sales: {predictions}")
```

### Example 2: Start with an Experimenter, Evolve to ML

Use an experimenter initially, then improve with data:

```python
from ML.MachineEasyAutoML import MachineEasyAutoML
from ML.Experimenter import Experimenter
import pandas as pd

# Create a custom experimenter for initial predictions
class PriceExperimenter(Experimenter):
    def _do_single(self, inputs):
        base_price = inputs.get("base_price", 0)
        demand = inputs.get("demand", 0)
        price = base_price * (1 + demand * 0.15)
        return {"price": price}

# Create model with experimenter
experimenter = PriceExperimenter()
model = MachineEasyAutoML(
    machine_name="price_estimator",
    experimenter=experimenter,
    access_user_id=1
)

# Make predictions immediately (uses experimenter)
prediction = model.do_predict(pd.DataFrame({
    "base_price": [100],
    "demand": [0.5]
}))
print(f"Experimenter-based prediction: {prediction}")

# Later, provide real training data
training_data = pd.DataFrame({
    "base_price": [100, 150, 200],
    "demand": [0.3, 0.5, 0.8],
    "price": [105, 162, 228]  # Actual observed prices
})
model.learn_this_inputs_outputs(training_data)

# Now predictions use the trained ML model
prediction = model.do_predict(pd.DataFrame({
    "base_price": [175],
    "demand": [0.6]
}))
print(f"ML-based prediction: {prediction}")
```

### Example 3: Text + Numeric Mixed Data

Combine textual and numerical features:

```python
from ML.MachineEasyAutoML import MachineEasyAutoML
import pandas as pd

# Training data with text and numbers
training_data = pd.DataFrame({
    "product_description": ["Great laptop", "Slow computer", "Amazing device"],
    "price": [1000, 500, 1200],
    "customer_rating": [4.5, 2.3, 4.8]  # What we want to predict
})

# Create model (automatically handles text encoding)
model = MachineEasyAutoML(
    machine_name="rating_predictor",
    access_user_id=1
)

# Train the model
model.learn_this_inputs_outputs(training_data)

# Predict ratings for new products
new_products = pd.DataFrame({
    "product_description": ["Fast laptop"],
    "price": [950]
})
predictions = model.do_predict(new_products)
print(f"Predicted rating: {predictions}")
```

### Example 4: Classification Model

Classify customers into risk categories:

```python
from ML.MachineEasyAutoML import MachineEasyAutoML
import pandas as pd

# Training data with categorical output
training_data = pd.DataFrame({
    "income": [30000, 50000, 80000, 120000],
    "debt_ratio": [0.8, 0.5, 0.3, 0.1],
    "risk_level": ["high", "medium", "low", "low"]  # Categories
})

# Create classification model
model = MachineEasyAutoML(
    machine_name="risk_classifier",
    access_user_id=1
)

# Train
model.learn_this_inputs_outputs(training_data)

# Predict risk for new customers
new_customers = pd.DataFrame({
    "income": [60000],
    "debt_ratio": [0.4]
})
predictions = model.do_predict(new_customers)
print(f"Risk level: {predictions}")
```

---

## 📁 Project Structure

```
EasyAutoML-Core/
├── ML/                              # 🧠 Core Machine Learning Engine
│   ├── Machine.py                   # Core ML model management & orchestration
│   ├── MachineEasyAutoML.py         # Simplified AutoML interface (start here!)
│   ├── MachineEasyAutoMLAPI.py      # REST API interface
│   ├── NNEngine.py                  # Neural network training engine
│   ├── NNConfiguration.py           # Neural network architecture config
│   ├── Experimenter.py              # Experimental learning algorithms
│   ├── DataFileReader.py            # Excel/CSV data loading
│   ├── FeatureEngineering*.py       # Automatic feature engineering
│   ├── SolutionFinder.py            # Model optimization engine
│   └── EncDec.py                    # Encoding/Decoding for various data types
│
├── models/                          # 📊 Django Database Models
│   ├── machine.py                   # Machine model (stores trained models)
│   ├── user.py                      # User management
│   ├── EasyAutoMLDBModels.py        # AutoML-specific database models
│   ├── nn_model.py                  # Neural network model storage
│   └── migrations/                  # Database migrations
│
├── tests/                           # 🧪 Comprehensive Test Suite
│   ├── Tests All AI modules/
│   │   ├── unit/                    # Unit tests for all modules
│   │   │   ├── test_machine.py      # Machine class tests
│   │   │   ├── test_nn_engine.py    # Neural network tests
│   │   │   └── ...
│   │   ├── fixtures/                # Test data generators
│   │   └── conftest.py              # Pytest configuration
│   └── Create All Tests Machines/   # Test machine generators
│
├── doc/                             # 📚 Documentation
│   ├── doc_machine.md               # Machine class documentation
│   ├── doc_nn_engine.md             # Neural network engine docs
│   ├── doc_machine_easy_automl.md   # EasyAutoML API docs
│   ├── doc_experimenter.md          # Experimenter algorithm docs
│   └── ...                          # 18+ documentation files
│
├── requirements.txt                 # 📦 Python dependencies
├── settings.py                      # ⚙️ Django settings
├── SharedConstants.py               # 🔧 Global configuration
├── manage.py                        # 🐍 Django management script
├── LICENSE.md                       # 📄 AGPLv3 License
└── README.md                        # 📖 This file
```

### Key Directories Explained

- **`ML/`** - The core AutoML engine. All machine learning logic lives here. Start with `MachineEasyAutoML.py` for the simplest API.
- **`models/`** - Django ORM models for persisting machines, users, and training data.
- **`tests/`** - Full test suite with unit tests, integration tests, and test data generators.
- **`doc/`** - Comprehensive technical documentation for every module.

---

## 🔄 How It Works

EasyAutoML Core uses a **progressive learning approach** that adapts to your data:

### The Three-Stage Learning Pipeline

```
Stage 1: Experimenter Mode       Stage 2: Neural Network Mode
┌─────────────────────┐         ┌─────────────────────┐
│ Statistical learning│    →    │ Deep neural nets    │
│ Limited data (10+)  │         │ Large datasets      │
│ Pattern recognition │         │ Maximum accuracy    │
└─────────────────────┘         └─────────────────────┘
```

1. **Experimenter Mode**: As data accumulates, the system learns patterns automatically using experimenters
2. **Neural Network Mode**: With sufficient data, trains deep learning models (CNNs)

### Key Technologies

- **TensorFlow 2.17** & **PyTorch 2.8** - Neural network training
- **scikit-learn** - Classical ML algorithms
- **sentence-transformers** - Text embedding for NLP
- **Django ORM** - Model persistence and versioning
- **Genetic Algorithms** - Hyperparameter optimization

### What Makes It Different?

| Feature | Traditional ML | EasyAutoML Core |
|---------|---------------|-----------------|
| **Setup Time** | Days/weeks | Minutes |
| **Code Required** | Hundreds of lines | 3-5 lines |
| **Feature Engineering** | Manual | Automatic |
| **Model Selection** | Manual testing | Automatic optimization |
| **Text + Numbers** | Complex pipelines | Built-in support |
| **Production Ready** | Extra work | Django integration included |

---

## 📚 Documentation

Comprehensive documentation is available in the `doc/` directory:

### Core Documentation

| Document | Description |
|----------|-------------|
| [`doc_machine.md`](doc/doc_machine.md) | Core Machine class - the foundation of everything |
| [`doc_machine_easy_automl.md`](doc/doc_machine_easy_automl.md) | MachineEasyAutoML - simplified interface |
| [`doc_machine_easy_automl_api.md`](doc/doc_machine_easy_automl_api.md) | REST API documentation |
| [`doc_nn_engine.md`](doc/doc_nn_engine.md) | Neural network engine internals |
| [`doc_experimenter.md`](doc/doc_experimenter.md) | Experimenter algorithm details |

### Additional Resources

- **[`README (core EasyAutoML).md`](README%20(core%20EasyAutoML).md)** – Detailed core engine documentation
- **[`README (EasyAutoML.com).md`](README%20(EasyAutoML.com).md)** – Full platform documentation
- **[`CONTRIBUTING.md`](CONTRIBUTING.md)** – How to contribute
- **[`SECURITY.md`](SECURITY.md)** – Security policy and vulnerability reporting

---

## 🧪 Testing

EasyAutoML Core comes with a comprehensive test suite covering all machine learning modules.

### Running Tests

**Run all tests:**
```bash
python -m pytest "tests/Tests All AI modules/" -v
```

**Run specific test modules:**
```bash
# Test the core Machine class
python -m pytest "tests/Tests All AI modules/unit/test_machine.py" -v

# Test neural network engine
python -m pytest "tests/Tests All AI modules/unit/test_nn_engine.py" -v

# Test MachineEasyAutoML interface
python -m pytest "tests/Tests All AI modules/unit/test_machine_2.py" -v
```

**Run with coverage report:**
```bash
python -m pytest "tests/Tests All AI modules/" --cov=ML --cov-report=html
# Open htmlcov/index.html to view coverage report
```

**Quick smoke test:**
```bash
# Run just the first few tests to verify installation
python -m pytest "tests/Tests All AI modules/unit/test_machine.py::test_machine_creation" -v
```

### Test Structure

```
tests/
├── Tests All AI modules/
│   ├── unit/                    # Unit tests for each module
│   │   ├── test_machine.py      # Machine class tests (core)
│   │   ├── test_machine_2.py    # MachineEasyAutoML tests
│   │   ├── test_nn_engine.py    # Neural network tests
│   │   └── ...
│   ├── fixtures/                # Test data generators
│   └── conftest.py              # Pytest configuration
└── Create All Tests Machines/   # Test machine creation scripts
```

### What's Tested?

✅ Model creation and training  
✅ Experimenter-based predictions  
✅ Experimenter learning  
✅ Neural network training  
✅ Feature engineering  
✅ Data encoding/decoding  
✅ Multi-modal learning (text + numeric)  
✅ Classification and regression  
✅ Model persistence and loading
 
---

## 🤝 Contributing

We welcome contributions from the community! Here's how to get started:

### Quick Contribution Guide

1. **Fork & Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/EasyAutoML-Core.git
   cd EasyAutoML-Core
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make Your Changes**
   - Write clear, documented code
   - Follow existing code style
   - Add docstrings to new functions/classes

4. **Add Tests**
   - Create tests in `tests/Tests All AI modules/unit/`
   - Ensure all tests pass:
     ```bash
     python -m pytest "tests/Tests All AI modules/" -v
     ```

5. **Commit & Push**
   ```bash
   git add .
   git commit -m "Add: Brief description of your feature"
   git push origin feature/amazing-feature
   ```

6. **Submit Pull Request**
   - Go to GitHub and create a pull request
   - Describe your changes clearly
   - Link any related issues

### Contribution Ideas

- 🐛 **Bug Fixes** - Fix issues from the issue tracker
- ✨ **New Features** - Add new ML algorithms or capabilities
- 📝 **Documentation** - Improve docs, add examples
- 🧪 **Tests** - Increase test coverage
- 🎨 **Code Quality** - Refactoring, optimization

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for detailed guidelines.

---

## 📊 Supported Data Types

EasyAutoML Core intelligently handles various data types automatically:

| Type | Description | Example |
|------|-------------|---------|
| **Numeric** (FLOAT) | Continuous numerical values | `123.45`, `0.95` |
| **Categorical** (LABEL) | Text categories and labels | `"red"`, `"medium"`, `"type_A"` |
| **Temporal** (DATE/TIME) | Date and time values | `"2024-01-15"`, `"14:30:00"` |
| **Text** (LANGUAGE) | Natural language text | `"Great product!"` |
| **Boolean** | True/False values | `True`, `False` |
| **JSON** | Structured nested data | `{"key": "value"}` |

**Automatic Detection**: The system automatically detects data types and applies appropriate encoding, normalization, and feature engineering.

---
 

### Machine Learning Configuration

Key parameters in [`SharedConstants.py`](SharedConstants.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `POSSIBLE_MACHINES_LEVELS` | `[1,2,3,4,5]` | Model complexity levels |
| `NNCONFIGURATION_MAX_POSSIBLE_LAYER_COUNT` | `10` | Max neural network layers |
| `DEBUG_TRAINING_ROWS_COUNT_LIMIT` | `1000` | Training data limit for testing |

---

## 🆘 Support & Community

### Getting Help

- 📖 **Documentation**: Browse the [`doc/`](doc/) directory
- 🐛 **Bug Reports**: [Create an issue](https://github.com/EasyAutoML-com/EasyAutoML-Core/issues)
- 💡 **Feature Requests**: [Open a feature request](https://github.com/EasyAutoML-com/EasyAutoML-Core/issues)
- 🌐 **Website**: [EasyAutoML.com](https://easyautoml.com)

### Commercial Support

For commercial licenses, support, or consulting:
- 📧 Email: [legal@easyautoml.com](mailto:legal@easyautoml.com)
- 🌐 Website: [EasyAutoML.com](https://easyautoml.com)

---

## 📄 License

**EasyAutoML Core** is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**.

### What This Means

✅ **Free to use** for personal, research, academic, and commercial purposes  
✅ **Modify and distribute** freely  
✅ **Use in private/internal applications** without restrictions  
⚠️ **If you offer a public SaaS** using this code, you must open-source your modifications  

See [`LICENSE.md`](LICENSE.md) for full details.

### Dual Licensing Available

Need to use EasyAutoML Core in a **closed-source commercial product**? We offer commercial licensing:
- 📧 Contact: [legal@easyautoml.com](mailto:legal@easyautoml.com)

---

## 🔐 Security

### Reporting Security Vulnerabilities

**Please DO NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via:
- 📧 Email: [security@easyautoml.com](mailto:security@easyautoml.com)
- See our full [Security Policy](SECURITY.md)

We take security seriously and will respond promptly to verified security issues.

### Best Practices

**Environment Variables:**
- ⚠️ **Never commit `.env` files** - Already in `.gitignore`
- Generate secure random keys for `DJANGO_SECRET_KEY`
- Set `DEBUG=False` in production
- Use environment variables for all sensitive data

**Database Security:**
- Use strong passwords for database credentials
- Restrict database access to localhost in development
- Use SSL/TLS for production database connections

**Production Deployment:**
- Keep dependencies updated: `pip install -r requirements.txt --upgrade`
- Monitor security advisories for TensorFlow, Django, and other dependencies
- Use HTTPS for all web interfaces
- Implement proper authentication and authorization

---

## 🌟 Star This Repository

If you find **EasyAutoML Core** useful, please consider giving it a ⭐ on [GitHub](https://github.com/EasyAutoML-com/EasyAutoML-Core)!

---

## 📈 Project Status

- ✅ **Active Development** - Regular updates and improvements
- ✅ **Production Ready** - Battle-tested in real-world applications
- ✅ **Well Documented** - Comprehensive docs in `doc/` directory
- ✅ **Fully Tested** - Extensive test suite included

---

## 🙏 Acknowledgments

**EasyAutoML Core** is built on top of amazing open-source projects:
- [TensorFlow](https://www.tensorflow.org/) - Neural network engine
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Django](https://www.djangoproject.com/) - Web framework & ORM
- [scikit-learn](https://scikit-learn.org/) - Machine learning library
- [pandas](https://pandas.pydata.org/) - Data manipulation
- [sentence-transformers](https://www.sbert.net/) - Text embeddings

---

**Made with ❤️ by the EasyAutoML Team**

[Website](https://easyautoml.com) • [GitHub](https://github.com/EasyAutoML-com) • [Documentation](doc/) • [License](LICENSE.md)

Copyright © 2025 Laurent Bruère / EasyAutoML.com
