# Contributing to EasyAutoML

🎉 Welcome! Thank you for your interest in contributing to EasyAutoML — a no-code AutoML platform that makes machine learning accessible to everyone. Whether you're a developer, data scientist, educator, or first-time contributor, your help is appreciated and essential to the community.

This guide outlines how to contribute effectively.

---

## 🚀 Our Mission

EasyAutoML aims to **democratize access to machine learning** through:
- No-code model building from Excel/CSV files
- CNN + Genetic Algorithm-driven optimization
- Text support via Large Language Models (LLMs)
- Monetization via model, dataset, and consulting marketplaces

We're making advanced ML **open source and open access.**

---

## 📌 How You Can Contribute

### 1. **Code Contributions**
- Bug fixes, refactoring, or optimization
- New model types or algorithm modules (e.g., tree-based models, transformers)
- Improvements to the Python SDK or REST API
- Frontend features (e.g., new dashboards or visualization tools)

### 2. **Documentation**
- Improve clarity in usage docs and tutorials
- Translate documentation into other languages
- Write new guides, case studies, or example notebooks

### 3. **Community Engagement**
- Answer questions in GitHub Discussions or forums
- Help new users troubleshoot
- Share EasyAutoML projects or integrations

### 4. **Marketplace Contributions**
- Share open datasets or models
- Offer your consulting services
- Review or test marketplace features

---

## 🛠️ Setup Instructions

To contribute code, you'll need to set up EasyAutoML Core locally:

### 1. Clone and Setup

```bash
# Clone the repo
git clone https://github.com/EasyAutoML-com/EasyAutoML-Core.git
cd EasyAutoML-Core

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows PowerShell:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up Environment

Create a `.env` file in the project root:

```bash
# Django settings
DJANGO_SECRET_KEY=your-secret-key-here
DEBUG=True

# Database (defaults to SQLite)
USE_SQLITE=True
```

### 3. Verify Installation

```bash
# Run tests to verify everything works
python -m pytest "tests/Tests All AI modules/" -v

# Quick import test
python -c "from ML.MachineEasyAutoML import MachineEasyAutoML; print('✅ Setup successful!')"
```

### 4. Development Workflow

- **Code Location**: Main ML code is in `ML/` directory
- **Models**: Django models are in `models/` directory
- **Tests**: Add tests in `tests/Tests All AI modules/unit/`
- **Documentation**: Update docs in `doc/` directory

---

## 📝 Contribution Guidelines

### Code Style

- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and single-purpose
- Comment complex logic

### Testing Requirements

All code contributions must include tests:

```bash
# Create tests in tests/Tests All AI modules/unit/
# Test file naming: test_<module_name>.py
# Test function naming: test_<functionality>

# Run your tests before submitting
python -m pytest "tests/Tests All AI modules/unit/test_your_module.py" -v

# Run all tests to ensure nothing breaks
python -m pytest "tests/Tests All AI modules/" -v
```

### Documentation

- Update relevant documentation in `doc/` directory
- Add docstrings with parameter descriptions
- Include usage examples for new features
- Update README.md if adding major features

### Pull Request Process

1. **Create Feature Branch**: `git checkout -b feature/your-feature-name`
2. **Make Changes**: Write code, add tests, update docs
3. **Test Thoroughly**: Ensure all tests pass
4. **Commit**: Use clear commit messages
   - `Add: new feature description`
   - `Fix: bug description`
   - `Update: documentation for X`
   - `Refactor: improve Y performance`
5. **Push**: `git push origin feature/your-feature-name`
6. **Create PR**: Submit pull request with clear description

### What We Look For

✅ **Clear Purpose**: PR solves one problem or adds one feature  
✅ **Tests Included**: New code has corresponding tests  
✅ **Documentation**: Changes are documented  
✅ **Code Quality**: Follows project style and best practices  
✅ **No Breaking Changes**: Existing functionality still works  

---

## 🐛 Reporting Bugs

Found a bug? Help us fix it!

### Before Reporting

1. Check if the issue already exists in [GitHub Issues](https://github.com/EasyAutoML-com/EasyAutoML-Core/issues)
2. Try to reproduce with the latest version
3. Gather relevant information (Python version, error messages, etc.)

### Creating a Bug Report

Include:
- **Description**: Clear description of the bug
- **Steps to Reproduce**: Minimal code to reproduce the issue
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**: Python version, OS, dependency versions
- **Error Messages**: Full stack traces if applicable

---

## 💡 Feature Requests

Have an idea? We'd love to hear it!

### Suggest New Features

1. Check existing [feature requests](https://github.com/EasyAutoML-com/EasyAutoML-Core/issues?q=is%3Aissue+label%3Aenhancement)
2. Open a new issue with the `enhancement` label
3. Describe:
   - **Problem**: What problem does this solve?
   - **Solution**: Your proposed solution
   - **Use Cases**: Real-world scenarios
   - **Alternatives**: Other solutions you considered

---

## 🤝 Code of Conduct

### Our Standards

- **Be Respectful**: Treat everyone with respect
- **Be Collaborative**: Work together constructively
- **Be Patient**: Help newcomers learn
- **Be Professional**: Keep discussions focused and productive

### Our Responsibilities

Maintainers will:
- Review pull requests promptly
- Provide constructive feedback
- Maintain a welcoming environment
- Enforce the code of conduct fairly

---

## 📧 Contact

- **General Questions**: [GitHub Discussions](https://github.com/EasyAutoML-com/EasyAutoML-Core/discussions)
- **Bug Reports**: [GitHub Issues](https://github.com/EasyAutoML-com/EasyAutoML-Core/issues)
- **Security Issues**: security@easyautoml.com (see [SECURITY.md](SECURITY.md))
- **Commercial Support**: legal@easyautoml.com

---

## 🙏 Thank You!

Thank you for contributing to EasyAutoML! Every contribution, no matter how small, helps make machine learning more accessible to everyone.

---

**Happy Coding! 🚀**