# Contributing to Model Card Generator

Thank you for your interest in contributing to Model Card Generator! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Git
- Docker (optional, for containerized development)

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/modelcard-as-code-generator.git
   cd modelcard-as-code-generator
   ```

2. **Set up development environment**
   ```bash
   make setup-dev
   ```

3. **Verify setup**
   ```bash
   make test
   ```

## 🛠️ Development Workflow

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   make dev-full  # Runs formatting, linting, type-checking, and tests
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

### Commit Message Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes
- `refactor:` - Code refactoring
- `test:` - Test additions or modifications
- `chore:` - Maintenance tasks

## 📋 Contribution Guidelines

### Code Quality

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Write comprehensive docstrings for all public functions and classes
- Maintain test coverage above 80%

### Pull Request Process

1. **Ensure your code passes all checks**
   ```bash
   make quality-check
   make test-coverage
   ```

2. **Update documentation**
   - Update relevant documentation files
   - Add docstrings for new functions/classes
   - Update examples if needed

3. **Create pull request**
   - Use a descriptive title
   - Provide detailed description of changes
   - Link any related issues
   - Include screenshots for UI changes

4. **Address review feedback**
   - Respond to reviewer comments
   - Make requested changes
   - Re-request review when ready

## 🧪 Testing

### Running Tests

```bash
# Run all tests
make test

# Run specific test types
make test-unit
make test-integration

# Run with coverage
make test-coverage
```

### Writing Tests

- Write unit tests for all new functions
- Include integration tests for complex features
- Use meaningful test names that describe behavior
- Follow the Arrange-Act-Assert pattern

## 📚 Documentation

### Updating Documentation

- Update relevant documentation files in `/docs`
- Update docstrings for code changes
- Update examples if API changes

### Building Documentation

```bash
make docs        # Build documentation
make docs-serve  # Serve locally at http://localhost:8000
```

## 🔒 Security

### Reporting Security Issues

Please report security vulnerabilities to security@terragonlabs.com rather than opening public issues.

### Security Guidelines

- Never commit secrets or credentials
- Use environment variables for sensitive configuration
- Follow secure coding practices
- Run security scans before submitting PRs

## 🏷️ Issue Guidelines

### Bug Reports

When reporting bugs, please include:

- Clear description of the issue
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (OS, Python version, etc.)
- Error messages and stack traces

### Feature Requests

For feature requests, please include:

- Clear description of the proposed feature
- Use case and motivation
- Acceptance criteria
- Potential implementation approach (if known)

## 🎯 Areas for Contribution

We welcome contributions in these areas:

### High Priority
- New model card formats and standards
- Integration with additional ML platforms
- Performance optimizations
- Security enhancements

### Medium Priority
- Additional validation rules
- More template options
- Documentation improvements
- Example notebooks

### Low Priority
- UI/UX improvements
- Additional output formats
- Internationalization
- Advanced analytics

## 🌟 Recognition

Contributors are recognized in several ways:

- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- GitHub contributor badges
- Special recognition for significant contributions

## 📞 Getting Help

If you need help:

- Check existing [documentation](https://docs.terragonlabs.com/modelcard-generator)
- Search [existing issues](https://github.com/terragonlabs/modelcard-as-code-generator/issues)
- Join our [Discord community](https://discord.gg/terragonlabs)
- Email support@terragonlabs.com

## 📋 Checklist

Before submitting your PR, ensure:

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] Commit messages follow convention
- [ ] PR description is clear and complete
- [ ] Security implications are considered

Thank you for contributing to Model Card Generator! 🙏