# Development Guide
## Model Card as Code Generator

### Quick Start

1. **Clone and Setup**
   ```bash
   git clone https://github.com/terragonlabs/modelcard-as-code-generator.git
   cd modelcard-as-code-generator
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -e ".[dev,test,docs]"
   ```

2. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   pre-commit install --hook-type commit-msg
   ```

3. **Run Tests**
   ```bash
   pytest
   ```

### Development Environment

#### Requirements
- Python 3.9 or higher
- Git
- Docker (optional, for containerized development)
- Node.js 16+ (for documentation tools)

#### IDE Configuration
We provide configurations for:
- **VS Code**: `.vscode/` directory with recommended extensions and settings
- **PyCharm**: Code style and inspection profiles
- **Vim/Neovim**: EditorConfig support

#### Development Container
For consistent development environments:
```bash
# Using VS Code Dev Containers
code .  # Open in VS Code, then "Reopen in Container"

# Using plain Docker
docker-compose -f docker-compose.dev.yml up -d
docker-compose -f docker-compose.dev.yml exec dev bash
```

### Code Organization

```
src/modelcard_generator/
├── __init__.py              # Package initialization
├── cli.py                   # Command-line interface
├── core/                    # Core business logic
│   ├── generator.py         # Main generator class
│   ├── parser.py           # Data parsing utilities
│   └── validator.py        # Validation logic
├── templates/              # Jinja2 templates
│   ├── huggingface/
│   ├── google/
│   └── eu_cra/
├── schemas/                # JSON schemas for validation
├── integrations/          # External service integrations
│   ├── mlflow.py
│   ├── wandb.py
│   └── huggingface_hub.py
├── monitoring/            # Observability components
└── security/              # Security utilities
```

### Development Workflow

#### 1. Feature Development
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes, run tests frequently
pytest tests/unit/
pytest tests/integration/

# Check code quality
make lint
make format
make type-check
make security-check

# Commit with conventional commit format
git commit -m "feat: add new template validation"
```

#### 2. Code Quality Standards

**Formatting**: We use `black` with line length 88
```bash
black src/ tests/
```

**Linting**: `ruff` for fast Python linting
```bash
ruff check src/ tests/
```

**Type Checking**: `mypy` for static type analysis
```bash
mypy src/
```

**Import Sorting**: `isort` compatible with black
```bash
isort src/ tests/
```

**Security**: `bandit` for security issue detection
```bash
bandit -r src/
```

#### 3. Testing Strategy

**Unit Tests**: Fast, isolated tests for individual components
```bash
pytest tests/unit/ -v
```

**Integration Tests**: Test component interactions
```bash
pytest tests/integration/ -v
```

**Property-Based Testing**: Using `hypothesis` for edge cases
```bash
pytest tests/property/ -v
```

**Performance Tests**: Benchmark critical paths
```bash
pytest tests/performance/ -v --benchmark-only
```

**Coverage Requirements**: Minimum 80% coverage
```bash
pytest --cov=modelcard_generator --cov-report=html --cov-fail-under=80
```

#### 4. Documentation

**API Documentation**: Auto-generated from docstrings
```bash
mkdocs serve  # Live preview
mkdocs build  # Build static site
```

**Docstring Format**: Google style docstrings
```python
def generate_card(data: Dict[str, Any]) -> str:
    """Generate a model card from input data.
    
    Args:
        data: Dictionary containing model metadata and evaluation results.
        
    Returns:
        Generated model card as markdown string.
        
    Raises:
        ValidationError: If input data fails schema validation.
        TemplateError: If template rendering fails.
    """
```

### Testing Guidelines

#### Test Structure
```python
# tests/unit/test_generator.py
import pytest
from modelcard_generator.core.generator import ModelCardGenerator

class TestModelCardGenerator:
    """Test suite for ModelCardGenerator class."""
    
    def test_generate_basic_card(self, sample_data):
        """Test basic model card generation."""
        generator = ModelCardGenerator()
        result = generator.generate(sample_data)
        assert result is not None
        assert "Model Card" in result
    
    def test_generate_with_invalid_data(self):
        """Test error handling for invalid input."""
        generator = ModelCardGenerator()
        with pytest.raises(ValidationError):
            generator.generate({})
```

#### Test Categories
- **Unit Tests**: Test single functions/methods in isolation
- **Integration Tests**: Test component interactions
- **Contract Tests**: Test API contracts and interfaces
- **Property Tests**: Test invariants with random data
- **Performance Tests**: Benchmark and regression tests

#### Test Data Management
- Use `tests/fixtures/` for test data files
- Create factories with `factory-boy` for complex objects
- Mock external services consistently

### Performance Guidelines

#### Profiling
```bash
# Profile with cProfile
python -m cProfile -o profile.stats -m modelcard_generator.cli generate sample.json

# Analyze with snakeviz
snakeviz profile.stats

# Memory profiling with memory-profiler
python -m memory_profiler script.py
```

#### Optimization Targets
- Model card generation: < 30 seconds for standard inputs
- Large file processing: < 2 minutes for 100MB+ files
- Memory usage: < 2GB peak for large models
- Startup time: < 5 seconds for CLI

### Security Considerations

#### Secure Coding Practices
- Never log sensitive data (API keys, model weights)
- Validate all external inputs
- Use secure defaults for file permissions
- Sanitize template inputs to prevent injection

#### Security Tools
```bash
# Check for known vulnerabilities
safety check --json

# Security linting
bandit -r src/

# Dependency scanning
pip-audit

# License compliance
pip-licenses --format=json
```

### Release Process

#### Version Management
We use semantic versioning (semver):
- **Major**: Breaking changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible

#### Release Checklist
1. Update CHANGELOG.md
2. Bump version in pyproject.toml
3. Tag release: `git tag v1.2.3`
4. Push tags: `git push --tags`
5. Build and publish: `make release`

#### Automated Releases
- Releases are triggered by version tags
- Changelog is auto-generated from conventional commits
- Packages are published to PyPI automatically
- Docker images are built and pushed to registry

### Contributing Guidelines

#### Pull Request Process
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Ensure all checks pass
5. Submit pull request with clear description

#### Commit Message Format
We use conventional commits:
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

#### Code Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests cover new functionality
- [ ] Documentation is updated
- [ ] Performance impact considered
- [ ] Security implications reviewed
- [ ] Breaking changes documented

### Troubleshooting

#### Common Issues

**Import Errors**
```bash
# Ensure package is installed in development mode
pip install -e .
```

**Test Failures**
```bash
# Run specific test for debugging
pytest tests/unit/test_generator.py::TestGenerator::test_basic -v -s

# Run with debugger
pytest tests/unit/test_generator.py::TestGenerator::test_basic --pdb
```

**Performance Issues**
```bash
# Profile the problematic code path
python -m cProfile -o profile.stats your_script.py
snakeviz profile.stats
```

#### Getting Help
- Check existing GitHub issues
- Join our Discord community
- Read the FAQ in documentation
- Contact maintainers for urgent issues

### Development Tools

#### Makefile Commands
```bash
make help           # Show available commands
make install        # Install development dependencies
make test           # Run all tests
make lint          # Run linting
make format        # Format code
make docs          # Build documentation
make clean         # Clean build artifacts
make release       # Build and publish release
```

#### Pre-commit Hooks
Automatically run on commit:
- `black` for code formatting
- `isort` for import sorting
- `ruff` for linting
- `mypy` for type checking
- `bandit` for security issues
- `pytest` for fast unit tests

#### CI/CD Pipeline
Our pipeline includes:
- Code quality checks
- Security scanning
- Multi-Python version testing
- Documentation building
- Performance benchmarking
- Dependency vulnerability scanning