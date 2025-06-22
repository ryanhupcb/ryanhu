# Agent AI System

A production-ready AI agent system with extensible architecture.

## Features

- Modular agent design
- Configurable retry mechanism
- Comprehensive test coverage
- REST API support
- Detailed documentation

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the system:
```bash
python -m prod_agent_system
```

3. Run tests:
```bash
pip install -r requirements-test.txt
pytest tests/
```

## API Documentation

View interactive API docs: [OpenAPI Spec](./api-docs/agent-api.yaml)

## Testing

Test types:
- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`
- Performance tests: `tests/performance/`

Generate coverage report:
```bash
pytest --cov=./ --cov-report=html
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Add tests for your changes
4. Submit a pull request

## License

MIT