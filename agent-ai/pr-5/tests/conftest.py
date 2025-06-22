import pytest
from prod_agent_system import Agent, AgentConfig
from typing import Generator

@pytest.fixture
def sample_config() -> AgentConfig:
    return AgentConfig(
        name="test_agent",
        description="Test agent",
        tools=[],
        max_retries=3
    )

@pytest.fixture
def agent(sample_config: AgentConfig) -> Generator[Agent, None, None]:
    agent = Agent(sample_config)
    yield agent
    agent.cleanup()