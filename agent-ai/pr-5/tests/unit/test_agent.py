import pytest
from prod_agent_system import AgentError

class TestAgent:
    def test_initialization(self, agent):
        assert agent.name == "test_agent"
        assert agent.max_retries == 3
        assert not agent.is_running

    def test_message_processing(self, agent):
        test_message = "ping"
        response = agent.process(test_message)
        assert response == f"Processed: {test_message}"

    def test_retry_mechanism(self, agent, mocker):
        mocker.patch.object(
            agent, 
            "_process_message", 
            side_effect=[AgentError("Error"), "success"]
        )
        result = agent.process("test")
        assert result == "success"

    def test_cleanup(self, agent):
        agent.start()
        assert agent.is_running
        agent.cleanup()
        assert not agent.is_running

@pytest.mark.slow
def test_high_load_performance(agent):
    # 模拟高负载测试
    for i in range(1000):
        assert agent.process(f"msg_{i}") == f"Processed: msg_{i}"