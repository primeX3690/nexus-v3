.PHONY: install test test-v demo-evolution demo-brain demo-swarm demo-all benchmark clean

install:
	pip install -r requirements.txt --break-system-packages

test:
	python3 -m pytest tests/ -q

test-v:
	python3 -m pytest tests/ -v

benchmark:
	python3 -m benchmarks.run_full_benchmark

demo-evolution:
	python3 -m demo.run_evolution

demo-brain:
	python3 -m demo.run_robot_brain

demo-swarm:
	python3 -m demo.run_multi_robot

demo-all: demo-evolution demo-brain demo-swarm

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true