# 🚀 CourtLLM Quick Start

Get up and running with CourtLLM in 5 minutes.

## Installation

```bash
# Clone the repository
git clone https://github.com/<username>/courtllm-env.git
cd courtllm-env

# Install dependencies
pip install -r requirements.txt
```

## Run the Demo

```bash
python demo_example.py
```

This will show you:
- How the environment generates cases
- Valid vs invalid testimonies
- Reward computation in action

## Start the Server

```bash
python -m uvicorn server.app:app --reload
```

Visit http://localhost:8000/docs for interactive API documentation.

## Use the Client

```python
from courtllm_env import CourtLLMClient, CourtAction

with CourtLLMClient("http://localhost:8000") as env:
    obs = env.reset()
    print(f"Query: {obs.plaintiff_query}")
    
    action = CourtAction(
        action_type="generate_testimony",
        content="Your factual claim here",
        claim_ids=["claim_000"],
        confidence=0.8,
        source_ids=["BIOMEDICAL_0001"]
    )
    
    result = env.step(action)
    print(f"Reward: {result.reward}")
```

## Train a Model

Open `training/courtllm_grpo_colab.ipynb` in Google Colab and run all cells.

## Next Steps

- Read [README.md](README.md) for full documentation
- Check [DEPLOYMENT.md](DEPLOYMENT.md) for deployment guide
- Explore the code in `server/` directory

## Need Help?

- Check the [Issues](https://github.com/<username>/courtllm-env/issues)
- Read the [OpenEnv docs](https://github.com/meta-pytorch/OpenEnv)
