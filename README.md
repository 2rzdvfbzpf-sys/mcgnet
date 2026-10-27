# MCG-Net

## Setup

- Python 3.9
- PyTorch 1.10
- requirements.txt

```bash
pip install -r requirements.txt
```

## Usage

Evaluate algorithm performance:

```bash
python .\test.py --data-dir .\results\test_cache\ --checkpoint .\checkpoints\5way-1shot-best.pt
python .\test.py --data-dir .\results\test_cache\ --checkpoint .\checkpoints\5way-5shot-best.pt
```
