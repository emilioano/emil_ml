# Mirrors the project into WSL2's native Linux filesystem (~/emil_ml) for
# GPU training. Source of truth stays on Windows; run this before training.
# See README.md "GPU training (WSL2)" for why this mirror exists.

wsl -d Ubuntu -- bash -lc "rsync -a --delete --exclude='.venv*' --exclude='__pycache__' --exclude='*.egg-info' --exclude='emil.db' --exclude='data/*' /mnt/c/Users/emilt/emil_dev/emil_ml/ ~/emil_ml/ && echo Synced to ~/emil_ml"
