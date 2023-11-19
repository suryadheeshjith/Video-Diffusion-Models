if [ ! -d src ]; then
    echo "Run this script outside scripts/ with ./scripts/setup_checkpoints.sh"
    exit
fi

mkdir -p checkpoints

# Breakout
wget -O ./checkpoints/Breakout.pt https://github.com/eloialonso/iris_pretrained_models/raw/main/pretrained_models/Breakout.pt

