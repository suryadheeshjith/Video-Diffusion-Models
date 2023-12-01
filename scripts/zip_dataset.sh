
# Run this script in the iris environment

if [ ! -d src ]; then
    echo "Run this script outside scripts/ with ./scripts/setup_roms.sh"
    exit
fi

# Breakout
zip -r ./saved_npy/$1.zip ./saved_npy/$1