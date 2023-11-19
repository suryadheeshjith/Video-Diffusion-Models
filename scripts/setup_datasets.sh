
# Run this script in the iris environment

if [ ! -d src ]; then
    echo "Run this script outside scripts/ with ./scripts/setup_roms.sh"
    exit
fi

mkdir -p roms

# Breakout
gdown -O ./roms/roms.zip 1yZWAvHI31HHkTPapHFGjUNg4hgEP3KHU
unzip ./roms/roms.zip -d roms/
rm ./roms/roms.zip

python -m atari_py.import_roms ./roms/Roms