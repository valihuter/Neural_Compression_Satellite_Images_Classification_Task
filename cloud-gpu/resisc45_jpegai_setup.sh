#!/bin/bash
# JPEG-AI Compression for RESISC45 on RunPod
# 
# This script sets up and runs JPEG-AI compression in a parallelized manner
# 
# USAGE:
#   1. Start a RunPod instance with:
#      - GPU: Any (GPU used for NN inference, not entropy coding)
#      - Template: PyTorch 2.0 / CUDA 12.1
#      - Disk: 50GB minimum
#   
#   2. Upload RESISC45 dataset to /workspace/data/NWPU-RESISC45/
#   
#   3. Run this script:
#      bash setup_jpegai_resisc45.sh
#
# PARALLELIZATION:
#   For faster processing, run two terminals:
#   Terminal 1: python3 compress_resisc45.py --quality q1,q2,q3
#   Terminal 2: python3 compress_resisc45.py --quality q4,q5,q6

set -e

echo "=================================================="
echo "JPEG-AI Setup for RESISC45 on RunPod"
echo "=================================================="

# Configuration
WORKSPACE="/workspace"
JPEGAI_DIR="$WORKSPACE/jpeg-ai-reference-software"
DATA_DIR="$WORKSPACE/data"
OUTPUT_DIR="$WORKSPACE/output/resisc45_jpegai"

# Create directories
mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR"

# Step 1: Clone JPEG-AI if not present
if [ ! -d "$JPEGAI_DIR" ]; then
    echo ""
    echo "[Step 1/5] Cloning JPEG-AI Reference Software..."
    cd "$WORKSPACE"
    git lfs install
    GIT_LFS_SKIP_SMUDGE=1 git clone https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software.git
    cd "$JPEGAI_DIR"
    git lfs pull
else
    echo "[Step 1/5] JPEG-AI already cloned"
fi

# Step 2: Install Python dependencies
echo ""
echo "[Step 2/5] Installing Python dependencies..."
pip install --quiet \
    pybind11 \
    einops \
    opencv-python \
    pillow \
    numpy \
    scipy \
    scikit-image \
    pandas \
    tqdm \
    matplotlib \
    pyyaml \
    commentjson \
    nvidia-ml-py \
    attrs \
    pytorch-msssim \
    addict \
    prettytable \
    ptflops

# Step 3: Build C++ extensions
echo ""
echo "[Step 3/5] Building C++ entropy coding extensions..."
cd "$JPEGAI_DIR/src/codec/entropy_coding/cpp_exts/mans"
make clean 2>/dev/null || true
make

cd "$JPEGAI_DIR/src/codec/entropy_coding/cpp_exts/direct"
make clean 2>/dev/null || true
make

# Copy .so files
echo "Copying compiled libraries..."
cp "$JPEGAI_DIR/src/codec/entropy_coding/cpp_exts/mans/"*.so \
   "$JPEGAI_DIR/src/codec/entropy_coding/lib_wrappers/mans/" 2>/dev/null || true
cp "$JPEGAI_DIR/src/codec/entropy_coding/cpp_exts/direct/"*.so \
   "$JPEGAI_DIR/src/codec/entropy_coding/lib_wrappers/direct/" 2>/dev/null || true

# Step 4: Verify RESISC45 dataset
echo ""
echo "[Step 4/5] Checking RESISC45 dataset..."

RESISC_DIR="$DATA_DIR/NWPU-RESISC45"

if [ -d "$RESISC_DIR" ]; then
    NUM_CLASSES=$(find "$RESISC_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
    NUM_IMAGES=$(find "$RESISC_DIR" -name "*.jpg" | wc -l)
    echo "✓ Found RESISC45: $NUM_CLASSES classes, $NUM_IMAGES images"
else
    echo "✗ RESISC45 not found at: $RESISC_DIR"
    echo ""
    echo "Please upload the dataset. Options:"
    echo ""
    echo "1. Using runpodctl (from your local machine):"
    echo "   runpodctl send /path/to/NWPU-RESISC45"
    echo ""
    echo "2. Direct download in RunPod (if you have a direct link):"
    echo "   wget <url> -O $DATA_DIR/NWPU-RESISC45.zip"
    echo "   unzip $DATA_DIR/NWPU-RESISC45.zip -d $DATA_DIR/"
    echo ""
    echo "3. Using rclone (if configured):"
    echo "   rclone copy gdrive:NWPU-RESISC45 $RESISC_DIR"
    echo ""
    exit 1
fi

# Step 5: Create compression script symlink
echo ""
echo "[Step 5/5] Setting up compression script..."

# Create a wrapper script that can be run from anywhere
cat > "$WORKSPACE/run_compression.sh" << 'SCRIPT'
#!/bin/bash
# Wrapper script for JPEG-AI compression

QUALITY=${1:-"all"}
LIMIT=${2:-0}

cd /workspace/jpeg-ai-reference-software

export PYTHONPATH="/workspace/jpeg-ai-reference-software:$PYTHONPATH"

python3 << PYEOF
import sys
sys.path.insert(0, '/workspace/jpeg-ai-reference-software')

import json
import subprocess
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import time
from datetime import datetime
import signal
import shutil
import tempfile

# Paths
INPUT_ROOT = Path('/workspace/data/NWPU-RESISC45')
OUTPUT_ROOT = Path('/workspace/output/resisc45_jpegai')
JPEG_AI_ROOT = Path('/workspace/jpeg-ai-reference-software')
CHECKPOINT_FILE = OUTPUT_ROOT / 'checkpoint.json'

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Quality levels
QUALITY_LEVELS = {
    'q1': 13,   # 0.13 BPP
    'q2': 25,   # 0.25 BPP
    'q3': 50,   # 0.50 BPP
    'q4': 75,   # 0.75 BPP
    'q5': 100,  # 1.00 BPP
    'q6': 150,  # 1.50 BPP
}

checkpoint_data = None

def save_checkpoint(data):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def signal_handler(sig, frame):
    global checkpoint_data
    print("\n\nInterrupted! Saving checkpoint...")
    if checkpoint_data:
        save_checkpoint(checkpoint_data)
    sys.exit(0)

def is_already_compressed(img_path, quality_name):
    output_path = OUTPUT_ROOT / f'jpeg_ai_{quality_name}' / img_path.parent.name / f'{img_path.stem}.png'
    return output_path.exists()

def compress_image(input_path, output_dir, target_bpp_x100):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(input_path)
    final_output = output_dir / f'{input_path.stem}.png'
    
    temp_dir = Path(tempfile.mkdtemp())
    temp_input = temp_dir / "input.png"
    temp_output = temp_dir / "output.png"
    bitstream = temp_dir / "compressed.bin"
    
    try:
        img = Image.open(input_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if img.size != (256, 256):
            img = img.resize((256, 256), Image.BICUBIC)
        img.save(temp_input)
        
        # Encode
        result = subprocess.run(
            ['python3', '-m', 'src.reco.coders.encoder',
             str(temp_input), str(bitstream),
             '--set_target_bpp', str(target_bpp_x100),
             '--cfg', 'cfg/tools_on.json', 'cfg/profiles/base.json'],
            cwd=JPEG_AI_ROOT,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            return False, f"Encode failed"
        
        # Decode
        result = subprocess.run(
            ['python3', '-m', 'src.reco.coders.decoder',
             str(bitstream), str(temp_output)],
            cwd=JPEG_AI_ROOT,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            return False, f"Decode failed"
        
        shutil.copy2(temp_output, final_output)
        bpp = (bitstream.stat().st_size * 8) / (256 * 256)
        return True, bpp
        
    except Exception as e:
        return False, str(e)[:100]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    global checkpoint_data
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    quality_arg = "${QUALITY}"
    limit_arg = int("${LIMIT}")
    
    print("=" * 80)
    print("JPEG-AI Compression for RESISC45")
    print("=" * 80)
    
    # Determine quality levels
    if quality_arg == 'all':
        quality_items = list(QUALITY_LEVELS.items())
    else:
        requested = [q.strip() for q in quality_arg.split(',')]
        quality_items = [(q, QUALITY_LEVELS[q]) for q in requested if q in QUALITY_LEVELS]
    
    print(f"Quality levels: {[q[0] for q in quality_items]}")
    
    # Find images
    test_images = []
    for class_dir in sorted(INPUT_ROOT.iterdir()):
        if class_dir.is_dir():
            for img in sorted(class_dir.glob("*.jpg")):
                test_images.append(img.relative_to(INPUT_ROOT))
    
    if limit_arg > 0:
        test_images = test_images[:limit_arg]
        print(f"[Limited to {limit_arg} images]")
    
    print(f"Found {len(test_images)} images")
    
    # Process each quality level
    for qname, target_bpp in quality_items:
        print(f"\n{'='*60}")
        print(f"{qname}: Target {target_bpp/100:.2f} BPP")
        print(f"{'='*60}")
        
        success_count = 0
        skipped_count = 0
        total_bpp = 0
        start = time.time()
        
        for i, img_path in enumerate(tqdm(test_images, desc=qname)):
            checkpoint_data = {'quality': qname, 'idx': i}
            
            if is_already_compressed(img_path, qname):
                skipped_count += 1
                success_count += 1
                continue
            
            input_full = INPUT_ROOT / img_path
            output_dir = OUTPUT_ROOT / f'jpeg_ai_{qname}' / img_path.parent.name
            
            ok, result = compress_image(input_full, output_dir, target_bpp)
            if ok:
                success_count += 1
                total_bpp += result
            
            if (i+1) % 100 == 0:
                save_checkpoint(checkpoint_data)
        
        elapsed = time.time() - start
        new_proc = success_count - skipped_count
        avg_bpp = total_bpp / new_proc if new_proc > 0 else 0
        
        print(f"\n{success_count}/{len(test_images)} done ({skipped_count} skipped)")
        print(f"Avg BPP: {avg_bpp:.3f}, Time: {elapsed/60:.1f} min")
    
    print("\n✓ COMPRESSION COMPLETE")

if __name__ == '__main__':
    main()
PYEOF
SCRIPT

chmod +x "$WORKSPACE/run_compression.sh"


echo ""
echo "SETUP COMPLETE!"
echo ""
echo "To run compression:"
echo ""
echo "  Full (all quality levels):"
echo "    ./run_compression.sh all"
echo ""
echo "  Parallel (2 terminals for ~2x speed):"
echo "    Terminal 1: ./run_compression.sh q1,q2,q3"
echo "    Terminal 2: ./run_compression.sh q4,q5,q6"
echo ""
echo "  Test with limited images:"
echo "    ./run_compression.sh all 100"
echo ""
echo "Output will be saved to:"
echo "  $OUTPUT_DIR/"
echo ""
echo "Estimated time (31.5k images × 6 quality levels):"
echo "  - Sequential: ~20-25 hours"
echo "  - Parallel (2 jobs): ~10-12 hours"
echo ""
