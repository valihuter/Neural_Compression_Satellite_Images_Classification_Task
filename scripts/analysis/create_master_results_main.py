#!/usr/bin/env python3
"""
Create comprehensive master results file with all evaluations and BPP data.
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

def load_json(filepath):
    """Load JSON file safely."""
    try:
        with open(filepath) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None

def main():
    master = {
        "metadata": {
            "created": "2025-12-16",
            "description": "Consolidated results for Neural Compression Satellite Images thesis",
            "datasets": ["EuroSAT", "RESISC45"],
            "codecs": ["JPEG", "JPEG2000", "Cheng2020", "MS-ILLM", "JPEG-AI"],
            "models": ["ResNet-18", "ViT-S/16"]
        },
        "bpp_data": {},
        "eurosat": {},
        "resisc45": {}
    }
    
    # Load BPP data
    bpp_data = load_json(RESULTS_DIR / "bpp" / "bpp_measurements_all.json")
    if bpp_data:
        master["bpp_data"] = bpp_data
    
    # ========== EUROSAT ==========
    print("Loading EuroSAT results...")
    
    # ViT evaluation (all codecs)
    vit_all = load_json(RESULTS_DIR / "json" / "eurosat_vit_all_codecs_evaluation.json")
    if vit_all:
        master["eurosat"]["vit_s16"] = {}
        master["eurosat"]["vit_s16"]["baseline"] = vit_all.get("baseline", {})
        for codec in ["jpeg", "jpeg2000", "cheng2020", "msillm", "jpeg_ai"]:
            if codec in vit_all:
                master["eurosat"]["vit_s16"][codec] = vit_all[codec]
    
    # ResNet-18 baseline
    baseline = load_json(RESULTS_DIR / "json" / "eurosat_resnet18_baseline.json")
    if baseline:
        master["eurosat"]["resnet18"] = {"baseline": baseline}
    
    # JPEG-AI EuroSAT
    jpegai_eurosat = load_json(RESULTS_DIR / "json" / "eurosat_resnet18_jpegai_evaluation.json")
    if jpegai_eurosat:
        if "resnet18" not in master["eurosat"]:
            master["eurosat"]["resnet18"] = {}
        master["eurosat"]["resnet18"]["jpegai"] = jpegai_eurosat
    
    # ========== RESISC45 ==========
    print("Loading RESISC45 results...")
    
    # Initialize
    master["resisc45"]["resnet18"] = {}
    master["resisc45"]["vit_s16"] = {}
    
    # JPEG & JPEG2000 RESISC45
    resisc_eval = load_json(RESULTS_DIR / "json" / "resisc45_all_codecs_evaluation.json")
    if resisc_eval:
        if "resnet18" in resisc_eval:
            for codec in resisc_eval["resnet18"]:
                master["resisc45"]["resnet18"][codec] = resisc_eval["resnet18"][codec]
        if "vit_s16" in resisc_eval:
            for codec in resisc_eval["vit_s16"]:
                master["resisc45"]["vit_s16"][codec] = resisc_eval["vit_s16"][codec]
    
    # MS-ILLM & Cheng2020 RESISC45
    msillm_cheng = load_json(RESULTS_DIR / "json" / "resisc45_neural_codecs_evaluation.json")
    if msillm_cheng:
        for codec in ["msillm", "cheng2020"]:
            if codec in msillm_cheng:
                # Restructure to match format
                resnet_data = {}
                vit_data = {}
                for q, data in msillm_cheng[codec].items():
                    if isinstance(data, dict):
                        if "resnet18" in data:
                            resnet_data[q] = data["resnet18"]
                        if "vit_small" in data:
                            vit_data[q] = data["vit_small"]
                master["resisc45"]["resnet18"][codec] = resnet_data
                master["resisc45"]["vit_s16"][codec] = vit_data
    
    # JPEG-AI RESISC45 (complete)
    jpegai_resisc = load_json(RESULTS_DIR / "json" / "resisc45_jpegai_evaluation.json")
    if jpegai_resisc:
        resnet_data = {}
        vit_data = {}
        for q, data in jpegai_resisc.items():
            if isinstance(data, dict):
                if "resnet18" in data:
                    resnet_data[q] = data["resnet18"]
                if "vit_small" in data:
                    vit_data[q] = data["vit_small"]
        master["resisc45"]["resnet18"]["jpegai"] = resnet_data
        master["resisc45"]["vit_s16"]["jpegai"] = vit_data
    
    # Save master results
    output_path = RESULTS_DIR / "json" / "master_results.json"
    with open(output_path, "w") as f:
        json.dump(master, f, indent=2)
    
    print(f"Master results saved to {output_path}")
    
    print("\nMASTER RESULTS SUMMARY")
    
    print("\nEuroSAT:")
    for model in ["resnet18", "vit_s16"]:
        if model in master["eurosat"]:
            codecs = [k for k in master["eurosat"][model].keys() if k != "baseline"]
            print(f"  {model}: {len(codecs)} codecs - {codecs}")
    
    print("\nRESISC45:")
    for model in ["resnet18", "vit_s16"]:
        if model in master["resisc45"]:
            codecs = list(master["resisc45"][model].keys())
            print(f"  {model}: {len(codecs)} codecs - {codecs}")
    
    print("\nJPEG-AI RESISC45 COMPLETE RESULTS")
    if "jpegai" in master["resisc45"]["resnet18"]:
        print(f"{'Quality':<8} {'ResNet-18':>12} {'ViT-S/16':>12}")
        for q in ["q1", "q2", "q3", "q4", "q5", "q6"]:
            resnet_acc = master["resisc45"]["resnet18"]["jpegai"].get(q, {}).get("overall_accuracy", 0)
            vit_acc = master["resisc45"]["vit_s16"]["jpegai"].get(q, {}).get("overall_accuracy", 0)
            print(f"{q:<8} {resnet_acc:>11.2f}% {vit_acc:>11.2f}%")

if __name__ == "__main__":
    main()
