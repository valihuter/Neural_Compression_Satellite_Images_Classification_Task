#!/usr/bin/env python3
"""
Analysiere MS-ILLM Kompletheit auf Google Drive
Erwartung: 11 Klassen × 700 Bilder × 6 Quality = 46200 Bilder
"""

import subprocess
import json

CLASSES = [
    "beach", "circular_farmland", "dense_residential", "forest", "freeway",
    "industrial_area", "lake", "meadow", "medium_residential",
    "rectangular_farmland", "river"
]

QUALITIES = ["q1", "q2", "q3", "q4", "q5", "q6"]

def get_file_count(remote_path):
    """Zähle Dateien in einem Drive-Pfad"""
    try:
        result = subprocess.run(
            ["rclone", "size", remote_path, "--json"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return data["count"]
    except:
        pass
    return 0

print("MS-ILLM KOMPLETHEIT AUF GOOGLE DRIVE\n")

total_expected = len(CLASSES) * 700 * len(QUALITIES)
total_found = 0
missing_details = []

for quality in QUALITIES:
    print(f"\nQuality: {quality}")
    
    quality_total = 0
    
    for class_name in CLASSES:
        remote_path = f"googledrive:MA_Thesis/RESISC45_SUBSET_MS-ILLM/{quality}/{class_name}"
        count = get_file_count(remote_path)
        quality_total += count
        
        status = "" if count == 700 else f" ({count}/700)"
        print(f"{class_name:25} {status}")
        
        if count != 700:
            missing_details.append({
                "quality": quality,
                "class": class_name,
                "found": count,
                "missing": 700 - count
            })
    
    print(f"\n{quality} TOTAL: {quality_total}/7700")
    total_found += quality_total

print("\nZUSAMMENFASSUNG")
print(f"Erwartet: {total_expected} Dateien")
print(f"Gefunden: {total_found} Dateien")
print(f"Fehlen:   {total_expected - total_found} Dateien ({100*(total_expected-total_found)/total_expected:.1f}%)")

if missing_details:
    print("\nFEHLENDE DATEN")
    for detail in missing_details:
        print(f"{detail['quality']}/{detail['class']}: {detail['missing']} Bilder fehlen ({detail['found']}/700)")
