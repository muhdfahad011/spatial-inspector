from typing import List, Dict, Optional
import numpy as np

CONCEALED_DAMAGE_RULES = [
    {
        "id": "RULE_WTR_STUD_BAY",
        "condition": lambda extent, surface: extent > 1.5 and surface == "wall",
        "description": "Moisture migration likely behind drywall into stud cavity; insulation replacement indicated."
    },
    {
        "id": "RULE_BASEBOARD_PERIMETER",
        "condition": lambda extent, surface: surface == "floor" or extent > 2.0,
        "description": "Floor-adjacent moisture; subfloor saturation and sill plate inspection required."
    }
]

def analyze_surface_damage(
    surface_id: str,
    surface_type: str,
    damage_class: str,
    extent_sq_m: float,
    bounding_box: List[float]
) -> Dict:
    """
    Evaluates damage against insurance carrier scoping standards and concealed damage heuristics.
    """
    flagged_rule: Optional[str] = None
    concealed_flag = False

    for rule in CONCEALED_DAMAGE_RULES:
        if rule["condition"](extent_sq_m, surface_type):
            concealed_flag = True
            flagged_rule = rule["id"]
            break

    # Determine standard carrier line item code
    if "water" in damage_class.lower():
        line_item = "WTR-DRYW-TSR" if concealed_flag else "WTR-DRYW-DRY"
    elif "mold" in damage_class.lower():
        line_item = "MED-REMED-BIO"
    elif "fire" in damage_class.lower() or "smoke" in damage_class.lower():
        line_item = "FEE-SMK-DEOD"
    else:
        line_item = "GEN-REPAIR-STD"

    return {
        "damage_id": f"dmg_{surface_id}_01",
        "surface_id": surface_id,
        "damage_class": damage_class,
        "metric_extent_sq_m": {
            "value": round(float(extent_sq_m), 3),
            "confidence_interval": 0.05,
            "unit": "meters"
        },
        "bounding_box_norm": bounding_box,
        "concealed_damage_flag": concealed_flag,
        "rule_fired": flagged_rule,
        "scope_line_item": line_item
    }
