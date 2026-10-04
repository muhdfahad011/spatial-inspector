from typing import List, Dict, Optional
from pydantic import BaseModel

class Measurement(BaseModel):
    value: float
    confidence_interval: float
    unit: str = "meters"

class Opening(BaseModel):
    opening_id: str
    type: str
    wall_id: str
    width: Measurement
    height: Measurement
    distance_along_wall: Measurement

class Wall(BaseModel):
    wall_id: str
    start_point: List[float]
    end_point: List[float]
    length: Measurement

class DamageRegion(BaseModel):
    damage_id: str
    surface_id: str
    damage_class: str
    metric_extent_sq_m: Measurement
    bounding_box_norm: List[float]
    concealed_damage_flag: bool
    rule_fired: Optional[str] = None
    scope_line_item: str

class Room(BaseModel):
    room_id: str
    name: str
    ceiling_height: Measurement
    floor_area_sq_m: Measurement
    walls: List[Wall]
    openings: List[Opening]
    damages: List[DamageRegion]

class WholePropertyPlan(BaseModel):
    tier: str
    capture_id: str
    total_area_sq_m: Measurement
    drift_correction_applied: bool
    rooms: List[Room]
    adjacency_graph: Dict[str, List[str]]
