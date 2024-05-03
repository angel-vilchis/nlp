from enum import Enum
from dataclasses import dataclass

class DrawObjectType(Enum):
    DRAW_ENTITY = "DrawEntity"
    DRAW_ITEM = "DrawItem"
    DRAW_LINE = "DrawLine"
    DRAW_SPHERE = "DrawSphere"
    DRAW_BLOCK = "DrawBlock"
    DRAW_CUBOID = "DrawCuboid"

@dataclass
class ObjectPosition:
    """Information required to go to the object"""
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    id: str

    def teleport_str(self, diff=(0, 0, 0)) -> str:
        return f"tp {self.x  + diff[0]} {self.y + diff[1]} {self.z + diff[2]}"

@dataclass
class ObjectSpecification:
    """Information required to create the object"""
    draw_object_type: DrawObjectType
    x: float or int
    y: float or int
    z: float or int
    find_with_yaw: float or int
    type: str

    def __str__(self) -> str:
        return f'<{self.draw_object_type.value} x="{self.x}" y="{self.y}" z="{self.z}" type="{self.type}"/>'
    
    def teleport_str(self, diff=(0, 0, 0)) -> str:
        return f"tp {self.x  + diff[0]} {self.y + diff[1]} {self.z + diff[2]}"

DEFAULT_OBJECTS = {
    "chest": ObjectSpecification(DrawObjectType.DRAW_BLOCK, 0, 227, 10, 0, "chest"),
    "red_flower": ObjectSpecification(DrawObjectType.DRAW_BLOCK, 5, 227, 5, 0, "red_flower"),
    "jukebox": ObjectSpecification(DrawObjectType.DRAW_BLOCK, -5, 227, 8, 0, "jukebox"),
    "fence_gate": ObjectSpecification(DrawObjectType.DRAW_BLOCK, -5, 227, -6, 0, "fence_gate"),
    "fence_left": ObjectSpecification(DrawObjectType.DRAW_BLOCK, -4, 227, -6, 0, "fence"),
    "fence_right": ObjectSpecification(DrawObjectType.DRAW_BLOCK, -6, 227, -6, 0, "fence"),
    "wooden_door": ObjectSpecification(DrawObjectType.DRAW_BLOCK, 4, 227, -6, 0, "wooden_door"),
    "netherrack": ObjectSpecification(DrawObjectType.DRAW_BLOCK, -5, 227, 1, 0, "netherrack"),
    "fire": ObjectSpecification(DrawObjectType.DRAW_BLOCK, -5, 228, 1, 0, "fire"),
    "water1": ObjectSpecification(DrawObjectType.DRAW_BLOCK, 10, 226, 0, 0, "water"),
    "water2": ObjectSpecification(DrawObjectType.DRAW_BLOCK, 9, 226, 0, 0, "water"),
    "water3": ObjectSpecification(DrawObjectType.DRAW_BLOCK, 10, 226, 1, 0, "water"),
    "water4": ObjectSpecification(DrawObjectType.DRAW_BLOCK, 9, 226, 1, 0, "water"),
    "Horse": ObjectSpecification(DrawObjectType.DRAW_ENTITY, 10, 227, 9, 20, "Horse"),
    "Witch": ObjectSpecification(DrawObjectType.DRAW_ENTITY, 12, 227, 2, 20, "Witch"),
    "Endermite": ObjectSpecification(DrawObjectType.DRAW_ENTITY, 14, 227, 8, 30, "Endermite"),
    "Ghast": ObjectSpecification(DrawObjectType.DRAW_ENTITY, 14, 227, 8, -55, "Ghast"),
    "Bat": ObjectSpecification(DrawObjectType.DRAW_ENTITY, 14, 227, 8, -60, "Bat"),
    "Guardian": ObjectSpecification(DrawObjectType.DRAW_ENTITY, 20, 227, 9, 10, "Guardian"),
    "Shulker": ObjectSpecification(DrawObjectType.DRAW_ENTITY, 16, 227, 9, 30, "Shulker"),
    "Donkey": ObjectSpecification(DrawObjectType.DRAW_ENTITY, 24, 227, 9, 20, "Donkey"),
    "Mule": ObjectSpecification(DrawObjectType.DRAW_ENTITY, 22, 227, 9, 20, "Mule"),
    "Pig": ObjectSpecification(DrawObjectType.DRAW_ENTITY, 26, 227, 9, 22, "Pig"),
    "Creeper": ObjectSpecification(DrawObjectType.DRAW_ENTITY, 30, 227, 9, 20, "Creeper"),
    "Spider": ObjectSpecification(DrawObjectType.DRAW_ENTITY, 20, 227, 15, 25, "Spider"),
    "Giant": ObjectSpecification(DrawObjectType.DRAW_ENTITY, 19, 227, 20, -20, "Giant"),
    "Cow": ObjectSpecification(DrawObjectType.DRAW_ENTITY, 12, 227, 18, 20, "Cow"),
    "Sheep": ObjectSpecification(DrawObjectType.DRAW_ENTITY, 10, 227, 34, 16, "Sheep"),
    "Chicken": ObjectSpecification(DrawObjectType.DRAW_ENTITY, 34, 227, 9, 25, "Chicken"),
    "Wolf": ObjectSpecification(DrawObjectType.DRAW_ENTITY, 20, 227, 15, 24, "Wolf"),
    "Rabbit": ObjectSpecification(DrawObjectType.DRAW_ENTITY, 16, 227, 20, 30, "Rabbit"),
    "Llama": ObjectSpecification(DrawObjectType.DRAW_ENTITY, 12, 227, 30, 20, "Llama"),
    "Villager": ObjectSpecification(DrawObjectType.DRAW_ENTITY, 14, 227, 34, 15, "Villager")
}