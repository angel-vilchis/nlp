import re
import time
import json
import warnings
import numpy as np
from object_information import ObjectPosition, DEFAULT_OBJECTS, DrawObjectType
import transformers
from setfit import SetFitModel

nlp_model = SetFitModel.from_pretrained("malmoTextClassifier")
text_gen = transformers.pipeline("text-generation", model="gpt2")

chest_open = False

LABEL_INFO = {
    0:  {"desc": "Open chest",            "response": "Ok, I'm opening the chest."}, 
    1:  {"desc": "Smell plant",           "response": "Heading over to smell the plant!"},
    2:  {"desc": "Go to mob",             "response": None},
    3:  {"desc": "Jump in water",         "response": "I'm a bad swimmer, but I'll do it!"},
    4:  {"desc": "Sit next to campfire",  "response": "Time to get cozy..."},
    5:  {"desc": "Play music",            "response": "I love music!"},
    6:  {"desc": "Go through fence",      "response": "Let's go see what is on the other side of that fence."},
    7:  {"desc": "Go inside door",        "response": "This is me, make yourself at home."},
    8:  {"desc": "Talk to user",          "response": "I'd love to talk to you!"},
    9:  {"desc": "Move forward",          "response": "Ok, moving forward now. Let me know when to stop."},
    10: {"desc": "Move backward",         "response": "Ok, moving backward now. Let me know when to stop."},
    11: {"desc": "Strafe left",           "response": "Ok, strafing left now. Let me know when to stop."},
    12: {"desc": "Strafe right",          "response": "Ok, strafing right now. Let me know when to stop."},
    13: {"desc": "Pitch upwards",         "response": "Ok, pitching upwards now. Let me know when to stop."},
    14: {"desc": "Pitch downwards",       "response": "Ok, pitching downwards now. Let me know when to stop."},
    15: {"desc": "Turn left",             "response": "Ok, turning left now. Let me know when to stop."},
    16: {"desc": "Turn right",            "response": "Ok, turning right now. Let me know when to stop."},
    17: {"desc": "Start jumping",         "response": "Ok, starting to jump now. Let me know when to stop."},
    18: {"desc": "Stop movement",         "response": "Stopping movement."},
    19: {"desc": "Start crouching",       "response": "I'll crouch, but don't leave me like this for too long!"},
    20: {"desc": "Stop crouching",        "response": "I'll stand straight up."},
    21: {"desc": "Start attacking",       "response": "I'm a lover, not a figher, but I'll attack this time."},
    22: {"desc": "Stop attacking",        "response": "Great, I'm tired of attacking, anyways."},
    23: {"desc": "Use this",              "response": "What does this do?"},
    24: {"desc": "Stop using this",       "response": "I'm done using this."},
}

def get_latest_world_observations(agent_host):
    world_state = agent_host.peekWorldState()
    latest_observation = json.loads(world_state.observations[-1].text)
    return latest_observation

def flush_world_observations(agent_host):
    agent_host.getWorldState()

def face_entity(agent_host, name: str, max_rotations=150):
    """Attempts to face entity. Returns True if successful, False otherwise."""
    rotations = 0
    while rotations <= max_rotations:
        latest_observation = get_latest_world_observations(agent_host)
        if 'LineOfSight' in latest_observation:
            if DEBUG: print(f"Line of sight observation:\n{latest_observation['LineOfSight']}\n")
            if latest_observation['LineOfSight']['type'] == name:
                agent_host.sendCommand("turn 0")
                break
        else:
            if DEBUG: print("Did not find line of sight")
        agent_host.sendCommand("turn 0.4")
        time.sleep(0.1)
        if DEBUG: print(f"Rotation {rotations}")
        rotations += 1
    else:
        agent_host.sendCommand("turn 0")
        return False
    return True

def find_entity(agent_host, name: str, max_retries=5):
    position = None
    retries = 0
    while position is None and retries <= max_retries:
        retries += 1
        latest_observation = get_latest_world_observations(agent_host)
        if DEBUG: print(f"Latest observation:\n{latest_observation}\n")
        if 'NearbyEntities' in latest_observation:
            for nearby_entity in latest_observation['NearbyEntities']:
                if DEBUG: print(f"Nearby Entity:\n{nearby_entity}\n")
                if nearby_entity['name'] == name:
                    if DEBUG: print(f"Matched {name} entity: {nearby_entity}\n")
                    position = ObjectPosition(
                        x=nearby_entity['x'], 
                        y=nearby_entity['y'], 
                        z=nearby_entity['z'], 
                        yaw=nearby_entity['yaw'], 
                        pitch=nearby_entity['pitch'], 
                        id=nearby_entity['id']
                    )
    return position

def get_prediction(input_text):
    probs = nlp_model.predict_proba([input_text])[0].tolist()
    print("Probability prediction:\n", dict(zip([v["desc"] for v in LABEL_INFO.values()], np.round(probs, 2))))
    return np.argmax(probs)

def task_0(agent_host):
    """
    Complete task of opening chest
    """
    global chest_open
    # Teleport
    reset_agent(agent_host)
    agent_host.sendCommand(DEFAULT_OBJECTS["chest"].teleport_str(diff=(0.5, 0, -0.5)))

    # Look down
    time.sleep(2.10)
    agent_host.sendCommand("pitch 0.5")
    time.sleep(0.5)
    agent_host.sendCommand("pitch 0")

    # Open chest
    agent_host.sendCommand("use 1")
    chest_open = True

def task_1(agent_host):
    """
    Complete task of smelling flower
    """
    # Teleport
    reset_agent(agent_host)
    agent_host.sendCommand(DEFAULT_OBJECTS["red_flower"].teleport_str())

    # Turn to face flower
    agent_host.sendCommand("turn -1")
    time.sleep(0.4)
    agent_host.sendCommand("turn 0")

    # Look down
    time.sleep(1.0)
    agent_host.sendCommand("pitch 0.5")
    time.sleep(0.9)
    agent_host.sendCommand("pitch 0")

def task_2(agent_host, input_text: str):
    """Go to entity"""
    request_words = re.sub(r"[^A-Za-z ]+", "", input_text).strip().title().split()
    environment_entities = {k for k, v in DEFAULT_OBJECTS.items() if v.draw_object_type == DrawObjectType.DRAW_ENTITY}
    request_entities = set(request_words) & environment_entities
    entity = None if len(request_entities) == 0 else next(iter(request_entities))
    if entity is None:
        agent_host.sendCommand("chat I could not find that entity in this environment.")
        print(f"Try one of these entities: {environment_entities}")
        return
    agent_host.sendCommand(f"chat Ok, I'm looking for the {entity} now.")
    entity_specification = DEFAULT_OBJECTS[entity]
    position = find_entity(agent_host, entity)
    if position is not None:
        if DEBUG: print(position.teleport_str())
        agent_host.sendCommand(f"setPitch {entity_specification.find_with_yaw}")
        agent_host.sendCommand(position.teleport_str(diff=(2, 0, 2)))
        facing_entity = face_entity(agent_host, entity)
        if facing_entity:
            agent_host.sendCommand(f"chat Found the {entity}!")
    else:
        if DEBUG: print(f"Could not find a {entity}.")
        agent_host.sendCommand(f"chat I could not find the {entity}. It may not be nearby.")
    flush_world_observations(agent_host)

def task_3(agent_host):
    """Get in water"""
    reset_agent(agent_host)
    agent_host.sendCommand(DEFAULT_OBJECTS["water1"].teleport_str(diff=(0.5, 0, -0.5)))

def task_4(agent_host):
    """Sit next to campfire"""
    reset_agent(agent_host)
    agent_host.sendCommand(DEFAULT_OBJECTS["netherrack"].teleport_str(diff=(0.5, 0, -0.5)))

def task_5(agent_host):
    """Hit jukebox"""
    reset_agent(agent_host)
    agent_host.sendCommand(DEFAULT_OBJECTS["jukebox"].teleport_str(diff=(0.5, 0, -0.5)))

    time.sleep(2.10)
    agent_host.sendCommand("pitch 0.5")
    time.sleep(0.5)
    agent_host.sendCommand("pitch 0")

def task_6(agent_host):
    """Go through gate"""
    reset_agent(agent_host)
    agent_host.sendCommand(DEFAULT_OBJECTS["fence_gate"].teleport_str(diff=(0.5, 0, -0.5)))
    go_through_entrance(agent_host)


def task_7(agent_host):
    """Go through door"""
    reset_agent(agent_host)
    agent_host.sendCommand(DEFAULT_OBJECTS["wooden_door"].teleport_str(diff=(0.5, 0, -0.5)))
    go_through_entrance(agent_host)
    time.sleep(0.1)
    agent_host.sendCommand("turn -0.5")
    time.sleep(0.25)
    agent_host.sendCommand("turn 0")

    time.sleep(0.1)
    agent_host.sendCommand("use 1"); agent_host.sendCommand("use 0")

def task_8(agent_host):
    """Chat with agent"""
    reset_agent(agent_host)
    user_prompt = "User: "
    agent_prompt = "\nMy long response: "

    chat_input = input("Say something to bot or 'quit'/'q' to stop chatting: ")
    while chat_input.lower() != "quit" and chat_input.lower() != "q":
        model_input = user_prompt + chat_input + agent_prompt

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_output = text_gen(model_input, max_new_tokens=30, pad_token_id=50256, num_return_sequences=1)[0]["generated_text"]
        chat_text = model_output.split(":")[-1].strip()
        agent_host.sendCommand(f"chat {chat_text}")
        chat_input = input("Say something to bot or 'quit'/'q' to stop chatting: ")

def task_9(agent_host):
    """Move forward"""
    agent_host.sendCommand("move 1")

def task_10(agent_host):
    """Move backward"""
    agent_host.sendCommand("move -1")

def task_11(agent_host):
    """Strafe left"""
    agent_host.sendCommand("strafe -1")

def task_12(agent_host):
    """Strafe right"""
    agent_host.sendCommand("strafe 1")

def task_13(agent_host):
    """Pitch upwards"""
    agent_host.sendCommand("pitch -0.1")

def task_14(agent_host):
    """Pitch downwards"""
    agent_host.sendCommand("pitch 0.1")

def task_15(agent_host):
    """Turn left"""
    agent_host.sendCommand("turn -0.2")

def task_16(agent_host):
    """Turn right"""
    agent_host.sendCommand("turn 0.2")

def task_17(agent_host):
    """Start jumping"""
    agent_host.sendCommand("jump 1")

def task_18(agent_host):
    """Stop movement"""
    agent_host.sendCommand("move 0")
    agent_host.sendCommand("strafe 0")
    agent_host.sendCommand("pitch 0")
    agent_host.sendCommand("turn 0")
    agent_host.sendCommand("jump 0")
    agent_host.sendCommand("attack 0")

def task_19(agent_host):
    """Start crouching"""
    agent_host.sendCommand("crouch 1")

def task_20(agent_host):
    """Stop crouching"""
    agent_host.sendCommand("crouch 0")

def task_21(agent_host):
    """Start attacking"""
    agent_host.sendCommand("attack 1")

def task_22(agent_host):
    """Stop attacking"""
    agent_host.sendCommand("attack 0")
    
def task_23(agent_host):
    """Use this"""
    agent_host.sendCommand("use 1")
   
def task_24(agent_host):
    """Stop using this"""
    agent_host.sendCommand("use 0")
    # Workaround to close chests, Issue #772
    global chest_open
    if chest_open:
        latest_observation = get_latest_world_observations(agent_host)
        agent_host.sendCommand(f"tp {latest_observation.get('XPos', 10.5)} {latest_observation.get('YPos', 227) + 8} {latest_observation.get('ZPos', 3) - 2}")
        agent_host.sendCommand("setYaw 180")
        agent_host.sendCommand("setYaw 0")
        agent_host.sendCommand("setPitch 0")
        chest_open = False

def go_through_entrance(agent_host):
    agent_host.sendCommand("pitch 0.5")
    time.sleep(0.5)
    agent_host.sendCommand("pitch 0")
    time.sleep(0.5)
    agent_host.sendCommand("use 1"); agent_host.sendCommand("use 0")

    time.sleep(0.5)
    agent_host.sendCommand("move 1")
    time.sleep(0.5)
    agent_host.sendCommand("move 0")

    agent_host.sendCommand("pitch -0.5")
    time.sleep(0.5)
    agent_host.sendCommand("pitch 0")

    time.sleep(0.1)
    agent_host.sendCommand("setYaw 180")
    time.sleep(0.2)

    agent_host.sendCommand("pitch 0.5")
    time.sleep(0.5)
    agent_host.sendCommand("pitch 0")

    time.sleep(0.5)
    agent_host.sendCommand("use 1"); agent_host.sendCommand("use 0")

def task_execution_print(agent_host, task):
    desc, response = LABEL_INFO[task]["desc"], LABEL_INFO[task]["response"]
    print(f"Executing task: {desc}")
    if response:
        agent_host.sendCommand(f"chat {response}")

def reset_agent(agent_host, teleport_x=0.5, teleport_z=0, teleport_to_spawn=False):
    """Directly teleport to spawn and reset direction agent is facing."""
    if teleport_to_spawn:
        tp_command = "tp " + str(teleport_x) + " 227 " + str(teleport_z)
        agent_host.sendCommand(tp_command)
    agent_host.sendCommand("setYaw 0")
    agent_host.sendCommand("setPitch 0")