import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import json
import re

MAX_SPELLS = 25

#############################################
# HELPER FUNCTIONS
#############################################

def extract_first_int(value, default=0):
    if pd.isna(value):
        return default

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, float):
        if np.isnan(value):
            return default
        return int(value)

    text = str(value).strip()
    match = re.search(r"-?\d+", text)
    if match:
        return int(match.group())

    return default


def normalize_save(value):
    if pd.isna(value):
        return "none"

    text = str(value).strip().upper()

    mapping = {
        "STR": "STR",
        "STRENGTH": "STR",
        "DEX": "DEX",
        "DEXTERITY": "DEX",
        "CON": "CON",
        "CONSTITUTION": "CON",
        "INT": "INT",
        "INTELLIGENCE": "INT",
        "WIS": "WIS",
        "WISDOM": "WIS",
        "CHA": "CHA",
        "CHARISMA": "CHA",
        "NONE": "none",
        "": "none",
    }

    return mapping.get(text, "none")


def normalize_attack_type(value, default="save"):
    if pd.isna(value):
        return default

    text = str(value).strip().lower()

    if text in ["attack", "spell attack", "attack roll", "roll"]:
        return "attack"
    if text in ["none", "auto", "automatic"]:
        return "none"
    return "save"


def attack_hit_probability(attack_bonus, target_ac, roll_mode="normal"):
    single_hit_outcomes = 0

    for roll in range(1, 21):
        if roll == 1:
            hit = False
        elif roll == 20:
            hit = True
        else:
            hit = (roll + attack_bonus) >= target_ac

        if hit:
            single_hit_outcomes += 1

    p_single = single_hit_outcomes / 20

    if roll_mode == "advantage":
        return 1 - (1 - p_single) ** 2
    elif roll_mode == "disadvantage":
        return p_single ** 2
    else:
        return p_single


def safe_norm_pdf(x, mu, sigma):
    sigma = max(float(sigma), 1e-9)
    return norm.pdf(x, mu, sigma)


def expected_targets(targets_from_csv, aoe_override):
    """
    Override only when the packet is clearly multi-target.
    """
    targets_from_csv = max(int(targets_from_csv), 1)
    if targets_from_csv > 1:
        return int(aoe_override)
    return targets_from_csv


#############################################
# LOAD SPELL DATABASE
#############################################

@st.cache_data
def load_spells():
    df = pd.read_csv("spells.csv")

    required_cols = ["name", "base_level", "max_rounds", "notes"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    for slot in range(1, 10):
        for phase in ["initial", "repeat"]:
            col = f"slot{slot}_{phase}"
            if col not in df.columns:
                df[col] = "[]"

    df["name"] = df["name"].fillna("").astype(str).str.strip()
    df["base_level"] = df["base_level"].apply(lambda x: extract_first_int(x, 1)).clip(lower=1, upper=9)
    df["max_rounds"] = df["max_rounds"].apply(lambda x: extract_first_int(x, 1)).clip(lower=1)
    df["notes"] = df["notes"].fillna("").astype(str).str.strip()

    for slot in range(1, 10):
        for phase in ["initial", "repeat"]:
            col = f"slot{slot}_{phase}"
            df[col] = df[col].fillna("[]").astype(str)

    df = df[df["name"] != ""].copy()
    df = df.drop_duplicates(subset="name", keep="first").reset_index(drop=True)

    return df


spell_db = load_spells()

#############################################
# JSON PACKET PARSING
#############################################

def parse_packet_list(cell_value):
    """
    Parse a JSON array of packet dicts from a CSV cell.
    Returns a list of normalized packet dicts.
    """
    if pd.isna(cell_value):
        return []

    text = str(cell_value).strip()
    if text == "" or text.lower() == "nan":
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    packets = []
    for raw in data:
        if not isinstance(raw, dict):
            continue

        packet = {
            "dice": extract_first_int(raw.get("dice", 0), 0),
            "sides": extract_first_int(raw.get("sides", 0), 0),
            "flat": extract_first_int(raw.get("flat", 0), 0),
            "attack_type": normalize_attack_type(raw.get("attack_type", "save"), default="save"),
            "save": normalize_save(raw.get("save", "none")),
            "damage_type": str(raw.get("damage_type", "none")).strip().lower(),
            "instances": max(extract_first_int(raw.get("instances", 1), 1), 1),
            "targets": max(extract_first_int(raw.get("targets", 1), 1), 1),
            "notes": str(raw.get("notes", "")).strip(),
        }
        packets.append(packet)

    return packets


def get_spell_packets(spell_row, slot_level, phase):
    """
    phase = 'initial' or 'repeat'
    Returns parsed packet list for the chosen spell slot.
    """
    slot_level = int(slot_level)
    if slot_level < 1 or slot_level > 9:
        return []

    col = f"slot{slot_level}_{phase}"
    if col not in spell_row.index:
        return []

    return parse_packet_list(spell_row[col])


#############################################
# DAMAGE STAT FUNCTIONS
#############################################

def base_damage_stats(dice, sides, flat=0):
    """
    Returns mean and variance for dice*sides + flat.
    """
    dice = int(dice)
    sides = int(sides)
    flat = float(flat)

    if dice <= 0 or sides <= 0:
        mu = flat
        var = 0.0
        return mu, var

    mu = dice * (sides + 1) / 2 + flat
    var_single_die = (sides**2 - 1) / 12
    var = dice * var_single_die
    return float(mu), float(var)


def packet_single_instance_stats(packet, DC, monster_saves, spell_attack_bonus, target_ac, attack_roll_mode):
    """
    Returns mean, variance, and resolution label for ONE instance against ONE target.
    """
    mu_hit, var_hit = base_damage_stats(packet["dice"], packet["sides"], packet["flat"])

    if mu_hit == 0 and var_hit == 0:
        return 0.0, 0.0, "No Damage"

    attack_type = packet["attack_type"]

    if attack_type == "attack":
        hit_prob = attack_hit_probability(spell_attack_bonus, target_ac, attack_roll_mode)
        e2_hit = var_hit + mu_hit**2
        mu = hit_prob * mu_hit
        e2 = hit_prob * e2_hit
        var = max(e2 - mu**2, 0.0)
        resolution = f"Attack ({attack_roll_mode})"

    elif attack_type == "none":
        mu = mu_hit
        var = var_hit
        resolution = "Automatic"

    else:
        save_type = packet["save"]
        save_mod = monster_saves.get(save_type, 0)

        fail_prob = (DC - save_mod - 1) / 20
        fail_prob = max(0.0, min(1.0, fail_prob))

        # Model save spells as full damage on fail, half on success
        mu_save = mu_hit / 2
        var_save = var_hit / 4

        e2_hit = var_hit + mu_hit**2
        e2_save = var_save + mu_save**2

        mu = fail_prob * mu_hit + (1 - fail_prob) * mu_save
        e2 = fail_prob * e2_hit + (1 - fail_prob) * e2_save
        var = max(e2 - mu**2, 0.0)
        resolution = f"{save_type} Save"

    return mu, var, resolution


def aggregate_packet_group_stats(packet_list, DC, monster_saves, spell_attack_bonus, target_ac, attack_roll_mode, aoe_override):
    """
    Aggregates mean/variance across all packets in one phase.
    Assumes independence between packets/instances/targets.
    """
    total_mu = 0.0
    total_var = 0.0
    breakdown_rows = []

    for packet in packet_list:
        mu_one, var_one, resolution = packet_single_instance_stats(
            packet=packet,
            DC=DC,
            monster_saves=monster_saves,
            spell_attack_bonus=spell_attack_bonus,
            target_ac=target_ac,
            attack_roll_mode=attack_roll_mode,
        )

        targets = expected_targets(packet["targets"], aoe_override)
        instances = int(packet["instances"])
        total_copies = instances * targets

        packet_mu = mu_one * total_copies
        packet_var = var_one * total_copies

        total_mu += packet_mu
        total_var += packet_var

        packet_label = f"{packet['dice']}d{packet['sides']}"
        if packet["flat"] != 0:
            packet_label += f" + {packet['flat']}"

        breakdown_rows.append({
            "Damage Type": packet["damage_type"],
            "Packet": packet_label,
            "Instances": instances,
            "Targets": targets,
            "Resolution": resolution,
            "Expected Damage": round(packet_mu, 2),
        })

    return total_mu, total_var, breakdown_rows


def make_distribution(mu, var):
    sigma = max(np.sqrt(max(var, 0.0)), 1e-9)
    xmax = max(mu + 4 * sigma, 1)
    x = np.linspace(0, xmax, 700)
    pdf = safe_norm_pdf(x, mu, sigma)
    return x, pdf


#############################################
# PAGE HEADER
#############################################

st.title("D&D 5e Spell Damage Analyzer")

st.write(
    "Compare D&D 5e spell damage using slot-specific damage packets, attack rolls, saves, repeat damage, and multi-round effects."
)

#############################################
# GLOBAL SETTINGS
#############################################

st.sidebar.header("Global Settings")

DC = st.sidebar.number_input(
    "Spell Save DC",
    min_value=1,
    max_value=30,
    value=15
)

target_ac = st.sidebar.number_input(
    "Target Armor Class",
    min_value=1,
    max_value=40,
    value=15
)

spell_attack_bonus = st.sidebar.number_input(
    "Spell Attack Bonus",
    min_value=-5,
    max_value=20,
    value=7
)

attack_roll_mode = st.sidebar.selectbox(
    "Attack Roll Mode",
    ["normal", "advantage", "disadvantage"],
    index=0
)

default_aoe_targets = st.sidebar.slider(
    "Average AoE Targets Hit",
    min_value=1,
    max_value=20,
    value=4
)

combat_rounds = st.sidebar.number_input(
    "Combat Length (Rounds)",
    min_value=1,
    max_value=20,
    value=3
)

#############################################
# ENEMY SAVE MODIFIERS
#############################################

st.sidebar.header("Enemy Save Modifiers")

monster_saves = {
    "STR": st.sidebar.number_input("STR Save", value=0),
    "DEX": st.sidebar.number_input("DEX Save", value=2),
    "CON": st.sidebar.number_input("CON Save", value=3),
    "INT": st.sidebar.number_input("INT Save", value=0),
    "WIS": st.sidebar.number_input("WIS Save", value=1),
    "CHA": st.sidebar.number_input("CHA Save", value=0),
}

#############################################
# SAVE / LOAD SPELL SETS
#############################################

st.sidebar.header("Spell Set")

uploaded = st.sidebar.file_uploader("Load Spell Set", type="json")

if uploaded:
    spell_config = json.load(uploaded)
else:
    spell_config = []

download_data = json.dumps(spell_config, indent=2)

st.sidebar.download_button(
    "Download Current Spell Set",
    download_data,
    file_name="spell_set.json"
)

#############################################
# NUMBER OF SPELLS
#############################################

num_spells = int(
    st.number_input(
        "Number of spells to compare",
        min_value=1,
        max_value=MAX_SPELLS,
        value=5
    )
)

#############################################
# SPELL INPUT
#############################################

st.header("Spell Selection")

spell_entries = []
spell_names = sorted(spell_db["name"].tolist())

for i in range(num_spells):
    with st.expander(f"Spell {i+1}", expanded=False):
        spell_name = st.selectbox(
            "Search Spell",
            spell_names,
            key=f"spell{i}"
        )

        spell = spell_db[spell_db["name"] == spell_name].iloc[0]
        base_level = int(spell["base_level"])

        slot_level = st.number_input(
            "Spell Slot Level",
            min_value=base_level,
            max_value=9,
            value=base_level,
            key=f"slot{i}"
        )

        spell_entries.append({
            "name": spell_name,
            "slot": int(slot_level)
        })

#############################################
# RESULTS: INITIAL CAST
#############################################

st.header("Initial Cast Damage Distributions")

fig_initial, ax_initial = plt.subplots()
summary_rows = []

for entry in spell_entries:
    spell = spell_db[spell_db["name"] == entry["name"]].iloc[0]
    slot = int(entry["slot"])

    initial_packets = get_spell_packets(spell, slot, "initial")

    mu_initial, var_initial, breakdown_initial = aggregate_packet_group_stats(
        packet_list=initial_packets,
        DC=DC,
        monster_saves=monster_saves,
        spell_attack_bonus=spell_attack_bonus,
        target_ac=target_ac,
        attack_roll_mode=attack_roll_mode,
        aoe_override=default_aoe_targets
    )

    x_init, pdf_init = make_distribution(mu_initial, var_initial)

    ax_initial.plot(
        x_init,
        pdf_init,
        label=f"{entry['name']} (Lv {slot})"
    )

    repeat_packets = get_spell_packets(spell, slot, "repeat")
    active_rounds = min(int(combat_rounds), int(spell["max_rounds"]))
    repeat_rounds = max(active_rounds - 1, 0)

    mu_repeat_one_round, var_repeat_one_round, breakdown_repeat = aggregate_packet_group_stats(
        packet_list=repeat_packets,
        DC=DC,
        monster_saves=monster_saves,
        spell_attack_bonus=spell_attack_bonus,
        target_ac=target_ac,
        attack_roll_mode=attack_roll_mode,
        aoe_override=default_aoe_targets
    )

    mu_repeat_total = mu_repeat_one_round * repeat_rounds
    var_repeat_total = var_repeat_one_round * repeat_rounds

    mu_full = mu_initial + mu_repeat_total
    var_full = var_initial + var_repeat_total

    damage_types = sorted({
        row["Damage Type"]
        for row in breakdown_initial + breakdown_repeat
        if row["Damage Type"] != "none"
    })
    damage_type_text = ", ".join(damage_types) if damage_types else "none"

    summary_rows.append({
        "Spell": entry["name"],
        "Slot Level": slot,
        "Damage Types": damage_type_text,
        "Initial Expected Damage": round(mu_initial, 2),
        "Repeat / Round": round(mu_repeat_one_round, 2),
        "Repeat Rounds": repeat_rounds,
        "Full Spell Expected Damage": round(mu_full, 2),
        "Max Rounds": int(spell["max_rounds"]),
        "Notes": spell["notes"],
    })

ax_initial.set_xlabel("Damage")
ax_initial.set_ylabel("Probability Density")
ax_initial.set_title("Initial Cast Damage Distribution")
ax_initial.legend(fontsize=8)

st.pyplot(fig_initial)

#############################################
# RESULTS: FULL SPELL DAMAGE
#############################################

st.header("Full Spell Damage Distributions")

fig_full, ax_full = plt.subplots()

for entry in spell_entries:
    spell = spell_db[spell_db["name"] == entry["name"]].iloc[0]
    slot = int(entry["slot"])

    initial_packets = get_spell_packets(spell, slot, "initial")
    repeat_packets = get_spell_packets(spell, slot, "repeat")

    mu_initial, var_initial, _ = aggregate_packet_group_stats(
        packet_list=initial_packets,
        DC=DC,
        monster_saves=monster_saves,
        spell_attack_bonus=spell_attack_bonus,
        target_ac=target_ac,
        attack_roll_mode=attack_roll_mode,
        aoe_override=default_aoe_targets
    )

    mu_repeat_one_round, var_repeat_one_round, _ = aggregate_packet_group_stats(
        packet_list=repeat_packets,
        DC=DC,
        monster_saves=monster_saves,
        spell_attack_bonus=spell_attack_bonus,
        target_ac=target_ac,
        attack_roll_mode=attack_roll_mode,
        aoe_override=default_aoe_targets
    )

    active_rounds = min(int(combat_rounds), int(spell["max_rounds"]))
    repeat_rounds = max(active_rounds - 1, 0)

    mu_full = mu_initial + mu_repeat_one_round * repeat_rounds
    var_full = var_initial + var_repeat_one_round * repeat_rounds

    x_full, pdf_full = make_distribution(mu_full, var_full)

    ax_full.plot(
        x_full,
        pdf_full,
        label=f"{entry['name']} (Lv {slot})"
    )

ax_full.set_xlabel("Total Spell Damage")
ax_full.set_ylabel("Probability Density")
ax_full.set_title("Full Spell Damage Distribution")
ax_full.legend(fontsize=8)

st.pyplot(fig_full)

#############################################
# SUMMARY TABLE
#############################################

st.header("Expected Damage Table")
summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, use_container_width=True)

#############################################
# OPTIONAL BREAKDOWN TABLES
#############################################

st.header("Per-Spell Packet Breakdown")

for entry in spell_entries:
    spell = spell_db[spell_db["name"] == entry["name"]].iloc[0]
    slot = int(entry["slot"])

    initial_packets = get_spell_packets(spell, slot, "initial")
    repeat_packets = get_spell_packets(spell, slot, "repeat")

    _, _, breakdown_initial = aggregate_packet_group_stats(
        packet_list=initial_packets,
        DC=DC,
        monster_saves=monster_saves,
        spell_attack_bonus=spell_attack_bonus,
        target_ac=target_ac,
        attack_roll_mode=attack_roll_mode,
        aoe_override=default_aoe_targets
    )

    _, _, breakdown_repeat = aggregate_packet_group_stats(
        packet_list=repeat_packets,
        DC=DC,
        monster_saves=monster_saves,
        spell_attack_bonus=spell_attack_bonus,
        target_ac=target_ac,
        attack_roll_mode=attack_roll_mode,
        aoe_override=default_aoe_targets
    )

    with st.expander(f"{entry['name']} (Lv {slot}) breakdown", expanded=False):
        st.write(f"**Notes:** {spell['notes'] if spell['notes'] else 'None'}")

        st.write("**Initial packets**")
        if breakdown_initial:
            st.dataframe(pd.DataFrame(breakdown_initial), use_container_width=True)
        else:
            st.write("No initial damage packets.")

        st.write("**Repeat packets**")
        if breakdown_repeat:
            st.dataframe(pd.DataFrame(breakdown_repeat), use_container_width=True)
        else:
            st.write("No repeat damage packets.")
