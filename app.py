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


#############################################
# LOAD SPELL DATABASE
#############################################

@st.cache_data
def load_spells():
    df = pd.read_csv("spells.csv")

    required_defaults = {
        "name": "",
        "base_level": 1,
        "base_dice": 0,
        "dice_sides": 0,
        "scaling_dice": 0,
        "save": "none",
        "damage_type": "none",
        "rays": 1,
        "scaling_rays": 0,
        "targets": 1,
        "attack_type": "save",
        "repeat_base_dice": 0,
        "repeat_dice_sides": 0,
        "repeat_scaling_dice": 0,
        "repeat_save": "none",
        "repeat_attack_type": "none",
        "repeat_rays": 1,
        "repeat_scaling_rays": 0,
        "repeat_targets": 1,
        "max_rounds": 1,
    }

    for col, default_value in required_defaults.items():
        if col not in df.columns:
            df[col] = default_value

    text_cols = ["name", "damage_type"]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["damage_type"] = df["damage_type"].str.lower()
    df["save"] = df["save"].apply(normalize_save)
    df["repeat_save"] = df["repeat_save"].apply(normalize_save)
    df["attack_type"] = df["attack_type"].apply(lambda x: normalize_attack_type(x, default="save"))
    df["repeat_attack_type"] = df["repeat_attack_type"].apply(lambda x: normalize_attack_type(x, default="none"))

    numeric_cols = [
        "base_level",
        "base_dice",
        "dice_sides",
        "scaling_dice",
        "rays",
        "scaling_rays",
        "targets",
        "repeat_base_dice",
        "repeat_dice_sides",
        "repeat_scaling_dice",
        "repeat_rays",
        "repeat_scaling_rays",
        "repeat_targets",
        "max_rounds",
    ]

    for col in numeric_cols:
        df[col] = df[col].apply(extract_first_int)

    df["base_level"] = df["base_level"].clip(lower=1, upper=9)
    df["base_dice"] = df["base_dice"].clip(lower=0)
    df["dice_sides"] = df["dice_sides"].clip(lower=0)
    df["scaling_dice"] = df["scaling_dice"].clip(lower=0)
    df["rays"] = df["rays"].clip(lower=1)
    df["scaling_rays"] = df["scaling_rays"].clip(lower=0)
    df["targets"] = df["targets"].clip(lower=1)

    df["repeat_base_dice"] = df["repeat_base_dice"].clip(lower=0)
    df["repeat_dice_sides"] = df["repeat_dice_sides"].clip(lower=0)
    df["repeat_scaling_dice"] = df["repeat_scaling_dice"].clip(lower=0)
    df["repeat_rays"] = df["repeat_rays"].clip(lower=1)
    df["repeat_scaling_rays"] = df["repeat_scaling_rays"].clip(lower=0)
    df["repeat_targets"] = df["repeat_targets"].clip(lower=1)
    df["max_rounds"] = df["max_rounds"].clip(lower=1)

    df = df[df["name"] != ""].copy()
    df = df.drop_duplicates(subset="name", keep="first").reset_index(drop=True)

    return df


spell_db = load_spells()

#############################################
# PAGE HEADER
#############################################

st.title("D&D 5e Spell Damage Analyzer")

st.write(
    "Compare D&D 5e spell damage including saves, attack rolls, multi-ray spells, AoE targets, repeat damage, and multiple rounds."
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
# SPELL SCALING
#############################################

def calculate_scaled_component(spell, slot, prefix=""):
    if prefix == "":
        base_dice = int(spell["base_dice"])
        dice_sides = int(spell["dice_sides"])
        scaling_dice = int(spell["scaling_dice"])
        rays = int(spell["rays"])
        scaling_rays = int(spell["scaling_rays"])
        targets = int(spell["targets"])
        save_type = spell["save"]
        attack_type = spell["attack_type"]
    else:
        base_dice = int(spell[f"{prefix}base_dice"])
        dice_sides = int(spell[f"{prefix}dice_sides"])
        scaling_dice = int(spell[f"{prefix}scaling_dice"])
        rays = int(spell[f"{prefix}rays"])
        scaling_rays = int(spell[f"{prefix}scaling_rays"])
        targets = int(spell[f"{prefix}targets"])
        save_type = spell[f"{prefix}save"]
        attack_type = spell[f"{prefix}attack_type"]

    base_level = int(spell["base_level"])

    if slot > base_level:
        level_diff = slot - base_level
        base_dice += level_diff * scaling_dice
        rays += level_diff * scaling_rays

    rays = max(rays, 1)
    targets = max(targets, 1)

    return {
        "dice": int(base_dice),
        "sides": int(dice_sides),
        "rays": int(rays),
        "targets": int(targets),
        "save": save_type,
        "attack_type": attack_type,
    }

#############################################
# DAMAGE STAT FUNCTIONS
#############################################

def base_damage_stats(n, s):
    if n <= 0 or s <= 0:
        return 0.0, 0.0

    mu = n * (s + 1) / 2
    var_single_die = (s**2 - 1) / 12
    var = n * var_single_die
    return float(mu), float(var)


def single_instance_stats_save(n, s, DC, save_mod):
    mu_hit, var_hit = base_damage_stats(n, s)

    if mu_hit == 0:
        return 0.0, 0.0

    fail_prob = (DC - save_mod - 1) / 20
    fail_prob = max(0.0, min(1.0, fail_prob))

    mu_save = mu_hit / 2
    var_save = var_hit / 4

    e2_hit = var_hit + mu_hit**2
    e2_save = var_save + mu_save**2

    mu = fail_prob * mu_hit + (1 - fail_prob) * mu_save
    e2 = fail_prob * e2_hit + (1 - fail_prob) * e2_save
    var = max(e2 - mu**2, 0.0)

    return mu, var


def single_instance_stats_attack(n, s, attack_bonus, target_ac, roll_mode):
    mu_hit, var_hit = base_damage_stats(n, s)

    if mu_hit == 0:
        return 0.0, 0.0

    hit_prob = attack_hit_probability(attack_bonus, target_ac, roll_mode)

    e2_hit = var_hit + mu_hit**2
    mu = hit_prob * mu_hit
    e2 = hit_prob * e2_hit
    var = max(e2 - mu**2, 0.0)

    return mu, var


def single_instance_stats_auto(n, s):
    mu, var = base_damage_stats(n, s)
    return mu, var


def component_stats(component, DC, monster_saves, attack_bonus, target_ac, roll_mode):
    n = component["dice"]
    s = component["sides"]
    attack_type = component["attack_type"]
    save_type = component["save"]

    if attack_type == "attack":
        mu_single, var_single = single_instance_stats_attack(
            n=n,
            s=s,
            attack_bonus=attack_bonus,
            target_ac=target_ac,
            roll_mode=roll_mode
        )
        resolution = f"Attack ({roll_mode})"
    elif attack_type == "none":
        mu_single, var_single = single_instance_stats_auto(n=n, s=s)
        resolution = "Automatic"
    else:
        save_mod = monster_saves.get(save_type, 0)
        mu_single, var_single = single_instance_stats_save(
            n=n,
            s=s,
            DC=DC,
            save_mod=save_mod
        )
        resolution = f"{save_type} Save"

    return mu_single, var_single, resolution


def apply_target_override(targets_from_csv, aoe_override):
    if int(targets_from_csv) > 1:
        return int(aoe_override)
    return int(targets_from_csv)


def make_distribution(mu, var):
    sigma = max(np.sqrt(max(var, 0.0)), 1e-9)
    xmax = max(mu + 4 * sigma, 1)
    x = np.linspace(0, xmax, 700)
    pdf = safe_norm_pdf(x, mu, sigma)
    return x, pdf


#############################################
# RESULTS: SINGLE-ROUND / INITIAL DAMAGE
#############################################

st.header("Initial Cast Damage Distributions")

fig_initial, ax_initial = plt.subplots()
table_data = []

for entry in spell_entries:
    spell = spell_db[spell_db["name"] == entry["name"]].iloc[0]

    initial = calculate_scaled_component(spell, entry["slot"], prefix="")
    init_targets = apply_target_override(initial["targets"], default_aoe_targets)

    mu_single, var_single, init_resolution = component_stats(
        component=initial,
        DC=DC,
        monster_saves=monster_saves,
        attack_bonus=spell_attack_bonus,
        target_ac=target_ac,
        roll_mode=attack_roll_mode
    )

    init_instances = int(initial["rays"]) * int(init_targets)
    mu_initial_total = mu_single * init_instances
    var_initial_total = var_single * init_instances

    x_init, pdf_init = make_distribution(mu_initial_total, var_initial_total)

    ax_initial.plot(
        x_init,
        pdf_init,
        label=f"{entry['name']} (Lv {entry['slot']})"
    )

    repeat = calculate_scaled_component(spell, entry["slot"], prefix="repeat_")
    repeat_targets = apply_target_override(repeat["targets"], default_aoe_targets)

    repeat_mu_single, repeat_var_single, repeat_resolution = component_stats(
        component=repeat,
        DC=DC,
        monster_saves=monster_saves,
        attack_bonus=spell_attack_bonus,
        target_ac=target_ac,
        roll_mode=attack_roll_mode
    )

    repeat_instances_per_round = int(repeat["rays"]) * int(repeat_targets)

    active_rounds = min(int(combat_rounds), int(spell["max_rounds"]))
    repeat_rounds = max(active_rounds - 1, 0)

    mu_repeat_total = repeat_mu_single * repeat_instances_per_round * repeat_rounds
    var_repeat_total = repeat_var_single * repeat_instances_per_round * repeat_rounds

    mu_full_spell = mu_initial_total + mu_repeat_total
    var_full_spell = var_initial_total + var_repeat_total

    initial_dice_text = f"{initial['dice']}d{initial['sides']}" if initial["dice"] > 0 and initial["sides"] > 0 else "0"
    repeat_dice_text = f"{repeat['dice']}d{repeat['sides']}" if repeat["dice"] > 0 and repeat["sides"] > 0 else "0"

    table_data.append({
        "Spell": entry["name"],
        "Slot Level": entry["slot"],
        "Initial Dice": initial_dice_text,
        "Initial Rays": int(initial["rays"]),
        "Initial Targets": int(init_targets),
        "Initial Resolution": init_resolution,
        "Repeat Dice": repeat_dice_text,
        "Repeat Rays": int(repeat["rays"]),
        "Repeat Targets": int(repeat_targets),
        "Repeat Resolution": repeat_resolution,
        "Repeat Rounds": int(repeat_rounds),
        "Initial Expected Damage": round(mu_initial_total, 2),
        "Repeat Expected Damage": round(mu_repeat_total, 2),
        "Full Spell Expected Damage": round(mu_full_spell, 2),
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

    initial = calculate_scaled_component(spell, entry["slot"], prefix="")
    init_targets = apply_target_override(initial["targets"], default_aoe_targets)

    mu_single_init, var_single_init, _ = component_stats(
        component=initial,
        DC=DC,
        monster_saves=monster_saves,
        attack_bonus=spell_attack_bonus,
        target_ac=target_ac,
        roll_mode=attack_roll_mode
    )

    init_instances = int(initial["rays"]) * int(init_targets)
    mu_initial_total = mu_single_init * init_instances
    var_initial_total = var_single_init * init_instances

    repeat = calculate_scaled_component(spell, entry["slot"], prefix="repeat_")
    repeat_targets = apply_target_override(repeat["targets"], default_aoe_targets)

    mu_single_repeat, var_single_repeat, _ = component_stats(
        component=repeat,
        DC=DC,
        monster_saves=monster_saves,
        attack_bonus=spell_attack_bonus,
        target_ac=target_ac,
        roll_mode=attack_roll_mode
    )

    repeat_instances_per_round = int(repeat["rays"]) * int(repeat_targets)
    active_rounds = min(int(combat_rounds), int(spell["max_rounds"]))
    repeat_rounds = max(active_rounds - 1, 0)

    mu_repeat_total = mu_single_repeat * repeat_instances_per_round * repeat_rounds
    var_repeat_total = var_single_repeat * repeat_instances_per_round * repeat_rounds

    mu_full = mu_initial_total + mu_repeat_total
    var_full = var_initial_total + var_repeat_total

    x_full, pdf_full = make_distribution(mu_full, var_full)

    ax_full.plot(
        x_full,
        pdf_full,
        label=f"{entry['name']} (Lv {entry['slot']})"
    )

ax_full.set_xlabel("Total Spell Damage")
ax_full.set_ylabel("Probability Density")
ax_full.set_title("Full Spell Damage Distribution Across Rays, Targets, and Rounds")
ax_full.legend(fontsize=8)

st.pyplot(fig_full)

#############################################
# RESULTS TABLE
#############################################

df = pd.DataFrame(table_data)

st.header("Expected Damage Table")
st.dataframe(df)
