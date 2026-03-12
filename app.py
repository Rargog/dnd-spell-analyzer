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
    """
    Convert values like:
    4 -> 4
    '4' -> 4
    '4th-level' -> 4
    'Level 3' -> 3
    NaN / invalid -> default
    """
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
    """
    Normalize save values to one of:
    STR, DEX, CON, INT, WIS, CHA, none
    """
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


#############################################
# LOAD SPELL DATABASE
#############################################

@st.cache_data
def load_spells():
    df = pd.read_csv("spells.csv")

    # Ensure required columns exist
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
    }

    for col, default_value in required_defaults.items():
        if col not in df.columns:
            df[col] = default_value

    # Clean text columns
    df["name"] = df["name"].fillna("").astype(str).str.strip()
    df["damage_type"] = df["damage_type"].fillna("none").astype(str).str.strip().str.lower()
    df["save"] = df["save"].apply(normalize_save)

    # Clean numeric columns, including things like "4th-level"
    numeric_cols = [
        "base_level",
        "base_dice",
        "dice_sides",
        "scaling_dice",
        "rays",
        "scaling_rays",
        "targets",
    ]

    for col in numeric_cols:
        df[col] = df[col].apply(extract_first_int)

    # Enforce sane minimums
    df["base_level"] = df["base_level"].clip(lower=1, upper=9)
    df["base_dice"] = df["base_dice"].clip(lower=0)
    df["dice_sides"] = df["dice_sides"].clip(lower=0)
    df["scaling_dice"] = df["scaling_dice"].clip(lower=0)
    df["rays"] = df["rays"].clip(lower=1)
    df["scaling_rays"] = df["scaling_rays"].clip(lower=0)
    df["targets"] = df["targets"].clip(lower=1)

    # Remove blank spell names
    df = df[df["name"] != ""].copy()

    # Optional: drop duplicate names, keeping first occurrence
    df = df.drop_duplicates(subset="name", keep="first").reset_index(drop=True)

    return df


spell_db = load_spells()

#############################################
# PAGE HEADER
#############################################

st.title("D&D 5e Spell Damage Analyzer")

st.write(
    "Compare damage distributions for D&D 5e spells including multi-ray and multi-target mechanics."
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

def calculate_scaled_values(spell, slot):
    n = int(spell["base_dice"])
    s = int(spell["dice_sides"])
    rays = int(spell["rays"])
    scaling_dice = int(spell["scaling_dice"])
    scaling_rays = int(spell["scaling_rays"])
    base_level = int(spell["base_level"])
    targets = int(spell["targets"])

    if slot > base_level:
        level_diff = slot - base_level
        n += level_diff * scaling_dice
        rays += level_diff * scaling_rays

    rays = max(rays, 1)
    targets = max(targets, 1)

    return n, s, rays, targets

#############################################
# DAMAGE DISTRIBUTION MODEL
#############################################

def spell_distribution(n, s, DC, save_mod, has_save=True):
    """
    Returns x, pdf, mu for a single damage instance.
    If has_save is False, damage is treated as always full damage.
    """

    mu_hit = n * (s + 1) / 2

    if n <= 0 or s <= 0:
        x = np.linspace(0, 1, 500)
        pdf = np.zeros_like(x)
        pdf[0] = 1
        return x, pdf, 0.0

    # Variance of nds
    var_single_die = (s**2 - 1) / 12
    var = n * var_single_die
    sigma_hit = np.sqrt(max(var, 1e-9))

    if has_save:
        mu_save = mu_hit / 2
        sigma_save = sigma_hit / 2

        fail_prob = (DC - save_mod - 1) / 20
        fail_prob = max(0, min(1, fail_prob))

        mu = fail_prob * mu_hit + (1 - fail_prob) * mu_save

        x = np.linspace(0, max(mu_hit * 2, 1), 500)
        pdf = (
            fail_prob * norm.pdf(x, mu_hit, sigma_hit)
            + (1 - fail_prob) * norm.pdf(x, mu_save, sigma_save)
        )
    else:
        mu = mu_hit
        x = np.linspace(0, max(mu_hit * 2, 1), 500)
        pdf = norm.pdf(x, mu_hit, sigma_hit)

    return x, pdf, mu

#############################################
# DAMAGE RESULTS
#############################################

st.header("Damage Distributions")

fig, ax = plt.subplots()
table_data = []

for entry in spell_entries:
    spell = spell_db[spell_db["name"] == entry["name"]].iloc[0]

    n, s, rays, targets = calculate_scaled_values(spell, entry["slot"])
    save_type = spell["save"]

    has_save = save_type in monster_saves
    save_mod = monster_saves.get(save_type, 0)

    x, pdf, mu_single = spell_distribution(
        n=n,
        s=s,
        DC=DC,
        save_mod=save_mod,
        has_save=has_save
    )

    mu_total = mu_single * rays * targets

    ax.plot(
        x,
        pdf,
        label=f"{entry['name']} (Lv {entry['slot']})"
    )

    dice_text = f"{n}d{s}" if n > 0 and s > 0 else "0"

    table_data.append({
        "Spell": entry["name"],
        "Slot Level": entry["slot"],
        "Dice": dice_text,
        "Rays": rays,
        "Targets": targets,
        "Save": save_type,
        "Damage Type": spell["damage_type"],
        "Expected Damage": round(mu_total, 2)
    })

ax.set_xlabel("Damage")
ax.set_ylabel("Probability Density")
ax.set_title("Spell Damage Distribution")
ax.legend(fontsize=8)

st.pyplot(fig)

#############################################
# RESULTS TABLE
#############################################

df = pd.DataFrame(table_data)

st.header("Expected Damage Table")
st.dataframe(df)
