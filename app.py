import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import json

MAX_SPELLS = 25

#############################################
# LOAD DATABASES
#############################################

@st.cache_data
def load_spells():
    return pd.read_csv("spells.csv")

@st.cache_data
def load_monsters():
    return pd.read_csv("monsters.csv")

spell_db = load_spells()
monster_db = load_monsters()

#############################################
# COMPUTE RESISTANCE STATISTICS
#############################################

def compute_resistance_rates(monster_df):

    damage_types = [
        "fire","cold","lightning","acid","poison",
        "necrotic","radiant","thunder","psychic","force"
    ]

    results = {}

    total = len(monster_df)

    for dmg in damage_types:

        resistant = monster_df["resistances"].fillna("").str.contains(dmg).sum()
        immune = monster_df["immunities"].fillna("").str.contains(dmg).sum()

        results[dmg] = {
            "resistant": resistant/total,
            "immune": immune/total
        }

    return results

RESISTANCE_STATS = compute_resistance_rates(monster_db)

#############################################
# PAGE HEADER
#############################################

st.title("D&D 5e Spell Damage Analyzer")

st.write(
"Compare damage distributions using PHB spell mechanics and Monster Manual resistances."
)

#############################################
# GLOBAL SETTINGS
#############################################

st.sidebar.header("Global Settings")

DC = st.sidebar.number_input("Spell Save DC", min_value=1, max_value=30, value=15)

use_resistance = st.sidebar.checkbox(
    "Account for Monster Manual resistances",
    value=False
)

#############################################
# MONSTER SAVE MODIFIERS
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

num_spells = int(st.number_input(
    "Number of spells to compare",
    min_value=1,
    max_value=MAX_SPELLS,
    value=5
))

#############################################
# SPELL INPUT SECTION
#############################################

st.header("Spell Selection")

spell_entries = []

for i in range(num_spells):

    with st.expander(f"Spell {i+1}", expanded=False):

        spell_name = st.selectbox(
            "Search Spell",
            spell_db["name"].sort_values(),
            key=f"spell{i}"
        )

        spell = spell_db[spell_db["name"] == spell_name].iloc[0]

        slot_level = st.number_input(
            "Spell Slot Level",
            min_value=int(spell["base_level"]),
            max_value=9,
            value=int(spell["base_level"]),
            key=f"slot{i}"
        )

        spell_entries.append({
            "name": spell_name,
            "slot": slot_level
        })

#############################################
# SPELL SCALING FUNCTION
#############################################

def calculate_scaled_dice(spell, slot):

    n = spell["base_dice"]
    s = spell["dice_sides"]

    if slot > spell["base_level"]:
        n += (slot - spell["base_level"]) * spell["scaling_dice"]

    return n, s

#############################################
# DAMAGE DISTRIBUTION MODEL
#############################################

def spell_distribution(n, s, DC, save_mod):

    mu_hit = n * (s + 1) / 2

    var = n * ((s + 1) * (2*s + 1) / 6) - ((s+1)/2)**2
    sigma_hit = np.sqrt(var)

    mu_save = mu_hit / 2
    sigma_save = sigma_hit / 2

    fail_prob = (DC - save_mod - 1)/20
    fail_prob = max(0, min(1, fail_prob))

    mu = fail_prob*mu_hit + (1-fail_prob)*mu_save

    x = np.linspace(0, mu_hit*2, 500)

    pdf = fail_prob*norm.pdf(x, mu_hit, sigma_hit) + (1-fail_prob)*norm.pdf(x, mu_save, sigma_save)

    return x, pdf, mu

#############################################
# RESULTS
#############################################

st.header("Damage Distributions")

fig, ax = plt.subplots()

table_data = []

for entry in spell_entries:

    spell = spell_db[spell_db["name"] == entry["name"]].iloc[0]

    n, s = calculate_scaled_dice(spell, entry["slot"])

    save_mod = monster_saves[spell["save"]]

    x, pdf, mu = spell_distribution(
        n,
        s,
        DC,
        save_mod
    )

    dmg_type = spell["damage_type"]

    if use_resistance:

        stats = RESISTANCE_STATS[dmg_type]

        resist = stats["resistant"]
        immune = stats["immune"]

        normal = 1 - resist - immune

        mu = normal*mu + resist*(mu/2)

    ax.plot(x, pdf, label=f"{entry['name']} (Lv {entry['slot']})")

    table_data.append({
        "Spell": entry["name"],
        "Slot Level": entry["slot"],
        "Dice": f"{n}d{s}",
        "Save": spell["save"],
        "Damage Type": dmg_type,
        "Expected Damage": round(mu,2)
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

#############################################
# RESISTANCE STATISTICS VIEW
#############################################

if use_resistance:

    st.header("Monster Manual Resistance Statistics")

    res_table = []

    for dmg in RESISTANCE_STATS:

        res_table.append({
            "Damage Type": dmg,
            "Resistant %": round(RESISTANCE_STATS[dmg]["resistant"]*100,2),
            "Immune %": round(RESISTANCE_STATS[dmg]["immune"]*100,2)
        })

    st.dataframe(pd.DataFrame(res_table))
