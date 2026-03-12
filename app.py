import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import json

MAX_SPELLS = 25

#############################################
# LOAD SPELL DATABASE
#############################################

@st.cache_data
def load_spells():
    return pd.read_csv("spells.csv")

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
# SPELL INPUT
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
# SPELL SCALING
#############################################

def calculate_scaled_values(spell, slot):

    n = spell["base_dice"]
    s = spell["dice_sides"]
    rays = spell["rays"]

    if slot > spell["base_level"]:
        n += (slot - spell["base_level"]) * spell["scaling_dice"]
        rays += (slot - spell["base_level"]) * spell["scaling_rays"]

    targets = spell["targets"]

    return n, s, rays, targets

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
# DAMAGE RESULTS
#############################################

st.header("Damage Distributions")

fig, ax = plt.subplots()

table_data = []

for entry in spell_entries:

    spell = spell_db[spell_db["name"] == entry["name"]].iloc[0]

    n, s, rays, targets = calculate_scaled_values(spell, entry["slot"])

    save_mod = monster_saves.get(spell["save"], 0)

    x, pdf, mu_single = spell_distribution(
        n,
        s,
        DC,
        save_mod
    )

    mu_total = mu_single * rays * targets

    ax.plot(
        x,
        pdf,
        label=f"{entry['name']} (Lv {entry['slot']})"
    )

    table_data.append({
        "Spell": entry["name"],
        "Slot Level": entry["slot"],
        "Dice": f"{n}d{s}",
        "Rays": rays,
        "Targets": targets,
        "Save": spell["save"],
        "Damage Type": spell["damage_type"],
        "Expected Damage": round(mu_total,2)
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
