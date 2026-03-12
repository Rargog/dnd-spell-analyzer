import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import json

MAX_SPELLS = 25

st.title("D&D Spell Damage Analyzer")

st.write("Compare damage distributions for multiple spells.")

########################################
# GLOBAL SETTINGS
########################################

st.sidebar.header("Spell Save DC")

DC = st.sidebar.number_input("Spell Save DC", value=15)

########################################
# MONSTER SAVE MODIFIERS
########################################

st.sidebar.header("Monster Save Modifiers")

monster_saves = {
    "STR": st.sidebar.number_input("STR Save", value=0),
    "DEX": st.sidebar.number_input("DEX Save", value=2),
    "CON": st.sidebar.number_input("CON Save", value=3),
    "INT": st.sidebar.number_input("INT Save", value=0),
    "WIS": st.sidebar.number_input("WIS Save", value=1),
    "CHA": st.sidebar.number_input("CHA Save", value=0),
}

########################################
# LOAD SAVED SPELLS
########################################

uploaded_file = st.sidebar.file_uploader("Load Spell List", type="json")

if uploaded_file:
    spells_data = json.load(uploaded_file)
else:
    spells_data = [{"name":"Fireball","n":8,"s":6,"save":"DEX"}]

########################################
# NUMBER OF SPELLS
########################################

num_spells = st.number_input(
    "Number of spells",
    min_value=1,
    max_value=MAX_SPELLS,
    value=len(spells_data),
    step=1
)

########################################
# SPELL INPUTS
########################################

spells = []

st.header("Spell Definitions")

for i in range(num_spells):

    if i < len(spells_data):
        default = spells_data[i]
    else:
        default = {"name":f"Spell {i+1}","n":8,"s":6,"save":"DEX"}

    with st.expander(f"Spell {i+1}", expanded=False):

        col1,col2,col3,col4 = st.columns(4)

        name = col1.text_input("Name", default["name"], key=f"name{i}")
        n = col2.number_input("Dice",1,50,default["n"], key=f"n{i}")
        s = col3.number_input("Sides",2,20,default["s"], key=f"s{i}")
        save = col4.selectbox(
            "Save Stat",
            ["STR","DEX","CON","INT","WIS","CHA"],
            index=["STR","DEX","CON","INT","WIS","CHA"].index(default["save"]),
            key=f"save{i}"
        )

        spells.append({
            "name":name,
            "n":n,
            "s":s,
            "save":save
        })

########################################
# SAVE SPELL LIST
########################################

st.sidebar.header("Save Spell List")

json_data = json.dumps(spells,indent=2)

st.sidebar.download_button(
    "Download Spell List",
    json_data,
    file_name="spells.json"
)

########################################
# DAMAGE MODEL
########################################

def spell_distribution(n,s,DC,save_mod):

    mu_hit = n*(s+1)/2

    var = n*((s+1)*(2*s+1)/6) - ((s+1)/2)**2
    sigma_hit = np.sqrt(var)

    mu_save = mu_hit/2
    sigma_save = sigma_hit/2

    h = (DC - save_mod - 1)/20
    h = max(0,min(1,h))

    mu_total = h*mu_hit + (1-h)*mu_save

    x = np.linspace(0,mu_hit*2,500)

    pdf = h*norm.pdf(x,mu_hit,sigma_hit) + (1-h)*norm.pdf(x,mu_save,sigma_save)

    return x,pdf,mu_total

########################################
# RESULTS
########################################

st.header("Damage Distributions")

fig,ax = plt.subplots()

table_data = []

for spell in spells:

    save_mod = monster_saves[spell["save"]]

    x,pdf,mu = spell_distribution(
        spell["n"],
        spell["s"],
        DC,
        save_mod
    )

    ax.plot(x,pdf,label=spell["name"])

    table_data.append({
        "Spell":spell["name"],
        "Dice":f'{spell["n"]}d{spell["s"]}',
        "Save":spell["save"],
        "Enemy Save Mod":save_mod,
        "Expected Damage":round(mu,2)
    })

ax.set_xlabel("Damage")
ax.set_ylabel("Probability Density")
ax.set_title("Spell Damage Distributions")

ax.legend(fontsize=8)

st.pyplot(fig)

########################################
# TABLE
########################################

df = pd.DataFrame(table_data)

st.header("Expected Damage")

st.dataframe(df)

