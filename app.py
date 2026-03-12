import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

st.title("D&D Spell Damage Analyzer")

st.write("Compare expected damage distributions for multiple spells.")

# Sidebar inputs
st.sidebar.header("Global Settings")

DC = st.sidebar.number_input("Spell Save DC", value=15)
save_mod = st.sidebar.number_input("Enemy Save Modifier", value=2)

num_spells = st.sidebar.slider("Number of spells to compare", 1, 5, 2)

spells = []

st.header("Spell Inputs")

for i in range(num_spells):

    st.subheader(f"Spell {i+1}")

    col1, col2, col3 = st.columns(3)

    name = col1.text_input(f"Spell name {i}", f"Spell {i+1}")
    n = col2.number_input(f"Number of dice {i}", 1, 20, 8)
    s = col3.number_input(f"Sides per die {i}", 2, 20, 6)

    spells.append((name, n, s))

def spell_distribution(n, s, DC, save_mod):

    mu_hit = n * (s + 1) / 2

    var = n * ((s + 1) * (2*s + 1) / 6) - ((s+1)/2)**2
    sigma_hit = np.sqrt(var)

    mu_save = mu_hit / 2
    sigma_save = sigma_hit / 2

    h = (DC - save_mod - 1) / 20
    h = max(0, min(1, h))

    mu_total = h * mu_hit + (1-h) * mu_save

    x = np.linspace(0, mu_hit*2, 500)

    pdf = h*norm.pdf(x, mu_hit, sigma_hit) + (1-h)*norm.pdf(x, mu_save, sigma_save)

    return x, pdf, mu_total

st.header("Results")

fig, ax = plt.subplots()

data = []

for name, n, s in spells:

    x, pdf, mu = spell_distribution(n, s, DC, save_mod)

    ax.plot(x, pdf, label=name)

    data.append({
        "Spell": name,
        "Dice": f"{n}d{s}",
        "Mean Damage": round(mu,2)
    })

ax.set_xlabel("Damage")
ax.set_ylabel("Probability Density")
ax.set_title("Damage Distribution")
ax.legend()

st.pyplot(fig)

df = pd.DataFrame(data)

st.subheader("Expected Damage Table")

st.dataframe(df)