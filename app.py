import io
import json
import html
import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Draw
from rdkit.Chem import Lipinski, Crippen, rdFingerprintGenerator, DataStructs

st.set_page_config(
    page_title="ChemLab | Virtual Chemoinformatics Laboratory",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Educational example molecules. SMILES are parsed and properties are calculated with RDKit.
MOLECULES = {
    "Water": {"smiles": "O", "description": "A small polar molecule essential to many chemical and biological processes."},
    "Methane": {"smiles": "C", "description": "The simplest alkane and a major component of natural gas."},
    "Ethanol": {"smiles": "CCO", "description": "An alcohol used as a solvent and in many consumer and laboratory products."},
    "Benzene": {"smiles": "c1ccccc1", "description": "An aromatic hydrocarbon with a six-membered ring."},
    "Phenol": {"smiles": "Oc1ccccc1", "description": "An aromatic compound containing a hydroxyl group directly attached to a benzene ring."},
    "Acetic acid": {"smiles": "CC(=O)O", "description": "A carboxylic acid and the main acid in vinegar."},
    "Aspirin": {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "description": "Acetylsalicylic acid, a widely used medicine; this page is for educational chemistry only."},
    "Caffeine": {"smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "description": "A naturally occurring stimulant found in coffee, tea, and other plants."},
    "Glucose": {"smiles": "C(C1C(C(C(C(O1)O)O)O)O)O", "description": "A simple sugar that serves as an important energy source in living systems."},
    "Paracetamol": {"smiles": "CC(=O)Nc1ccc(O)cc1", "description": "Also known as acetaminophen; included here as a molecular structure example."},
}

CUSTOM_CSS = """
<style>
:root { color-scheme: dark; }
.stApp { background: #0d1118; color: #f4f7fb; }
[data-testid="stSidebar"] { background: #171d29; border-right: 1px solid #293548; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px; }
.hero {
  padding: 2rem 1.7rem; border-radius: 22px;
  background: linear-gradient(135deg, #14283b 0%, #102d36 55%, #17213b 100%);
  border: 1px solid #2b5263; margin-bottom: 1.2rem;
}
.hero h1 { font-size: clamp(2rem, 5vw, 3.4rem); line-height: 1.12; margin: .3rem 0 1rem 0; }
.hero p { color: #c6d8e5; font-size: 1.08rem; max-width: 850px; }
.kicker { color: #54d6c4; text-transform: uppercase; letter-spacing: .13em; font-weight: 700; font-size: .8rem; }
.dev-card { background: #151d2a; border: 1px solid #2a3749; padding: 1.2rem; border-radius: 16px; }
.small-muted { color: #a9b8c8; font-size: .9rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def parse_smiles(smiles):
    if not smiles or not smiles.strip():
        return None, "Enter a SMILES string first."
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None, "This SMILES could not be parsed. Check the notation and try again."
    return mol, None

def properties(mol):
    return {
        "Molecular formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular weight (g/mol)": round(Descriptors.MolWt(mol), 4),
        "Exact molecular weight (Da)": round(Descriptors.ExactMolWt(mol), 5),
        "Atoms (including implicit H)": int(mol.GetNumAtoms() + sum(a.GetTotalNumHs() for a in mol.GetAtoms())),
        "Heavy atoms": int(mol.GetNumHeavyAtoms()),
        "Bonds": int(mol.GetNumBonds()),
        "Rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "H-bond donors": int(Lipinski.NumHDonors(mol)),
        "H-bond acceptors": int(Lipinski.NumHAcceptors(mol)),
        "TPSA (Å²)": round(Descriptors.TPSA(mol), 3),
        "Estimated LogP": round(Crippen.MolLogP(mol), 3),
        "Rotatable bonds": int(Lipinski.NumRotatableBonds(mol)),
        "Formal charge": int(Chem.GetFormalCharge(mol)),
    }

def create_2d_image(mol):
    copy = Chem.Mol(mol)
    Chem.rdDepictor.Compute2DCoords(copy)
    return Draw.MolToImage(copy, size=(850, 520), kekulize=True)

def create_3d_molblock(mol):
    """Return a 3D MOL block after conformer generation and force-field optimization."""
    work = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = 2026
    status = AllChem.EmbedMolecule(work, params)
    if status != 0:
        raise ValueError("RDKit could not generate 3D coordinates for this structure.")
    try:
        if AllChem.MMFFHasAllMoleculeParams(work):
            AllChem.MMFFOptimizeMolecule(work, maxIters=250)
        elif AllChem.UFFHasAllMoleculeParams(work):
            AllChem.UFFOptimizeMolecule(work, maxIters=250)
    except Exception:
        pass
    return Chem.MolToMolBlock(work), work

def show_3d_viewer(molblock, style, show_surface, height=520):
    # json.dumps safely serializes the molecular data before it is inserted into JavaScript.
    mol_json = json.dumps(molblock)
    style_map = {
        "Ball and stick": "{ stick: { radius: 0.16 }, sphere: { scale: 0.28 } }",
        "Stick": "{ stick: { radius: 0.18 } }",
        "Space-filling": "{ sphere: { scale: 0.85 } }",
        "Wireframe": "{ line: { linewidth: 2 } }",
    }
    style_js = style_map.get(style, style_map["Ball and stick"])
    surface_js = "viewer.addSurface($3Dmol.SurfaceType.VDW, {opacity: 0.72, colorscheme: 'Jmol'});" if show_surface else ""
    document = f"""
    <html><head>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
      <style>
        html, body {{ margin:0; padding:0; background:#ffffff; overflow:hidden; }}
        #viewer {{ width:100%; height:{height}px; position:relative; }}
        #fallback {{ padding:16px; font:14px sans-serif; color:#222; }}
      </style>
    </head><body>
      <div id="viewer"></div>
      <script>
        const el = document.getElementById('viewer');
        if (typeof $3Dmol === 'undefined') {{
          el.innerHTML = '<div id="fallback">The 3D viewer library could not load. Check your internet connection and reload.</div>';
        }} else {{
          const viewer = $3Dmol.createViewer(el, {{backgroundColor: 'white', antialias: true}});
          const molData = {mol_json};
          viewer.addModel(molData, 'mol');
          viewer.setStyle({{}}, {style_js});
          {surface_js}
          viewer.zoomTo();
          viewer.zoom(0.9);
          viewer.render();
          window.addEventListener('resize', () => viewer.resize());
        }}
      </script>
    </body></html>
    """
    components.html(document, height=height, scrolling=False)

def molecule_picker(label="Choose molecule", key="molecule_picker"):
    names = list(MOLECULES.keys())
    selected = st.selectbox(label, names, key=key)
    return selected, MOLECULES[selected]["smiles"]

def get_active():
    name = st.session_state.get("active_name", "Ethanol")
    smiles = st.session_state.get("active_smiles", MOLECULES["Ethanol"]["smiles"])
    mol, err = parse_smiles(smiles)
    return name, smiles, mol, err

def set_active(name, smiles):
    mol, err = parse_smiles(smiles)
    if err:
        st.error(err)
        return False
    st.session_state.active_name = name or "Custom molecule"
    st.session_state.active_smiles = smiles.strip()
    st.session_state.active_canonical = Chem.MolToSmiles(mol)
    st.success(f"Loaded valid structure: {st.session_state.active_name}")
    return True

if "active_name" not in st.session_state:
    st.session_state.active_name = "Ethanol"
    st.session_state.active_smiles = MOLECULES["Ethanol"]["smiles"]
    st.session_state.active_canonical = MOLECULES["Ethanol"]["smiles"]

with st.sidebar:
    st.markdown("## 🧪 ChemLab")
    st.caption("Virtual Chemoinformatics Laboratory")
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🧬 Molecular Lab", "📊 Properties", "📈 Compare Molecules", "📚 Theory", "📝 Assessment", "📄 Project Report", "👨‍💻 About"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("### 🧑‍🎓 Student project")
    st.caption("Developed by")
    st.markdown("**Sanjey Barathi M I**")
    st.caption("Student of Artificial Intelligence and Machine Learning (AI & ML)")
    st.divider()
    st.caption("Calculated molecular descriptors are educational estimates, not experimental measurements.")

name, smiles, active_mol, active_error = get_active()

if page == "🏠 Home":
    st.markdown("""
    <div class="hero">
      <div class="kicker">Interactive computational chemistry</div>
      <h1>🧪 ChemLab Virtual Laboratory</h1>
      <p>Explore molecular structures, calculate chemical descriptors, and inspect molecules in an interactive 3D viewer — all in one educational workspace.</p>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Example molecules", len(MOLECULES))
    c2.metric("Visualization modes", "2D + 3D")
    c3.metric("Core engine", "RDKit")
    st.subheader("Explore the laboratory")
    a, b, c = st.columns(3)
    with a:
        st.markdown("#### 🧬 Molecular visualization")
        st.write("Inspect 2D structures and rotate 3D molecular models.")
    with b:
        st.markdown("#### 📊 Molecular properties")
        st.write("Calculate molecular formula, weight, TPSA, LogP, and more.")
    with c:
        st.markdown("#### 📈 Compare structures")
        st.write("Compare molecular descriptors and fingerprint similarity.")
    st.subheader("Featured molecules")
    cols = st.columns(5)
    for col, mol_name in zip(cols, list(MOLECULES.keys())[:5]):
        with col:
            mol, _ = parse_smiles(MOLECULES[mol_name]["smiles"])
            st.image(create_2d_image(mol), use_container_width=True)
            st.markdown(f"**{mol_name}**")
            st.caption(rdMolDescriptors.CalcMolFormula(mol))
            if st.button("Load", key=f"home_load_{mol_name}", use_container_width=True):
                set_active(mol_name, MOLECULES[mol_name]["smiles"])
                st.rerun()
    st.divider()
    st.markdown('<div class="dev-card"><h3>👨‍💻 Developed by</h3><h2>Sanjey Barathi M I</h2><p>Student of Artificial Intelligence and Machine Learning (AI & ML)</p><p class="small-muted">An educational project combining computational chemistry, molecular visualization, and software development.</p></div>', unsafe_allow_html=True)

elif page == "🧬 Molecular Lab":
    st.title("🧬 Molecular Laboratory")
    st.info("Enter a SMILES string or load an example. The selected molecule is shared across the lab pages.")
    left, right = st.columns([1, 1])
    with left:
        current_input = st.text_input("Enter SMILES", value=smiles, key="smiles_input")
        example = st.selectbox("Choose example molecule", list(MOLECULES.keys()), index=list(MOLECULES.keys()).index(name) if name in MOLECULES else 2)
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Load example", use_container_width=True):
                set_active(example, MOLECULES[example]["smiles"])
                st.rerun()
        with b2:
            if st.button("Use entered SMILES", use_container_width=True):
                if set_active("Custom molecule", current_input):
                    st.rerun()
    name, smiles, active_mol, active_error = get_active()
    if active_error:
        st.error(active_error)
    else:
        st.success("Valid molecular structure")
        st.code(Chem.MolToSmiles(active_mol), language="text")
        st.subheader(f"2D molecular structure — {name}")
        st.image(create_2d_image(active_mol), use_container_width=True)
        st.subheader("Interactive 3D molecular structure")
        style = st.selectbox("Molecular model", ["Ball and stick", "Stick", "Space-filling", "Wireframe"])
        surface = st.checkbox("Overlay van der Waals surface (may be visually dense)")
        try:
            with st.spinner("Generating a 3D conformer..."):
                molblock, _ = create_3d_molblock(active_mol)
            show_3d_viewer(molblock, style, surface, height=520)
            st.caption("Drag to rotate; use the mouse wheel or pinch to zoom. The model is a computed conformer.")
        except Exception as exc:
            st.error(f"Could not generate a 3D model for this structure: {exc}")

elif page == "📊 Properties":
    st.title("📊 Molecular Properties Report")
    selected, selected_smiles = molecule_picker(key="properties_picker")
    if st.button("Load selected molecule", key="props_load"):
        set_active(selected, selected_smiles)
        st.rerun()
    name, smiles, active_mol, active_error = get_active()
    if active_mol is not None:
        st.subheader(f"{name} · {rdMolDescriptors.CalcMolFormula(active_mol)}")
        st.code(f"SMILES: {Chem.MolToSmiles(active_mol)}")
        props = properties(active_mol)
        metric_items = [
            ("Molecular weight", f'{props["Molecular weight (g/mol)"]} g/mol'),
            ("Formula", props["Molecular formula"]),
            ("Estimated LogP", str(props["Estimated LogP"])),
            ("TPSA", f'{props["TPSA (Å²)"]} Å²'),
        ]
        mcols = st.columns(4)
        for col, (label, value) in zip(mcols, metric_items):
            col.metric(label, value)
        st.dataframe(pd.DataFrame([{"Property": k, "Calculated value": v} for k, v in props.items()]), hide_index=True, use_container_width=True)
        st.caption("LogP is a model-based estimate. Other calculated descriptors are not a substitute for experimental measurements.")

elif page == "📈 Compare Molecules":
    st.title("📈 Structure–Property Analysis")
    choices = st.multiselect("Choose molecules to compare", list(MOLECULES.keys()), default=["Ethanol", "Benzene", "Caffeine"])
    if len(choices) >= 2:
        rows = []
        mols = []
        for mol_name in choices:
            mol, err = parse_smiles(MOLECULES[mol_name]["smiles"])
            if mol:
                p = properties(mol)
                rows.append({"Molecule": mol_name, **p})
                mols.append((mol_name, mol))
        df = pd.DataFrame(rows)
        numeric_options = ["Molecular weight (g/mol)", "TPSA (Å²)", "Estimated LogP", "H-bond donors", "H-bond acceptors", "Rotatable bonds", "Rings"]
        prop = st.selectbox("Property to visualize", numeric_options)
        fig = px.bar(df, x="Molecule", y=prop, color="Molecule", title=f"{prop} comparison", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, hide_index=True, use_container_width=True)
        if len(mols) >= 2:
            fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            st.subheader("Fingerprint similarity")
            pairs = []
            for i in range(len(mols)):
                for j in range(i + 1, len(mols)):
                    fp1 = fpgen.GetFingerprint(mols[i][1])
                    fp2 = fpgen.GetFingerprint(mols[j][1])
                    pairs.append({"Molecule A": mols[i][0], "Molecule B": mols[j][0], "Tanimoto similarity": round(DataStructs.TanimotoSimilarity(fp1, fp2), 3)})
            st.dataframe(pd.DataFrame(pairs), hide_index=True, use_container_width=True)
            st.caption("Morgan-fingerprint Tanimoto similarity ranges from 0 to 1; interpretation depends on the structures and fingerprint settings.")

elif page == "📚 Theory":
    st.title("📚 Chemoinformatics Theory")
    topics = {
        "What is chemoinformatics?": "Chemoinformatics applies computational and information-science methods to chemical data, including molecular representation, search, prediction, and analysis.",
        "SMILES notation": "SMILES (Simplified Molecular Input Line Entry System) is a text representation of molecular connectivity. For example, CCO represents ethanol. Aromatic atoms are commonly written in lowercase, as in c1ccccc1 for benzene.",
        "Molecular descriptors": "Descriptors summarize aspects of a molecule numerically. Examples include molecular weight, topological polar surface area (TPSA), estimated LogP, hydrogen-bond counts, and rotatable bonds.",
        "2D versus 3D structures": "A 2D drawing emphasizes connectivity and bond types. A 3D conformer adds spatial coordinates and can help illustrate shape, but one generated conformer does not necessarily represent the only or experimentally preferred geometry.",
        "Fingerprints and similarity": "Molecular fingerprints encode structural patterns as features. Similarity metrics such as Tanimoto similarity can compare fingerprints, but they do not by themselves prove similar biological activity.",
        "Structure–property relationships": "Molecular structure influences properties through factors such as functional groups, size, flexibility, charge, and polarity. Computed descriptors are useful for exploration but should not be treated as experimental data."
    }
    for title, body in topics.items():
        with st.expander(title):
            st.write(body)
    st.info("All content is intended for education. Medicinal examples are not medical advice.")

elif page == "📝 Assessment":
    st.title("📝 Chemoinformatics Assessment")
    questions = [
        ("Which SMILES string represents ethanol in this application?", ["CCO", "c1ccccc1", "O=C=O", "N"], 0, "CCO represents a two-carbon alcohol, ethanol."),
        ("Which molecule contains an aromatic six-membered carbon ring?", ["Methane", "Benzene", "Water", "Ethanol"], 1, "Benzene has an aromatic six-membered carbon ring."),
        ("What does TPSA estimate?", ["Polar surface area", "Boiling point exactly", "Colour", "Density experimentally"], 0, "TPSA is a calculated topological polar surface area descriptor."),
        ("Which group is characteristic of carboxylic acids?", ["-COOH", "-CH3 only", "-Cl only", "-NH2 only"], 0, "Carboxylic acids contain a carboxyl group."),
        ("What does a 3D conformer provide?", ["Spatial coordinates", "A guaranteed experimental structure", "A measured melting point", "A guaranteed drug effect"], 0, "A conformer provides a possible 3D arrangement of atoms."),
        ("Which toolkit calculates descriptors in this app?", ["RDKit", "Spreadsheet formulas only", "A web browser alone", "A text editor"], 0, "RDKit provides cheminformatics functions."),
        ("What is the usual range of Tanimoto similarity?", ["0 to 1", "1 to 1000", "-100 to 100", "Only 0"], 0, "For binary fingerprints, Tanimoto similarity ranges from 0 to 1."),
        ("Which is a molecular descriptor?", ["Molecular weight", "Browser brightness", "Screen resolution", "File name"], 0, "Molecular weight is a commonly used molecular descriptor."),
        ("What is a SMILES string?", ["Text representation of molecular structure", "A microscope", "A type of bond energy measurement", "A lab safety label"], 0, "SMILES encodes molecular connectivity as text."),
        ("Does a computed descriptor always equal an experimental measurement?", ["Yes, exactly", "No, it is a calculation or estimate", "Only for caffeine", "Only in 3D"], 1, "Calculated descriptors and experimentally measured properties are different kinds of information."),
    ]
    with st.form("assessment_form"):
        answers = {}
        for i, (question, options, correct, explanation) in enumerate(questions):
            answers[i] = st.radio(f"{i+1}. {question}", options, key=f"assessment_{i}")
        submitted = st.form_submit_button("Submit assessment")
    if submitted:
        score = sum(1 for i, (_, options, correct, _) in enumerate(questions) if answers[i] == options[correct])
        st.session_state.assessment_score = score
        st.session_state.assessment_answers = answers
        st.success(f"Your score: {score}/{len(questions)} ({score/len(questions)*100:.0f}%)")
        for i, (question, options, correct, explanation) in enumerate(questions):
            is_correct = answers[i] == options[correct]
            st.markdown(f"**{i+1}. {'✅ Correct' if is_correct else '❌ Incorrect'}** — {explanation}")
        if st.button("Try again"):
            st.rerun()

elif page == "📄 Project Report":
    st.title("📄 Downloadable Molecular Report")
    name, smiles, active_mol, active_error = get_active()
    if active_mol is not None:
        st.subheader(name)
        st.write("This report summarizes computed molecular descriptors for the currently selected molecule.")
        props = properties(active_mol)
        st.image(create_2d_image(active_mol), use_container_width=True)
        st.dataframe(pd.DataFrame([{"Property": k, "Value": v} for k, v in props.items()]), hide_index=True, use_container_width=True)
        st.code(f"SMILES: {Chem.MolToSmiles(active_mol)}")
        csv_bytes = pd.DataFrame([{"Molecule": name, "SMILES": Chem.MolToSmiles(active_mol), **props}]).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download properties as CSV", csv_bytes, file_name="chemlab_molecular_properties.csv", mime="text/csv")
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import inch
            pdf_buffer = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)
            styles = getSampleStyleSheet()
            story = [Paragraph("ChemLab Molecular Report", styles["Title"]), Spacer(1, 10),
                     Paragraph(f"<b>Molecule:</b> {html.escape(name)}", styles["Heading2"]),
                     Paragraph(f"<b>SMILES:</b> {html.escape(Chem.MolToSmiles(active_mol))}", styles["BodyText"]),
                     Spacer(1, 12)]
            img_buf = io.BytesIO()
            create_2d_image(active_mol).save(img_buf, format="PNG")
            img_buf.seek(0)
            story.append(RLImage(img_buf, width=5.8*inch, height=3.5*inch))
            story.append(Spacer(1, 10))
            table_data = [["Property", "Calculated value"]] + [[str(k), str(v)] for k, v in props.items()]
            table = Table(table_data, colWidths=[3.2*inch, 2.4*inch], repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#183b4a")),
                ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#eef4f7")]),
            ]))
            story.extend([table, Spacer(1, 12),
                          Paragraph("Note: Values are calculated molecular descriptors for educational use and are not experimental measurements or medical advice.", styles["Italic"]),
                          Spacer(1, 8), Paragraph("Developed by Sanjey Barathi M I — Student of Artificial Intelligence and Machine Learning (AI & ML)", styles["BodyText"])])
            doc.build(story)
            st.download_button("⬇️ Download PDF report", pdf_buffer.getvalue(), file_name="chemlab_molecular_report.pdf", mime="application/pdf")
        except Exception as exc:
            st.warning(f"PDF generation is unavailable: {exc}. CSV export is still available.")

elif page == "👨‍💻 About":
    st.title("👨‍💻 About ChemLab")
    st.markdown("""
    <div class="hero">
      <div class="kicker">Student project</div>
      <h1>ChemLab</h1>
      <p>An educational virtual laboratory for exploring chemical structures and computational molecular descriptors.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    ### Developed by
    # Sanjey Barathi M I
    **Student of Artificial Intelligence and Machine Learning (AI & ML)**

    This educational project demonstrates the application of computational chemistry, scientific visualization, and software development in an interactive learning environment.
    """)
    st.caption("Built with Streamlit, RDKit, Plotly, and 3Dmol.js.")
    st.warning("Scientific note: molecular descriptors and generated conformers are computational results. They should not be confused with experimentally measured properties.")
