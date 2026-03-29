import streamlit as st
import py3Dmol
from stmol import showmol
from molecule_generator import generate_aa_conformer, get_pdb_string, AA_SMILES
from fragment_mapper import partition_fragments
from explode_alg import apply_explosion

TOKEN_TO_FORMULA = {
    "METHYL": "-CH3",
    "METHYLENE": "-CH2-",
    "HYDROXYL": "-OH",
    "PHENYL": "-C6H5",
    "PHENYLENE": "-C6H4-",
    "AMINE": "-NH3+",
    "CARBOXYL": "-COO-",
    "AMIDE": "-CONH2",
    "GUANIDINE": "-NH-C(NH2)=NH+",
    "IMIDAZOLE": "Imidazole (C3H3N2)",
    "INDOLE": "Indole (C8H6N)",
    "THIOL": "-SH",
    "THIOETHER": "-S-",
    "PYRROLIDINE_RING": "Pyrrolidine Ring",
    "BRANCH_CH": ">CH-"
}

st.set_page_config(layout="wide", page_title="Protein Side-chain Fragmentizer")

st.title("🧩 原子级自适应蛋白质侧链设计 - Fragment 可视化")
st.markdown("通过 RDKit 渲染原始 3D 侧链，并根据 `vocabulary.py` 规则展现 '由易到难' 的拆解碎片模型。")

st.sidebar.header("控制台")
aa_choice = st.sidebar.selectbox("选择氨基酸", list(AA_SMILES.keys()), index=9)
explode_dist = st.sidebar.slider("碎片独立断开距离（可选）", min_value=0.0, max_value=5.0, value=1.5, step=0.1)

show_group_labels = st.sidebar.checkbox("显示化学式标记", value=True)
show_atom_labels = st.sidebar.checkbox("显示原子元素符号", value=False)

try:
    mol = generate_aa_conformer(aa_choice)
    partials = partition_fragments(mol, aa_choice)
    mol_exploded = apply_explosion(mol, partials, explode_factor=explode_dist)
    
    pdb_original = get_pdb_string(mol)
    pdb_exploded = get_pdb_string(mol_exploded)
    
except Exception as e:
    st.error(f"处理化学结构时出错: {e}")
    st.stop()

col1, col2 = st.columns(2)

def build_3d_viewer(pdb_data, width=450, height=450):
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_data, 'pdb')
    view.setBackgroundColor('white')
    return view

with col1:
    st.subheader(f"完整连通氨基酸构象 (Original: {aa_choice})")
    view_orig = build_3d_viewer(pdb_original)
    
    # 采用球棍模型（类似你附图的风格：原子是圆球，键是粗柱子）
    view_orig.setStyle({'stick': {'colorscheme': 'magentaCarbon', 'radius': 0.15}, 'sphere': {'colorscheme': 'magentaCarbon', 'radius': 0.4}})
    view_orig.zoomTo()
    showmol(view_orig, height=450, width=450)

with col2:
    st.subheader(f"球棍模型 Fragment 分组渲染")
    view_expl = build_3d_viewer(pdb_exploded)
    
    colors = ['cyan', 'orange', 'hotpink', 'lime', 'yellow', 'purple', 'teal']
    for idx, part in enumerate(partials):
        color = colors[idx % len(colors)]
        for atom_i in part["atoms"]:
            # 使用经典的球棍结合风格（半径适中模拟出原子球相连的感觉）
            view_expl.setStyle({'serial': atom_i}, {'stick': {'color': color, 'radius': 0.2}, 'sphere': {'radius': 0.5, 'color': color}})
            
            if show_atom_labels:
                atom = mol_exploded.GetAtomWithIdx(atom_i)
                symbol = atom.GetSymbol()
                view_expl.addLabel(symbol, 
                                 {'fontColor': 'black', 'fontSize': 12, 'backgroundColor': 'rgba(255,255,255,0.5)', 'backgroundOpacity': 0.5}, 
                                 {'serial': atom_i})

        if show_group_labels and part["atoms"]:
            center_atom_idx = part["atoms"][len(part["atoms"])//2]
            formula = TOKEN_TO_FORMULA.get(part['token'], part['token'])
            view_expl.addLabel(f" {formula} ", 
                             {'fontColor': 'white', 'fontSize': 14, 'backgroundColor': 'black', 'backgroundOpacity': 0.7, 'inFront': True}, 
                             {'serial': center_atom_idx})
            
    # 主链骨架设置为暗灰色，也用球棍模型以匹配整体风格
    for i in range(mol.GetNumAtoms()):
        is_sidechain = False
        for part in partials:
            if i in part["atoms"]:
                is_sidechain = True
                break
        if not is_sidechain:
            view_expl.setStyle({'serial': i}, {'stick': {'color': 'dimGray', 'radius': 0.15}, 'sphere': {'radius': 0.35, 'color': 'dimGray'}})
            if show_atom_labels:
                sym = mol_exploded.GetAtomWithIdx(i).GetSymbol()
                view_expl.addLabel(sym, {'fontColor': 'gray', 'fontSize': 10, 'backgroundColor': 'transparent'}, {'serial': i})
            
    view_expl.zoomTo()
    showmol(view_expl, height=450, width=450)

st.divider()
st.subheader("底层 Vocabulary 映射关系")
if partials:
    st.write(f"在我们的词表(`vocabulary.py`)中，**{aa_choice}** 被分解成了 **{len(partials)}** 个 Token：")
    for part in partials:
        formula = TOKEN_TO_FORMULA.get(part['token'], part['token'])
        st.info(f"🧩 Token: `{part['token']}` ➡  **{formula}**  (关联 {len(part['atoms'])} 个侧链原子)")
else:
    st.info(f"{aa_choice} 没有侧链 Fragment 分配。")
