"""
SmartParkTN – Streamlit Dashboard
Run: streamlit run ui/dashboard.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import cv2
import numpy as np
import base64
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_BASE = f"http://localhost:{os.getenv('API_PORT', 8000)}/api/v1"

st.set_page_config(
    page_title="SmartParkTN",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%);
    padding: 1rem; border-radius: 12px; color: white;
    text-align: center; margin: 0.3rem;
}
.allowed  { color: #00e676; font-weight: bold; }
.denied   { color: #ff5252; font-weight: bold; }
.pending  { color: #ffd740; font-weight: bold; }
.plate-display {
    font-family: monospace; font-size: 2rem;
    background: #fff; color: #000;
    padding: 0.5rem 1.5rem; border-radius: 8px;
    border: 4px solid #1e3a5f; display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────
def api_get(path, default=None):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return default

def api_post(path, data=None, files=None):
    try:
        if files:
            r = requests.post(f"{API_BASE}{path}", data=data, files=files, timeout=15)
        else:
            r = requests.post(f"{API_BASE}{path}", json=data, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# ── Sidebar ────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/car--v1.png", width=80)
st.sidebar.title("SmartParkTN")
st.sidebar.markdown("Système ALPR Tunisien")
page = st.sidebar.radio("Navigation", [
    "📊 Tableau de bord",
    "📷 Détection en direct",
    "🚗 Véhicules",
    "📋 Événements",
    "💬 Assistant IA",
    "⚙️ Paramètres",
])

# ── Check API ──────────────────────────────────────────────────────────
health = api_get("/health")
api_ok = health is not None and "ok" in str(health.get("status", ""))
st.sidebar.markdown(
    "🟢 **API connectée**" if api_ok else "🔴 **API hors ligne**\n\n`uvicorn main:app`"
)

# ─────────────────────────────────────────────────────────────────────
if "📊" in page:
    st.title("📊 Tableau de bord – SmartParkTN")
    events = api_get("/events?limit=500", [])
    vehicles = api_get("/vehicles", [])
    tariffs  = api_get("/tariffs", [])

    df = pd.DataFrame(events)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🚗 Total événements", len(df))
    with col2:
        n_allowed = len(df[df.decision == "allowed"]) if not df.empty else 0
        st.metric("✅ Accès autorisés", n_allowed)
    with col3:
        n_denied = len(df[df.decision == "denied"]) if not df.empty else 0
        st.metric("🚫 Accès refusés", n_denied)
    with col4:
        rev = df.amount_tnd.sum() if not df.empty and "amount_tnd" in df.columns else 0
        st.metric("💰 Revenus (TND)", f"{rev:.2f}")

    # Currently parked count
    stats = api_get("/stats/summary", {})
    col5, col6, col7 = st.columns(3)
    with col5:
        st.metric("🅿️ Véhicules présents", stats.get("currently_parked", "—"))
    with col6:
        st.metric("📅 Événements aujourd'hui", stats.get("events_today", "—"))
    with col7:
        rev_today = stats.get("revenue_today", 0)
        st.metric("💵 Revenus aujourd'hui", f"{rev_today:.2f} TND")

    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(df, names="category", title="Catégories de véhicules",
                         color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.histogram(df, x="hour", title="Trafic par heure",
                                nbins=24, color_discrete_sequence=["#1e88e5"])
            fig2.update_layout(xaxis_title="Heure", yaxis_title="Véhicules")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Derniers événements")
        disp = df.sort_values("timestamp", ascending=False).head(20)[[
            "timestamp", "plate", "category", "type", "decision", "amount_tnd", "reason"
        ]]
        st.dataframe(disp, use_container_width=True)
    else:
        st.info("Aucune donnée. Lancez la détection pour voir des statistiques.")

    st.subheader("Tarifs en vigueur")
    if tariffs:
        st.dataframe(pd.DataFrame(tariffs), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
elif "📷" in page:
    st.title("📷 Détection ALPR en direct")

    mode = st.radio("Source", ["📁 Téléverser une image", "🎥 Webcam en direct"])
    camera_id = st.selectbox("Caméra", ["CAM_ENTRY_01", "CAM_EXIT_01",
                                         "CAM_ENTRY_02", "CAM_EXIT_02"])

    if "📁" in mode:
        uploaded = st.file_uploader("Image véhicule / plaque", type=["jpg","jpeg","png"])
        if uploaded:
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(uploaded, caption="Image originale", use_column_width=True)
            with col_b:
                with st.spinner("Analyse en cours…"):
                    result = api_post(
                        "/process-image",
                        data={"camera_id": camera_id},
                        files={"file": (uploaded.name, uploaded.getvalue(), "image/jpeg")},
                    )
                if "error" not in result:
                    if result.get("annotated_image_b64"):
                        img_bytes = base64.b64decode(result["annotated_image_b64"])
                        st.image(img_bytes, caption="Plaque détectée", use_column_width=True)

                    plate = result.get("plate", "—")
                    dec   = result.get("decision", "pending")
                    color_map = {"allowed": "🟢", "denied": "🔴", "pending": "🟡"}
                    emoji = color_map.get(dec, "🟡")

                    st.markdown(f"### {emoji} Décision : **{dec.upper()}**")
                    st.markdown(f"<div class='plate-display'>{plate}</div>",
                                unsafe_allow_html=True)
                    cols = st.columns(3)
                    cols[0].metric("Catégorie", result.get("category", "—"))
                    cols[1].metric("Confiance OCR", f"{result.get('confidence', 0):.0%}")
                    cols[2].metric("Confiance Détection", f"{result.get('detect_conf', 0):.0%}")

                    if result.get("amount_tnd") is not None:
                        st.success(f"Durée: {result['duration_min']:.0f} min | "
                                   f"Montant: {result['amount_tnd']:.2f} TND")
                    st.info(f"Raison: {result.get('reason', '—')}")
                else:
                    st.error(f"Erreur: {result['error']}")

    else:  # Webcam
        st.warning("Assurez-vous que la caméra est accessible (cv2.VideoCapture(0)).")
        run = st.checkbox("▶ Activer la webcam")
        frame_ph = st.empty()
        result_ph = st.empty()

        if run:
            cap = cv2.VideoCapture(0)
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Impossible d'ouvrir la caméra."); break
                _, buf = cv2.imencode(".jpg", frame)
                b64 = base64.b64encode(buf).decode()
                frame_ph.image(frame[:, :, ::-1], channels="RGB",
                               caption="Flux caméra", use_column_width=True)
                result = api_post("/process-frame",
                                  {"frame_b64": b64, "camera_id": camera_id})
                if result.get("plate"):
                    result_ph.success(
                        f"🎯 {result['plate']} | {result['category']} | {result['decision']}"
                    )
                time.sleep(0.5)
            cap.release()

# ─────────────────────────────────────────────────────────────────────
elif "🚗" in page:
    st.title("🚗 Gestion des Véhicules")
    tab1, tab2, tab3 = st.tabs(["Liste véhicules", "Ajouter / Modifier", "🔑 Abonnements"])

    with tab1:
        vehicles = api_get("/vehicles", [])
        if vehicles:
            df_v = pd.DataFrame(vehicles)
            search = st.text_input("🔍 Rechercher par plaque")
            if search:
                df_v = df_v[df_v["plate"].str.contains(search.upper(), na=False)]
            st.dataframe(df_v, use_container_width=True)
            # Quick lookup
            st.subheader("Vérification d'accès rapide")
            plate_check = st.text_input("Entrer une plaque")
            if plate_check:
                detail = api_get(f"/vehicles/{plate_check.upper()}")
                if detail:
                    dec = detail.get("access", "pending")
                    emoji = {"allowed": "✅", "denied": "❌", "pending": "⏳"}.get(dec, "❓")
                    st.markdown(f"{emoji} **{plate_check.upper()}** – {dec.upper()}")
                    st.write(detail)
        else:
            st.info("Aucun véhicule enregistré.")

    with tab2:
        with st.form("add_vehicle"):
            st.subheader("Enregistrer un véhicule")
            plate = st.text_input("Plaque (ex: 100 TN 1234)").upper()
            owner = st.text_input("Propriétaire")
            cat   = st.selectbox("Catégorie",
                ["visitor","subscriber","vip","employee","blacklist","emergency"])
            notes = st.text_area("Notes")
            if st.form_submit_button("Enregistrer"):
                r = api_post("/vehicles", {"plate": plate, "owner_name": owner,
                                           "category": cat, "notes": notes})
                if "error" not in r:
                    st.success(f"Véhicule {plate} enregistré ✓")
                else:
                    st.error(str(r))

    with tab3:
        st.subheader("Abonnements actifs")
        subs = api_get("/subscriptions", [])
        if subs:
            df_s = pd.DataFrame(subs)
            df_s["end_date"] = pd.to_datetime(df_s["end_date"])
            df_s["expires_in"] = (df_s["end_date"] - pd.Timestamp.now()).dt.days
            df_s["statut"] = df_s.apply(
                lambda r: "✅ Actif" if r["active"] and r["expires_in"] >= 0
                else ("⚠️ Expiré" if r["expires_in"] < 0 else "❌ Annulé"), axis=1
            )
            st.dataframe(df_s[["id","plate","zone","start_date","end_date","expires_in","statut"]],
                         use_container_width=True)
            cancel_id = st.number_input("ID abonnement à annuler", min_value=1, step=1)
            if st.button("❌ Annuler cet abonnement"):
                import requests as _req
                rd = _req.delete(f"{API_BASE}/subscriptions/{int(cancel_id)}", timeout=5)
                if rd.ok:
                    st.success("Abonnement annulé ✓")
                    st.rerun()
                else:
                    st.error(rd.text)
        else:
            st.info("Aucun abonnement enregistré.")

        st.divider()
        st.subheader("Créer un abonnement")
        with st.form("add_sub"):
            sub_plate = st.text_input("Plaque (ex: 100 TN 1234)").upper()
            sub_zone  = st.selectbox("Zone", ["A", "B", "A+B", "VIP"])
            col_s, col_e = st.columns(2)
            sub_start = col_s.date_input("Date début")
            sub_end   = col_e.date_input("Date fin")
            if st.form_submit_button("Créer l'abonnement"):
                r = api_post("/subscriptions", {
                    "plate": sub_plate,
                    "start_date": sub_start.isoformat(),
                    "end_date":   sub_end.isoformat(),
                    "zone": sub_zone,
                })
                if "error" not in r and "id" in r:
                    st.success(f"Abonnement #{r['id']} créé pour {sub_plate} ✓")
                    st.rerun()
                else:
                    st.error(str(r))

# ─────────────────────────────────────────────────────────────────────
elif "📋" in page:
    st.title("📋 Historique des Événements")
    limit = st.slider("Nombre d'événements", 50, 500, 100)
    events = api_get(f"/events?limit={limit}", [])
    if events:
        df_e = pd.DataFrame(events)
        # Filter
        cats = ["Toutes"] + list(df_e["category"].unique()) if "category" in df_e.columns else ["Toutes"]
        cat_f = st.selectbox("Filtrer par catégorie", cats)
        dec_f = st.selectbox("Filtrer par décision", ["Toutes", "allowed", "denied", "pending"])
        if cat_f != "Toutes":
            df_e = df_e[df_e["category"] == cat_f]
        if dec_f != "Toutes":
            df_e = df_e[df_e["decision"] == dec_f]

        st.dataframe(df_e, use_container_width=True)
        # Export CSV
        csv = df_e.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Télécharger CSV", csv, "events.csv", "text/csv")
    else:
        st.info("Aucun événement enregistré.")

# ─────────────────────────────────────────────────────────────────────
elif "💬" in page:
    st.title("💬 Assistant IA – SmartParkTN")
    st.caption("Posez vos questions sur les règles, tarifs et procédures.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content":
             "Bonjour ! Je suis l'assistant SmartParkTN. "
             "Posez-moi vos questions sur les règlements, tarifs et procédures."}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Votre question…")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Recherche en cours…"):
                result = api_post("/assistant/ask", {"question": question})
            answer = result.get("answer", "Désolé, une erreur est survenue.")
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    st.divider()
    st.subheader("Expliquer une décision")
    col1, col2, col3 = st.columns(3)
    exp_plate = col1.text_input("Plaque", key="exp_plate")
    exp_dec   = col2.selectbox("Décision", ["denied", "allowed"], key="exp_dec")
    exp_reason= col3.text_input("Raison", key="exp_reason")
    if st.button("Expliquer"):
        r = api_post("/assistant/explain",
                     {"plate": exp_plate, "decision": exp_dec, "reason": exp_reason})
        st.info(r.get("explanation", "Erreur"))

# ─────────────────────────────────────────────────────────────────────
elif "⚙️" in page:
    st.title("⚙️ Paramètres")
    st.subheader("Ingestion des règles (RAG)")
    if st.button("🔄 Réingérer les documents"):
        r = api_post("/assistant/ingest")
        st.success(str(r))

    st.subheader("Base de données")
    if st.button("🌱 Seeder les véhicules de test"):
        import subprocess
        subprocess.run(["python", "scripts/seed_vehicles.py"], cwd="..")
        st.success("Seeding lancé !")

    st.subheader("Configuration API")
    st.code(f"API_BASE = {API_BASE}", language="text")
    st.json(health or {"status": "offline"})
